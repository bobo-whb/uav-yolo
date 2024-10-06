import os
import time

import cv2
import torch
import torch.nn.functional as F
from ultralytics import YOLO
from ultralytics.engine.results import Boxes
from ultralytics.trackers import BYTETracker, BOTSORT
from ultralytics.utils import IterableSimpleNamespace, yaml_load
from ultralytics.utils.checks import check_yaml

from .nms import GreedyNMMPostprocess


def calculate_overlap(size, slice_size, num_slices):
    if num_slices == 1:
        return 0
    overlap = (num_slices * slice_size - size) / (num_slices - 1)
    return overlap


def split_image(image, slice_height, slice_width, height_slices_num, width_slices_num, auto_overlap,
                overlap_height_ratio, overlap_width_ratio, device, use_float16, target_size):
    # 将图像转换为PyTorch张量并归一化
    if use_float16:
        image = torch.tensor(image, dtype=torch.float16).permute(2, 0, 1).unsqueeze(0).to(device) / 255.0
    else:
        image = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(device) / 255.0  # 1, 3, H, W

    # 获取图像的高度和宽度
    _, _, height, width = image.shape
    slice_height = slice_height or height
    slice_width = slice_width or width

    # 缩放图像到与图像块相同的大小
    scaled_image = F.interpolate(image, size=(slice_height, slice_width), mode='bilinear', align_corners=False)

    # 计算重叠
    if auto_overlap:
        overlap_height = int(calculate_overlap(height, slice_height, height_slices_num))
        overlap_width = int(calculate_overlap(width, slice_width, width_slices_num))
    else:
        overlap_height = int(slice_height * overlap_height_ratio)
        overlap_width = int(slice_width * overlap_width_ratio)

    # 使用unfold函数提取图像块
    chunks = image.unfold(2, slice_height, slice_height - overlap_height).unfold(3, slice_width,
                                                                                 slice_width - overlap_width)
    n, c, h, w, slice_h, slice_w = chunks.shape
    chunks = chunks.permute(0, 2, 3, 1, 4, 5).contiguous().view(n * h * w, c, slice_h, slice_w)

    # 将缩放的图像添加到0维度
    chunks = torch.cat((scaled_image, chunks), dim=0)

    # 生成坐标
    coords_x = torch.arange(0, w * (slice_width - overlap_width), slice_width - overlap_width, device=device).repeat(h)
    coords_y = torch.arange(0, h * (slice_height - overlap_height), slice_height - overlap_height,
                            device=device).repeat_interleave(w)
    coords = torch.stack((coords_x, coords_y), dim=1).float()

    # 如果指定了target_size，进行分片缩放
    if target_size:
        target_height, target_width = target_size

        # 扩展缩放比例以匹配chunks的数量
        scale_factors = torch.tensor([target_height / slice_height, target_width / slice_width], device=device).repeat(
            chunks.size(0) - 1, 1)

        # 缩放chunks
        chunks = F.interpolate(chunks, size=(target_height, target_width), mode='bilinear', align_corners=False)
        # 计算原图缩放比例
        scale_factor_height = target_height / height
        scale_factor_width = target_width / width
    else:
        scale_factors = torch.tensor([1, 1], device=device).repeat(chunks.size(0) - 1, 1)
        # 计算原图缩放比例
        scale_factor_height = slice_height / height
        scale_factor_width = slice_width / width

    scale_factors = torch.cat((torch.tensor([[scale_factor_height, scale_factor_width]], device=device), scale_factors),
                              dim=0)
    # 返回图像块、坐标和缩放比例
    return chunks, coords, scale_factors


def adjust_boxes(boxes, chunk_coords, scale_factors, is_original=False, class_mapping=None):
    adjusted_boxes = []
    x_offset, y_offset = chunk_coords
    scale_factor_height, scale_factor_width = scale_factors

    for box in boxes:
        x1, y1, x2, y2 = box.xyxy[0].tolist()

        # 先应用缩放因子，再加上偏移量
        x1 = x1 / scale_factor_width + x_offset
        y1 = y1 / scale_factor_height + y_offset
        x2 = x2 / scale_factor_width + x_offset
        y2 = y2 / scale_factor_height + y_offset
        class_num = box.cls[0].clone()

        if class_mapping:
            class_num_int = class_mapping.get(int(class_num), int(class_num))
            class_num = torch.tensor(class_num_int, device=class_num.device)

        if is_original:
            score = box.conf[0].clone()
            score += 0.2
            if score > 1.0:
                score = torch.tensor(1.0, device=score.device)
            adjusted_boxes.append([x1, y1, x2, y2, score, class_num])
        else:
            adjusted_boxes.append([x1, y1, x2, y2, box.conf[0], class_num])

    return adjusted_boxes


def init_boxes_class(boxes, orig_shape):
    boxes_list = []
    for box in boxes:
        x1, y1, x2, y2, conf, cls = box
        boxes_list.append([x1, y1, x2, y2, conf, cls])
    boxes_list = torch.tensor(boxes_list)
    return Boxes(boxes_list, (orig_shape[0], orig_shape[1]))


def process_output(boxes, tracks):
    track_dict = {int(track[-1]): int(track[4]) for track in tracks}
    updated_boxes = [box + [track_dict.get(i, None)] for i, box in enumerate(boxes)]
    return updated_boxes


def draw_boxes(image, boxes):
    if boxes is None:
        return
    for box in boxes:
        x1, y1, x2, y2, conf, cls, track_id = box
        label = f"{int(cls)}: {conf:.2f}"
        if track_id is not None:
            cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(image, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            label = f"ID: {int(track_id)}"
            cv2.putText(image, label, (int(x1), int(y2) + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)


class YOLOVideoProcessor:
    def __init__(self, model_path, tracker_path):
        os.environ['CUDA_MODULE_LOADING'] = 'LAZY'
        self.model = YOLO(model_path, task="detect")
        tracker = check_yaml(tracker_path)
        cfg = IterableSimpleNamespace(**yaml_load(tracker))
        tracker_map = {"bytetrack": BYTETracker, "botsort": BOTSORT}
        self.tracker = tracker_map[cfg.tracker_type](args=cfg, frame_rate=30)

    def process_frame(self, frame, slice_height=640, slice_width=640, auto_overlap=True, overlap_height_ratio=0.2,
                      overlap_width_ratio=0.2, device='cuda', use_float16=False, class_mapping=None):
        orig_shape = frame.shape
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        chunks, coords, scale_factor = split_image(frame, slice_height, slice_width, 1, 2, auto_overlap,
                                                   overlap_height_ratio, overlap_width_ratio, device, use_float16,
                                                   (640, 640))
        results = self.model.predict(chunks, conf=0.6)
        all_boxes = []

        for index, result in enumerate(results):
            if index != 0:
                adjusted_boxes = adjust_boxes(result.boxes, coords[index - 1], scale_factor[index],
                                              class_mapping=class_mapping)
            else:
                adjusted_boxes = adjust_boxes(result.boxes, (0, 0), scale_factor[index], is_original=True,
                                              class_mapping=class_mapping)
            all_boxes.extend(adjusted_boxes)

        if len(all_boxes) > 0:
            all_boxes_tensor = torch.tensor(all_boxes)
            final_boxes = GreedyNMMPostprocess(all_boxes_tensor, match_threshold=0.5, match_metric="IOS")
            boxes_class = init_boxes_class(final_boxes, orig_shape)
            tracks = self.tracker.update(boxes_class, frame)
            return process_output(final_boxes, tracks)
        return None

    def process_video(self, video_path, slice_height=None, slice_width=None, auto_overlap=False,
                      overlap_height_ratio=0.2, overlap_width_ratio=0.2, output_resolution=None, class_mapping=None):
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        delay = int(1000 / fps)

        prev_time = 0
        fps_display = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if output_resolution:
                frame = cv2.resize(frame, output_resolution)

            processed_frame = self.process_frame(frame, slice_height, slice_width, auto_overlap,
                                                 overlap_height_ratio, overlap_width_ratio, class_mapping=class_mapping)
            draw_boxes(frame, processed_frame)

            # 计算FPS
            curr_time = time.time()
            time_diff = curr_time - prev_time
            if time_diff > 0:
                fps_display = 1 / time_diff
            prev_time = curr_time

            # 在帧上绘制FPS
            cv2.putText(frame, f'FPS: {fps_display:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            cv2.imshow('Processed Video', frame)
            if cv2.waitKey(delay) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    class_mapping_dict = {
        3: 2,
        6: 5, 7: 5,
        9: 8,
        12: 11, 13: 11, 14: 11,
        17: 16,
        19: 18,
        25: 24, 26: 24, 27: 24, 28: 24, 29: 24,
        32: 31, 33: 31, 34: 31, 35: 31, 36: 31
    }
    processor = YOLOVideoProcessor("ckpts/yolov8.engine", "./UltralyticsYOLO/cfg/bytetrack.yaml")
    processor.process_video("2025.mp4", slice_height=720, slice_width=720,
                            auto_overlap=True, output_resolution=(1280, 720), class_mapping=class_mapping_dict)
