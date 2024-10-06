import datetime
import json
import os
import queue
import threading
from datetime import timedelta
import cv2
import numpy as np
from minio import Minio
from minio.error import S3Error
from engine.predictor import YOLOVideoProcessor
from utils.utils import cv2AddChineseText
from cfg.config import FPS, GOP_SIZE

label_set = [
    [0, 1, 2, 3, 4],
    [5, 6, 7],
    [8, 9],
    [10],
    [11, 12, 13, 14],
    [15],
    [16, 17],
    [18, 19],
    [20],
    [21],
    [22],
    [23],
    [24, 25, 26, 27, 28, 29],
    [30],
    [31, 32, 33, 34, 35, 36],
    [37],
    [38],
]
new2old = {}
for i, l_set in enumerate(label_set):
    for label in l_set:
        new2old[label] = i

label_color = [
    ["FLC1", (255, 128, 0)],
    ["FLA1", (255, 153, 51)],
    ["FLA3", (255, 178, 102)],
    ["FLA4", (230, 230, 0)],
    ["FLA5", (255, 153, 255)],
    ["FLA6", (153, 204, 255)],
    ["FLB1", (255, 102, 255)],
    ["FLB2", (255, 51, 255)],
    ["FLB2", (255, 51, 255)],
    ["FLE1", (51, 153, 255)],
    ["FLE5", (255, 153, 153)],
    ["FLE3", (255, 153, 153)],
    ["FLE4", (255, 51, 51)],
    ["FLE4", (255, 51, 51)],
    ["FLE4", (255, 51, 51)],
    ["FLE4", (255, 51, 51)],
    ["FLF1", (0, 255, 0)],
]

CHNlabel_color = [
    ["锥筒", (255, 128, 0)],
    ["施工标志", (255, 153, 51)],
    ["车道减少标志", (255, 178, 102)],
    ["声光报警器", (230, 230, 0)],
    ["限速标志", (255, 153, 255)],
    ["注意交通引导人员", (153, 204, 255)],
    ["导向标志", (255, 102, 255)],
    ["安全员", (255, 51, 255)],
    ["安全警示牌", (255, 51, 255)],
    ["防撞车", (51, 153, 255)],
    ["汽车", (255, 153, 153)],
    ["汽车", (255, 153, 153)],
    ["人员", (255, 51, 51)],
    ["人员", (255, 51, 51)],
    ["人员", (255, 51, 51)],
    ["人员", (255, 51, 51)],
    ["解除限速标志", (0, 255, 0)],
]

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


def check_positions(cone_positions, given_position, x_threshold):
    x_given, y_given = given_position
    has_above = False
    has_below = False
    x_min = x_given + x_threshold

    for x, y in cone_positions:
        if np.sqrt((x - x_given) ** 2 + (y - y_given) ** 2) <= x_threshold:
            if y > y_given:
                has_above = True
            elif y < y_given:
                has_below = True
            if x < x_min:
                x_min = x
    if x_given > x_min:
        return has_above and has_below
    return False


class VideoObjectTracker:
    def __init__(
            self,
            model_path,
            tracker_config,
            output_path,
            zoom=None,
            uav_height=None,
            angle=None,
            retry_interval=5,
            max_retry_time=120,
    ):
        self.model = YOLOVideoProcessor(model_path, tracker_config)
        self.output_path = output_path
        self.uav_height = uav_height
        self.angle = angle
        self.flc_id_count = [0 for i in range(len(label_color))]  # 保存每个类别的id数量
        self.known_target_id = [
            {} for i in range(len(label_color))
        ]  # 保存每个类别已知的id对应关系
        # self.known_target_id = {}
        # self.next_id = 0
        self.detection_threshold = 0.7
        self.first_distances = {}
        self.distance_threshold = 50
        self.flight_id = None
        self.fps = None
        self.zoom = zoom
        self.mqtt_client = None

        self.minio_frame_queue = queue.Queue(maxsize=60)
        self.minio_client = Minio(
            "59.46.115.177:49000",
            access_key="ats",
            secret_key="Abc131419..",
            secure=False,  # set to True if using HTTPS
        )
        self.bucket_name = "ats"

        self.first_detection_time = 0
        self.second_detection_time = 0
        self.third_detection_time = 0
        self.forth_detection_time = 0

        self.total_length = 0
        self.first_length = 0
        self.second_length = 0
        self.third_length = 0

        self.retry_interval = retry_interval
        self.max_retry_time = max_retry_time

        self.sent_events = {i: set() for i in range(len(label_color))}
        self.event_infos = []
        self.event_info_lock = threading.Lock()
        self.json_file_path = 'event_infos.json'
        self.frame_height = 0
        self.target_detection_counter = [0 for _ in range(len(label_color))]  # 针对每个类别的计数器
        self.target_status = [False for _ in range(len(label_color))]  # 目标存在状态

    def stream_frame_to_minio_publish(self, screenshot_path, image_name, event_info):
        if_uploaded = False
        try:
            self.minio_client.fput_object(
                self.bucket_name,
                image_name,
                screenshot_path,
                content_type="image/jpeg"
            )
            if_uploaded = True
            print(f"Successfully uploaded {image_name} to MinIO")
        except S3Error as e:
            print(f"Failed to upload image to MinIO: {e}")
        try:
            url = self.minio_client.presigned_get_object(self.bucket_name, image_name, expires=timedelta(hours=168))
            print(f"Presigned URL: {url}")
        except S3Error as e:
            print("error occurred while generating presigned URL.", e)
        if if_uploaded:
            topic = f"/ATS/yunying/task/ai/out/FLXJ/{self.flight_id}"
            try:
                self.mqtt_client.publish(topic, json.dumps(event_info))
            except Exception as e:
                print(f"Failed to publish event: {e}")
        # with self.event_info_lock:
        #     self.save_event_info_to_json(event_info)

    def save_event_info_to_json(self, event_info):
        with open(self.json_file_path, 'a') as f:
            json.dump(event_info, f)
            f.write('\n')  # 写入换行符以便阅读
    def calculate_distances(self, curr_vehicles):
        if len(curr_vehicles) < 2:
            return []
        vehicle_positions = np.array([v["positison"] for v in curr_vehicles])
        distances = np.sqrt(
            (
                    (
                            vehicle_positions[:, np.newaxis, :]
                            - vehicle_positions[np.newaxis, :, :]
                    )
                    ** 2
            ).sum(axis=2)
        )
        consecutive_distances = distances[
            np.arange(len(distances) - 1), np.arange(1, len(distances))
        ]
        return consecutive_distances

    def process_frames(self, info):
        (frame, current_frame, fps) = info
        self.im0 = frame.copy()
        self.frame_height = frame.shape[0]
        self.fps = FPS
        frame = self.process_frame(
            frame, self.frame_height, current_frame, self.fps
        )
        #frame = cv2.resize(frame, (1920, 1080))

        # 计算距离
        if self.first_detection_time != 0:
            end_time = current_frame / self.fps
            total_time = end_time - self.first_detection_time
            print(f'total_length: {total_time * 5:.2f} m')
            self.total_length = round(total_time * 5)
            cv2.putText(
                frame,
                f'total_length:{self.total_length} m',
                (960, 100),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2,
            )
        if self.second_detection_time != 0 and self.second_detection_time > self.first_detection_time:
            first_time = self.second_detection_time - self.first_detection_time
            print(f'first_length: {first_time * 5:.2f} m')
            self.first_length = round(first_time * 5)
            cv2.putText(
                frame,
                f'first_length:{self.first_length} m',
                (960, 150),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2,
            )
        if self.third_detection_time != 0 and self.third_detection_time > self.second_detection_time:
            if self.second_detection_time == 0:
                second_time = self.third_detection_time - self.first_detection_time
                print(f'second_length: {second_time * 5:.2f} m')
                self.second_length = round(second_time * 5)
                cv2.putText(
                    frame,
                    f'second_length:{self.second_length} m',
                    (960, 200),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2,
                )
            else:
                second_time = self.third_detection_time - self.second_detection_time
                print(f'second_length: {second_time * 5:.2f} m')
                self.second_length = round(second_time * 5)
                cv2.putText(
                    frame,
                    f'second_length:{self.second_length} m',
                    (960, 200),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2,
                )
        if self.forth_detection_time != 0 and self.forth_detection_time > self.third_detection_time:
            if self.third_detection_time == 0:
                third_time = self.forth_detection_time - self.second_detection_time
                print(f'third_length: {third_time * 5:.2f} m')
                self.third_length = round(third_time * 5)
                cv2.putText(
                    frame,
                    f'third_length:{self.third_length} m',
                    (960, 250),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2,
                )
            elif self.second_detection_time == 0:
                third_time = self.forth_detection_time - self.first_detection_time
                print(f'third_length: {third_time * 5:.2f} m')
                self.third_length = round(third_time * 5)
                cv2.putText(
                    frame,
                    f'third_length:{self.third_length} m',
                    (960, 250),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2,
                )
            else:
                third_time = self.forth_detection_time - self.third_detection_time
                print(f'third_length: {third_time * 5:.2f} m')
                self.third_length = round(third_time * 5)
                cv2.putText(
                    frame,
                    f'third_length:{self.third_length} m',
                    (960, 250),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2,
                )
        return frame

    def process_frame(self, frame, frame_height, current_frame, fps):
        frame_third = frame_height / 3
        cone_positions = []
        curr_vehicles = []
        else_vehicles = []
        first_distances = self.first_distances
        outputs = self.model.process_frame(frame, slice_height=720, slice_width=720, class_mapping=class_mapping_dict)
        if outputs is None:
            return frame
        if len(outputs) != 0:
            outputs = sorted(outputs, key=lambda x: x[1], reverse=True)
        else:
            return frame
        for output in outputs:
            if new2old[int(output[5])] == 0:
                xmin, ymin, xmax, ymax = map(int, output[:4])
                center_x, center_y = (xmin + xmax) / 2, (ymin + ymax) / 2
                cone_positions.append((center_x, center_y))
        for output in outputs:
            xmin, ymin, xmax, ymax = map(int, output[:4])
            center_x, center_y = (xmin + xmax) / 2, (ymin + ymax) / 2
            class_id = new2old[int(output[5])]
            target_id = output[6]
            if target_id is None:
                continue
            # 限制检测框的位置，避免检测框位于图像边缘
            x_edge = frame.shape[1] / 23
            y_edge = frame.shape[0] / 21
            if center_x < x_edge or center_x > frame.shape[1] - x_edge or center_y < y_edge or center_y > frame.shape[0] - y_edge:
                continue
            # 限制小汽车和工程车的检测位置
            if class_id == 10 or class_id == 11:
                if not check_positions(cone_positions, (center_x, center_y), frame.shape[1]/6) is True:
                    continue

            if target_id not in self.known_target_id[class_id].keys():
                self.known_target_id[class_id][target_id] = len(
                    self.known_target_id[class_id].keys()
                )
                self.flc_id_count[class_id] += 1
            target_id = self.known_target_id[class_id][target_id]
            class_label, color = label_color[class_id]
            CHNlabel, color2 = CHNlabel_color[class_id]

            self.target_detection_counter[class_id] += 1
            if self.target_detection_counter[class_id] >= 5:
                self.target_status[class_id] = True

                if "FLA1" in class_label and self.first_detection_time == 0 and center_y >= self.frame_height/2 :
                    self.first_detection_time = current_frame / fps
                if "FLA3" in class_label and self.second_detection_time == 0 and center_y >= self.frame_height/2:
                    self.second_detection_time = current_frame / fps
                if "FLA6" in class_label and self.third_detection_time == 0 and center_y >= self.frame_height/2:
                    self.third_detection_time = current_frame / fps
                if "FLB1" in class_label and self.forth_detection_time == 0 and center_y >= self.frame_height/2:
                    self.forth_detection_time = current_frame / fps

                cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 2)
                if class_id == 0:
                    curr_vehicles.append(
                        {
                            "target_id": target_id,
                            "class_label": class_label,
                            "class_id": class_id,
                            "position": np.array(
                                [(xmin + xmax) / 2, (ymin + ymax) / 2]
                            ),
                            "center_x": center_x,
                            "center_y": center_y,
                            "xmax": xmax,
                            "ymax": ymax,
                        }
                    )
                else:
                    else_vehicles.append(
                        {
                            "target_id": target_id,
                            "class_label": class_label,
                            "class_id": class_id,
                            "position": np.array(
                                [(xmin + xmax) / 2, (ymin + ymax) / 2]
                            ),
                            "center_x": center_x,
                            "center_y": center_y,
                            "xmax": xmax,
                            "ymax": ymax,
                        }
                    )
                frame = cv2AddChineseText(frame, f"{CHNlabel} _ {target_id}", (xmin, ymin - 10), (255, 0, 0), 10)
        curr_vehicles.sort(key=lambda v: v["center_y"])
        curr_vehicles = [v for v in curr_vehicles if v["center_y"] > frame_third]
        for i in range(len(else_vehicles)):
            vehicle = else_vehicles[i]
            current_ms = round((current_frame / fps) * 1000, 2)
            gop_ms = (GOP_SIZE / FPS) * 1000
            time_ms = current_ms - gop_ms
            if time_ms < 0:
                time_ms = 0
            if vehicle["target_id"] not in self.sent_events[vehicle["class_id"]]:
                screenshot_path = os.path.join(
                    self.screenshot_folder,
                    f'{vehicle["class_label"]}_{vehicle["target_id"]}_{time_ms}.png',
                )
                image_name = (
                    f'{vehicle["class_label"]}_{vehicle["target_id"]}_{time_ms}.png'
                )
                object_name = f"{self.flight_id}/{image_name}"
                cv2.imwrite(screenshot_path, frame)
                if "FLA3" in vehicle["class_label"]:
                    distance = self.first_length * 100
                    event_info = {
                        "event": "meta",
                        "type": vehicle["class_label"],
                        "isWarning": 0,
                        "time": time_ms,
                        "message": None,
                        "info": {
                            "id": vehicle["target_id"],
                            "lon": 125.16,
                            "lat": 41.72,
                            "distance": distance,
                            "same-distance": None,
                        },
                        "url": object_name,
                    }
                elif "FLA6" in vehicle["class_label"]:
                    distance = self.second_length * 100
                    event_info = {
                        "event": "meta",
                        "type": vehicle["class_label"],
                        "isWarning": 0,
                        "time": time_ms,
                        "message": None,
                        "info": {
                            "id": vehicle["target_id"],
                            "lon": 125.16,
                            "lat": 41.72,
                            "distance": distance,
                            "same-distance": None,
                        },
                        "url": object_name,
                    }
                elif "FLB1" in vehicle["class_label"]:
                    distance = self.third_length * 100
                    event_info = {
                        "event": "meta",
                        "type": vehicle["class_label"],
                        "isWarning": 0,
                        "time": time_ms,
                        "message": None,
                        "info": {
                            "id": vehicle["target_id"],
                            "lon": 125.16,
                            "lat": 41.72,
                            "distance": distance,
                            "same-distance": None,
                        },
                        "url": object_name,
                    }
                else:
                    event_info = {
                        "event": "meta",
                        "type": vehicle["class_label"],
                        "isWarning": 0,
                        "time": time_ms,
                        "message": None,
                        "info": {
                            "id": vehicle["target_id"],
                            "lon": 125.16,
                            "lat": 41.72,
                            "distance": None,
                            "same-distance": None,
                        },
                        "url": object_name,
                    }
                threading.Thread(
                    target=self.stream_frame_to_minio_publish,
                    args=(screenshot_path, object_name, event_info),
                ).start()
                self.sent_events[vehicle["class_id"]].add(vehicle["target_id"])
        for i in range(len(curr_vehicles) - 1):
            vehicle = curr_vehicles[i]
            next_vehicle = curr_vehicles[i + 1]
            distance_key = (vehicle["target_id"], next_vehicle["target_id"])
            if distance_key not in first_distances:
                # 计算距离
                distance_pixels = np.sqrt(
                    (vehicle["center_x"] - next_vehicle["center_x"]) ** 2
                    + (vehicle["center_y"] - next_vehicle["center_y"]) ** 2
                )
                distance_meters = distance_pixels * self.get_ratio()
                first_distances[distance_key] = distance_meters
                isWarning = 1 if distance_meters > 4 else 0
                message = "间距过大" if isWarning else "正常"
                # 绘制和显示距离
                cv2.line(
                    frame,
                    (int(vehicle["center_x"]), int(vehicle["center_y"])),
                    (int(next_vehicle["center_x"]), int(next_vehicle["center_y"])),
                    (0, 255, 0),
                    2,
                )
                text = f"{vehicle['class_label']}_{vehicle['target_id']} to {next_vehicle['class_label']}_{next_vehicle['target_id']}: {distance_meters:.2f} m"
                text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.5, 2)[
                    0
                ]
                # 定位文本到每个检测框的右方，稍微偏上一点以避免与检测框重叠
                # 使用xmax和ymax来定位文本，确保文本位于检测框的右侧
                color1 = (0, 0, 255) if distance_meters > 400 else (0, 255, 0)
                text_x = max(int(vehicle["xmax"]), int(next_vehicle["xmax"])) + 50
                text_y = (
                        min(int(vehicle["center_y"]), int(next_vehicle["center_y"]))
                        + 10
                )  # 基于两个检测框的较上方中心点向上偏移10个像素
                # 确保文本位置不会超出图像边界
                # text_x = min(text_x, frame.shape[1] - text_size[0] - 10)
                # text_y = max(text_y, text_size[1] + 10)
                cv2.putText(
                    frame,
                    text,
                    (text_x, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    color1,
                    2,
                )
                current_ms = round((current_frame / fps) * 1000, 2)
                gop_ms = (GOP_SIZE / FPS) * 1000
                time_ms = current_ms - gop_ms
                if time_ms < 0:
                    time_ms = 0
                # timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S%f")
                screenshot_path = os.path.join(
                    self.screenshot_folder,
                    f'{vehicle["class_label"]}_{vehicle["target_id"]}_{time_ms}.png',
                )
                image_name = (
                    f'{vehicle["class_label"]}_{vehicle["target_id"]}_{time_ms}.png'
                )
                cv2.imwrite(screenshot_path, frame)
                object_name = f"{self.flight_id}/{image_name}"
                if (
                        vehicle["target_id"]
                        not in self.sent_events[vehicle["class_id"]]
                ):
                    distance = round(distance_meters * 100)
                    event_info = {
                        "event": "meta",
                        "type": vehicle["class_label"],
                        "isWarning": isWarning,
                        "time": time_ms,
                        "message": message,
                        "info": {
                            "id": vehicle["target_id"],
                            "lon": 125.16,
                            "lat": 41.72,
                            "distance": distance,
                            "same-distance": distance,
                        },
                        "url": object_name,
                    }
                    threading.Thread(
                        target=self.stream_frame_to_minio_publish,
                        args=(screenshot_path, object_name, event_info),
                    ).start()
                    self.sent_events[vehicle["class_id"]].add(vehicle["target_id"])
            else:
                distance_meters = first_distances[distance_key]
                # 绘制和显示距离
                cv2.line(
                    frame,
                    (int(vehicle["center_x"]), int(vehicle["center_y"])),
                    (int(next_vehicle["center_x"]), int(next_vehicle["center_y"])),
                    (0, 255, 0),
                    2,
                )
                text = f"{vehicle['class_label']}_{vehicle['target_id']} to {next_vehicle['class_label']}_{next_vehicle['target_id']}: {distance_meters:.2f} m"
                text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.5, 2)[
                    0
                ]
                # 定位文本到每个检测框的右方，稍微偏上一点以避免与检测框重叠
                # 使用xmax和ymax来定位文本，确保文本位于检测框的右侧
                color1 = (0, 0, 255) if distance_meters > 400 else (0, 255, 0)
                text_x = max(int(vehicle["xmax"]), int(next_vehicle["xmax"])) + 50
                text_y = (
                        min(int(vehicle["center_y"]), int(next_vehicle["center_y"]))
                        + 10
                )  # 基于两个检测框的较上方中心点向上偏移10个像素
                # 确保文本位置不会超出图像边界
                #text_x = min(text_x, frame.shape[1] - text_size[0] - 10)
                #text_y = max(text_y, text_size[1] + 10)
                cv2.putText(
                    frame,
                    text,
                    (text_x, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    color1,
                    2,
                )
        return frame

    def get_ratio(self):
        equal_length = 35
        return equal_length / int(self.frame_height)

    def generate_new_paths(self):
        timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        self.output_path = os.path.join("runs", "detect", f"{timestamp}.mp4")
        self.screenshot_folder = os.path.join("runs", "screenshots", timestamp)
        os.makedirs(self.screenshot_folder, exist_ok=True)
        print(
            f"Generated new paths: output_path={self.output_path}, screenshot_folder={self.screenshot_folder}"
        )

    def update_parameters(self, uav_height, angle, zoom, flight_id, mqtt_client):
        self.uav_height = uav_height / 10
        self.angle = -angle / 10
        self.zoom = zoom
        self.flight_id = flight_id
        self.generate_new_paths()  # 生成新的输出路径和截屏路径
        self.mqtt_client = mqtt_client
        print(
            f"Updated parameters: uav_height={self.uav_height}, angle={self.angle}, flight_if={self.flight_id}"
        )

    def stop_detection(self):
        self.first_distances = {}
        self.sent_events = {i: set() for i in range(len(label_color))}
        self.first_detection_time = 0
        self.second_detection_time = 0
        self.third_detection_time = 0
        self.forth_detection_time = 0
        self.next_id = 0
        print("Detection stopped.")
