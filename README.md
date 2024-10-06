## 配置环境

### 1. 安装依赖

使用cuda12.1，cudnn9.2.1，python3.11

```bash
# Install dependencies
pip install -r requirements.txt

# Install ffmpeg
conda install -c conda-forge ffmpeg
```

### 2. 修改 byte_tracker.py

路径：`...\anaconda3\envs\your_env\lib\python3.11\site-packages\ultralytics\trackers\byte_tracker.py`

```python
# 267行
remain_inds = scores >= self.args.track_high_thresh
inds_low = scores > self.args.track_low_thresh
inds_high = scores < self.args.track_high_thresh

# 修改为
remain_inds = scores.numpy() >= self.args.track_high_thresh
inds_low = scores.numpy() > self.args.track_low_thresh
inds_high = scores.numpy() < self.args.track_high_thresh
```

### 3. 生成TensorRT模型

打开`utils/export_TensorRT.py`，修改pt模型路径后运行

### 4. 修改配置文件

修改`cfg/config.py`中的`MODEL_PATH`为生成的TensorRT模型路径

## 文件结构

- **cfg**
  - **bytetrack.yaml**: ByteTrack的配置文件，包含跟踪算法的相关参数。
  - **config.py**: 配置相关的Python脚本。

- **engine**
  - **nms.py**: 非极大值抑制（NMS）的实现脚本，用于处理目标检测结果。
  - **predictor.py**: 预测器脚本，包含模型预测的相关逻辑。

- **models**
  - **yolov10m1.pt**: YOLOv10模型的PyTorch格式权重文件。
  - **yolov10m2.engine**: YOLOv10模型的TensorRT引擎文件，用于优化后的推理。
  - **yolov10m2.pt**: 另一个YOLOv10模型的PyTorch格式权重文件。

- **utils**
  - **export_TensorRT.py**: 导出TensorRT模型的工具脚本，用于将PyTorch模型转换为TensorRT引擎。

