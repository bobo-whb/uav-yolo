from ultralytics import YOLO

# Load a model
model = YOLO(r"/root/UltralyticsYOLO/runs/detect/train2/weights/best.pt")  # load a pretrained model (recommended for training)

# Train the model with 2 GPUs
results = model.train(data="/root/datasets/new_data.yaml", epochs=200, batch=300, imgsz=640, cache=True,
                      verbose=True, deterministic=False, perspective=0.0005, device=[0, 1], fliplr=0)