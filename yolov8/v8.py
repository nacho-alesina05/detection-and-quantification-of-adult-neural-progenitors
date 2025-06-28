from ultralytics import YOLO
import os

os.chdir('..')

model = YOLO('yolov8x.pt')
model.train(
    data='dataset.yaml',
    imgsz=1280,
    epochs=150,
    batch=4,
    device=0,
    name='v8x_150epochs',
    project='runs/train',
    cache=True,
    workers=4,
    patience=70,
    save_txt=True,
    save_conf=True,
    plots=True
)

