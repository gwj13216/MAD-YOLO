from ultralytics import YOLO

model = YOLO('runs/train/yolov8/weights/best.pt')
model.predict(
   source='D:/app/datasettag/test/images/WIN_20250807_05_27_33_Pro.jpg',
   imgsz=640,
   save=True,
   visualize=True # 启用中间层特征图可视化
)