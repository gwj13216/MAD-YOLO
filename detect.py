from ultralytics import YOLO

# 加载训练好的模型
model = YOLO('runs/train/2026-211/weights/best.pt')

# 在测试集上评估
metrics = model.val(
    data='dataset/dataset.yaml',  # 数据集配置
    split='test',         # 指定测试集
    imgsz=640,            # 图像尺寸
    batch=32,
    workers=16,
    device=0              # 设备
)

# 打印关键指标
print(f"mAP@0.5: {metrics.box.map50:.4f}")
print(f"mAP@0.5:0.95: {metrics.box.map:.4f}")
print(f"Recall: {metrics.box.recall:.4f}")