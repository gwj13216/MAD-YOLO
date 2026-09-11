import cv2
from ultralytics import YOLO
import os
from timm.layers import create_act
# 初始化模型和视频参数
model = YOLO('yolov8n.pt')  # 可替换为yolov8s.pt/yolov8x.pt等
video_path = "test2.mp4"  # 输入视频路径
output_dir = "highres_captures"  # 输出目录
os.makedirs(output_dir, exist_ok=True)

# 视频属性获取
cap = cv2.VideoCapture(video_path)
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# 输出视频配置
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(f"{output_dir}12.1510.mp4", fourcc, fps, (width, height))

# 处理循环
frame_count = 0
while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    # 每1帧检测一次
    if frame_count % 1 == 0:
        results = model(frame, verbose=False)
        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                cls = int(box.cls[0])

                # 绘制检测框和标签
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = f"{model.names[cls]} {conf:.2f}"
                cv2.putText(frame, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # 写入输出视频
    out.write(frame)

    # 实时显示（按q退出）
    cv2.imshow('YOLOv8 Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    frame_count += 1

# 资源释放
cap.release()
out.release()
cv2.destroyAllWindows()