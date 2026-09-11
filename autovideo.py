from ultralytics import YOLO
import cv2
import os


def video_target_detection(video_path, output_dir, model_path='best.pt', conf_threshold=0.5):
    """检测视频目标并保存含目标的原始分辨率帧"""
    # 加载预训练模型
    model = YOLO(model_path)

    # 打开视频文件
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"无法打开视频文件: {video_path}")

    # 获取原视频参数
    original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    print(f"开始处理视频: {video_path}")
    print(f"原视频参数: {original_width}x{original_height}@{fps}fps")

    # 逐帧处理
    saved_frame_count = 0
    for frame_idx in range(total_frames):
        ret, frame = cap.read()
        if not ret:
            break

        # 执行目标检测
        results = model(frame, verbose=False)
        if results[0].boxes:
            # 保存检测帧（保持原始分辨率）
            frame_filename = f"frame_{frame_idx:06d}.jpg"
            frame_path = os.path.join(output_dir, frame_filename)
            cv2.imwrite(frame_path, frame)
            saved_frame_count += 1

            # 可选：绘制检测框（如需可视化）
            # annotated_frame = results[0].plot()
            # cv2.imwrite(frame_path, annotated_frame)

    cap.release()
    print(
        f"处理完成！共保存 {saved_frame_count}/{total_frames} 帧 ({100 * saved_frame_count / total_frames:.2f}% 保留率)")


# 使用示例
video_target_detection(
    video_path='dataset/mosquito (8).mp4',
    output_dir='dataset/mosquito（8）',
    model_path='best.pt',
    conf_threshold=0.10
)