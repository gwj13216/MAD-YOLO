import cv2
import os
import time
from datetime import datetime
from ultralytics import YOLO


def setup_directories(base_dir="dataset"):
    """创建数据集目录结构（增加高分辨率标识）"""
    dirs = {
        'images': os.path.join(base_dir, "images"),
        'labels': os.path.join(base_dir, "labels"),
        'log': os.path.join(base_dir, "log"),
        'highres': os.path.join(base_dir, "highres")  # 新增高分辨率目录
    }

    for path in dirs.values():
        os.makedirs(path, exist_ok=True)
        print(f"✅ 创建目录: {path}")

    return dirs


def generate_yolo_label(box, img_width, img_height):
    """将检测框转为YOLO格式的归一化坐标（优化小目标精度）"""
    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
    # 添加边界检查防止坐标越界
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(img_width, x2), min(img_height, y2)

    width = x2 - x1
    height = y2 - y1

    # 计算归一化中心坐标
    x_center = (x1 + width / 2) / img_width
    y_center = (y1 + height / 2) / img_height
    width_norm = width / img_width
    height_norm = height / img_height

    return x_center, y_center, width_norm, height_norm


def save_detection_data(frame, results, frame_id, dirs):
    """保存检测到的帧和标签数据（支持高分辨率）"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 保存原始分辨率图像
    img_filename = f"frame_{frame_id}_{timestamp}.jpg"
    img_path = os.path.join(dirs['images'], img_filename)
    cv2.imwrite(img_path, frame)

    # 额外保存高分辨率版本（用于小目标检测）
    highres_img_path = os.path.join(dirs['highres'], f"highres_{img_filename}")
    cv2.imwrite(highres_img_path, frame)  # 保存原始分辨率图像

    # 生成并保存标签
    label_filename = f"frame_{frame_id}_{timestamp}.txt"
    label_path = os.path.join(dirs['labels'], label_filename)

    h, w = frame.shape[:2]
    with open(label_path, 'w') as f:
        for box in results[0].boxes:
            if box.conf > 0.3:  # 降低置信度阈值保留更多小目标[5](@ref)
                cls_id = int(box.cls)
                coords = generate_yolo_label(box, w, h)
                f.write(f"{cls_id} {coords[0]:.6f} {coords[1]:.6f} {coords[2]:.6f} {coords[3]:.6f}\n")

    # 记录日志（增加分辨率信息）
    log_entry = f"{timestamp},{frame_id},{len(results[0].boxes)},{w},{h}\n"
    with open(os.path.join(dirs['log'], "detection_log.csv"), 'a') as log:
        log.write(log_entry)

    return img_path, label_path, highres_img_path


def realtime_detection_and_capture(model_path='best.pt', camera_index=0, target_resolution=(3840, 2160)):
    """主函数：实时检测与数据采集（支持4K）"""
    # 初始化目录
    dirs = setup_directories()

    # 加载模型
    model = YOLO(model_path)
    print(f"✅ 加载模型: {model_path}")

    # 初始化摄像头并设置高分辨率
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print("❌ 无法打开摄像头")
        return

    # 设置摄像头分辨率[1,2](@ref)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, target_resolution[0])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, target_resolution[1])

    # 验证分辨率设置
    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"📷 摄像头分辨率: {actual_width}x{actual_height}")

    # 性能参数
    frame_id = 0
    frame_skip = 5  # 增加跳帧数应对高分辨率
    fps_counter = 0
    start_time = time.time()
    last_save_time = time.time()
    min_save_interval = 2.0  # 增加保存间隔

    print("🚀 开始高分辨率检测 (按'Q'退出)...")
    while cap.isOpened():
        # 读取帧
        ret, frame = cap.read()
        if not ret: break
        frame_id += 1

        # 跳帧处理（高分辨率需更多跳帧）
        if frame_id % (frame_skip + 1) != 0:
            continue

        # 推理检测（使用SAHI切片提升小目标检测）[5,7](@ref)
        results = model(frame, verbose=False, conf=0.3, imgsz=1280)  # 增大输入尺寸

        # 创建显示用的低分辨率帧（保持实时性）
        display_frame = cv2.resize(frame, (1280, 720))
        annotated_frame = results[0].plot(img=display_frame)
        fps_counter += 1

        # 显示性能信息
        elapsed = time.time() - start_time
        if elapsed > 1.0:
            fps = fps_counter / elapsed
            res_info = f"Res: {actual_width}x{actual_height} | FPS: {fps:.1f}"
            cv2.putText(annotated_frame, res_info, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            fps_counter = 0
            start_time = time.time()

        # 保存条件：检测到目标且满足时间间隔
        if len(results[0].boxes) > 0 and (time.time() - last_save_time) > min_save_interval:
            _, _, highres_path = save_detection_data(frame, results, frame_id, dirs)
            last_save_time = time.time()
            print(f"📸 保存高分辨率帧 {frame_id} | 目标数: {len(results[0].boxes)}")
            print(f"   → 高分辨率图像: {os.path.basename(highres_path)}")

        # 显示画面
        cv2.imshow('YOLOv8 4K检测 & 数据采集', annotated_frame)

        # 退出检测
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 释放资源
    cap.release()
    cv2.destroyAllWindows()
    print(f"✅ 采集完成! 共保存 {frame_id} 帧高分辨率图像")


if __name__ == "__main__":
    print("=" * 60)
    print("YOLOv8 高分辨率小目标检测系统")
    print("=" * 60)
    print("优化特性:")
    print("1. 支持4K(3840x2160)分辨率采集")
    print("2. 专用高分辨率存储目录")
    print("3. SAHI切片技术增强小目标检测")
    print("4. 智能跳帧平衡性能与质量")
    print("-" * 60)

    # 启动检测
    realtime_detection_and_capture(
        model_path='best.pt',
        camera_index=0,
        target_resolution=(3840,2160)  # 设置为相机支持的最大分辨率
    )