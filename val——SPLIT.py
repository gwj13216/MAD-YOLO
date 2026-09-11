import os
import shutil
from sklearn.model_selection import train_test_split

# 设置参数
val_images_dir = "D:/app/datasettag/val/images"      # 验证集图片目录
val_labels_dir = "D:/app/datasettag/val/labels"      # 验证集标签目录
test_images_dir = "D:/app/datasettag/test/images"    # 测试集图片目录
test_labels_dir = "D:/app/datasettag/test/labels"    # 测试集标签目录

# 创建目标目录
os.makedirs(test_images_dir, exist_ok=True)
os.makedirs(test_labels_dir, exist_ok=True)

# 获取验证集中的所有文件
val_image_files = [f for f in os.listdir(val_images_dir)
                  if f.endswith('.jpg') or f.endswith('.png')]
val_label_files = [f.replace('.jpg', '.txt').replace('.png', '.txt')
                  for f in val_image_files]

# 确保标签文件存在
val_label_files = [f for f in val_label_files if os.path.exists(os.path.join(val_labels_dir, f))]

# 随机划分验证集为测试集和新的验证集（各50%）
test_image_files, new_val_image_files = train_test_split(val_image_files, test_size=0.5, random_state=42)
test_label_files, new_val_label_files = train_test_split(val_label_files, test_size=0.5, random_state=42)

# 移动测试集文件
for img_file, label_file in zip(test_image_files, test_label_files):
    shutil.move(os.path.join(val_images_dir, img_file), os.path.join(test_images_dir, img_file))
    shutil.move(os.path.join(val_labels_dir, label_file), os.path.join(test_labels_dir, label_file))

print(f"测试集划分完成！共移动 {len(test_image_files)} 个文件到测试集")
print("新的验证集已保留剩余文件，验证集大小减半")