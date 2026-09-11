#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
把自动标注工具产出的结果整理成 ultralytics 训练目录结构。

输入 (src):
  src/
    *.jpg / *.png ...           # 图片
    *.mp4 / *.avi ...           # 视频
    labels/
      <图片名>.txt              # 图片标签 (YOLO 格式)
      <视频名>/frame_xxxxxx.txt # 视频逐帧标签
      classes.txt               # 类别清单 (可选, 格式 "0 mosquito")

输出 (out):
  out/
    images/train/*.jpg
    images/val/*.jpg
    labels/train/*.txt
    labels/val/*.txt
    dataset.yaml

命令行用法:
  python export_dataset.py --src <已标注文件夹> [--out <输出目录>] [--val 0.1] [--seed 0]
"""

import os
import sys
import glob
import shutil
import random
import argparse

import cv2
import numpy as np

IMAGE_EXTS = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp')
VIDEO_EXTS = ('.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.m4v')


def imread_unicode(path):
    """cv2.imread 的非 ASCII 路径安全版 (中文路径可用)。"""
    data = np.fromfile(path, dtype=np.uint8)
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def imwrite_unicode(path, img, ext=None):
    """cv2.imwrite 的非 ASCII 路径安全版 (中文路径可用)。"""
    ext = ext or os.path.splitext(path)[1] or '.jpg'
    ok, buf = cv2.imencode(ext, img)
    if not ok:
        return False
    buf.tofile(path)
    return True


def image_size_unicode(path):
    """只读图片头获取 (宽, 高), 不解码整张图, 非 ASCII 路径安全。失败返回 (0, 0)。"""
    try:
        from PIL import Image
        with Image.open(path) as im:
            return im.size  # (W, H)
    except Exception:
        img = imread_unicode(path)
        if img is None:
            return (0, 0)
        return (img.shape[1], img.shape[0])


def _read_classes(src, labels_dir, classes):
    """读取类别名: classes.txt 优先, 其次传入的 classes, 最后默认 mosquito。"""
    ctxt = os.path.join(labels_dir, 'classes.txt')
    if os.path.exists(ctxt):
        names = {}
        with open(ctxt, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split(None, 1)
                try:
                    idx = int(parts[0])
                except ValueError:
                    continue
                names[idx] = parts[1] if len(parts) > 1 else str(idx)
        if names:
            # 补齐 0..max(idx)
            max_i = max(names.keys())
            return [names.get(i, str(i)) for i in range(max_i + 1)]
    if classes:
        return list(classes)
    return ['mosquito']


def _find_label(img_path, labels_dir):
    """在 labels/ 子目录或图片同目录下查找对应标签。"""
    stem = os.path.splitext(os.path.basename(img_path))[0]
    cands = [
        os.path.join(labels_dir, stem + '.txt'),
        os.path.join(os.path.dirname(img_path), stem + '.txt'),
    ]
    for c in cands:
        if os.path.exists(c) and os.path.getsize(c) > 0:
            return c
    return None


def build_dataset(src, out, val_split=0.1, classes=None, seed=0, progress_cb=None):
    """
    整理数据集。返回统计 dict: {images, videos, video_frames, train, val, out, yaml}
    """
    src = os.path.abspath(src)
    out = os.path.abspath(out)
    labels_dir = os.path.join(src, 'labels')

    if not os.path.isdir(labels_dir):
        raise RuntimeError(
            f"未找到 {labels_dir}\n"
            f"请先在自动标注工具中点击【保存全部】(默认保存到 labels/ 子文件夹)。")

    names = _read_classes(src, labels_dir, classes)

    # ---- 收集图片对 ----
    img_pairs = []   # (src_img, label_path)
    for ext in IMAGE_EXTS:
        for p in sorted(glob.glob(os.path.join(src, '*' + ext))):
            lab = _find_label(p, labels_dir)
            if lab:
                img_pairs.append((p, lab))

    # ---- 收集视频帧 ----
    vid_frames = []  # (video_path, frame_idx, label_path)
    for ext in VIDEO_EXTS:
        for vp in sorted(glob.glob(os.path.join(src, '*' + ext))):
            vdir = os.path.join(labels_dir, os.path.splitext(os.path.basename(vp))[0])
            if not os.path.isdir(vdir):
                continue
            for ftxt in sorted(glob.glob(os.path.join(vdir, 'frame_*.txt'))):
                base = os.path.basename(ftxt)[len('frame_'):-len('.txt')]
                try:
                    fidx = int(base)
                except ValueError:
                    continue
                if os.path.getsize(ftxt) > 0:
                    vid_frames.append((vp, fidx, ftxt))

    total = len(img_pairs) + len(vid_frames)
    if total == 0:
        raise RuntimeError("没有找到任何带标签的图片或视频帧, 请先完成批量标注并保存。")

    # ---- 划分 train / val ----
    all_items = [('img', x) for x in img_pairs] + [('vid', x) for x in vid_frames]
    rng = random.Random(seed)
    rng.shuffle(all_items)

    if total == 1:
        n_val = 0
    else:
        n_val = max(1, int(round(total * val_split)))
        n_val = min(n_val, total - 1)   # 保证 train 至少 1 张
    train_items = all_items[n_val:]
    val_items = all_items[:n_val]

    # ---- 写文件 ----
    def _report(done, msg):
        if progress_cb:
            progress_cb(done, total, msg)

    splits = [('train', train_items), ('val', val_items)]
    used_names = set()
    stats = {
        'images': len(img_pairs),
        'videos': len(set(vp for vp, _, _ in vid_frames)),
        'video_frames': len(vid_frames),
        'train': len(train_items),
        'val': len(val_items),
    }

    done = 0
    for split, items in splits:
        idir = os.path.join(out, 'images', split)
        ldir = os.path.join(out, 'labels', split)
        os.makedirs(idir, exist_ok=True)
        os.makedirs(ldir, exist_ok=True)
        for kind, payload in items:
            if kind == 'img':
                src_img, lab = payload
                ext = os.path.splitext(src_img)[1].lower()
                base = os.path.splitext(os.path.basename(src_img))[0]
                dst_img = os.path.join(idir, f"{base}{ext}")
            else:  # vid -> 抽取帧存为 jpg
                vp, fidx, lab = payload
                base = f"{os.path.splitext(os.path.basename(vp))[0]}__frame_{fidx:06d}"
                dst_img = os.path.join(idir, f"{base}.jpg")
                if not _extract_frame(vp, fidx, dst_img):
                    continue

            # 文件名去重
            k = 1
            while dst_img.lower() in used_names:
                dst_img = os.path.join(idir, f"{base}_{k}{os.path.splitext(dst_img)[1]}")
                k += 1
            used_names.add(dst_img.lower())
            if kind == 'img':
                shutil.copy2(src_img, dst_img)
            dst_label = os.path.join(ldir, os.path.splitext(os.path.basename(dst_img))[0] + '.txt')
            shutil.copy2(lab, dst_label)

            done += 1
            _report(done, f"写 {split} {done}/{total}")

    # ---- dataset.yaml ----
    yaml_path = os.path.join(out, 'dataset.yaml')
    path_str = out.replace('\\', '/')
    val_field = "images/val" if val_items else "images/train"
    with open(yaml_path, 'w', encoding='utf-8') as f:
        f.write(f"# generated by export_dataset.py\n")
        f.write(f"path: {path_str}\n")
        f.write(f"train: images/train\n")
        f.write(f"val: {val_field}\n")
        f.write(f"names:\n")
        for i, n in enumerate(names):
            f.write(f"  {i}: {n}\n")

    stats['out'] = out
    stats['yaml'] = yaml_path
    stats['names'] = names
    return stats


def _extract_frame(video_path, frame_idx, dst_img):
    """抽取视频某一帧存为 jpg。返回是否成功。"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        cap.release()
        return False
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        return False
    return imwrite_unicode(dst_img, frame)


def _cli_progress(done, total, msg):
    print(f"\r[{done}/{total}] {msg}", end='', flush=True)


def main():
    ap = argparse.ArgumentParser(description="整理自动标注结果 -> ultralytics 训练集")
    ap.add_argument('--src', required=True, help='已标注的文件夹 (含 labels/ 子目录)')
    ap.add_argument('--out', default=None, help='输出目录 (默认 <src>/yolo_dataset)')
    ap.add_argument('--val', type=float, default=0.1, help='验证集占比 (默认 0.1)')
    ap.add_argument('--seed', type=int, default=0, help='随机种子 (默认 0)')
    args = ap.parse_args()

    out = args.out or os.path.join(args.src, 'yolo_dataset')
    stats = build_dataset(args.src, out, args.val, seed=args.seed, progress_cb=_cli_progress)
    print("\n完成:")
    for k, v in stats.items():
        print(f"  {k}: {v}")


if __name__ == '__main__':
    main()
