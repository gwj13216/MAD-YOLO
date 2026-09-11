#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
自动标注 + 人工修正工具 (labelimg 风格)

功能:
  1. 选择一个文件夹, 对其中的图片和视频批量运行 YOLOv8 检测 (自动打标签)。
  2. 在画布上查看检测框 (类别 + 置信度), 支持缩放 / 平移。
  3. 像 labelimg 一样手动修正:
       - 空白处按住左键拖动  -> 新建框
       - 点击 / 拖动框体      -> 选中并移动
       - 拖动四角手柄        -> 调整框大小
       - Delete / 右键       -> 删除框
       - 右侧类别下拉框      -> 修改选中框的类别
  4. 保存为 YOLO 格式标签 (images -> labels/xxx.txt; 视频 -> labels/视频名/frame_xxxxxx.txt)。
  5. 视频逐帧浏览 / 播放, 逐帧修正。

运行环境: D:\\minicoonda\\envs\\yolov8 (本脚本会优先使用本地 custom ultralytics fork)
"""

import os
import sys
import glob
import time
import traceback

# ---------------------------------------------------------------------------
# 引导: 使用本目录下的 custom ultralytics fork (包含 SLM-AFPN / C2f_SCConv 等自定义模块)
# ---------------------------------------------------------------------------
ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
os.chdir(ROOT)

import cv2
import numpy as np
import export_dataset

from PyQt5.QtCore import Qt, QThread, pyqtSignal, QRectF, QPointF, QTimer
from PyQt5.QtGui import (QImage, QPixmap, QPainter, QPen, QColor, QBrush, QFont,
                         QKeySequence)
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QLabel, QPushButton, QToolBar, QAction, QListWidget, QListWidgetItem,
    QFileDialog, QDockWidget, QGraphicsView, QGraphicsScene, QGraphicsPixmapItem,
    QGraphicsItem, QGraphicsRectItem, QComboBox, QDoubleSpinBox, QSpinBox,
    QSlider, QProgressBar, QMessageBox, QGroupBox, QCheckBox, QShortcut,
    QFrame, QStatusBar, QStyle, QLineEdit, QInputDialog,
)

IMAGE_EXTS = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp')
VIDEO_EXTS = ('.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.m4v')

# 类别颜色 (按 class id 循环取色)
_COLOR_HEX = ['#ff6384', '#ff9f40', '#ffcd56', '#4bc0c0', '#36a2eb',
              '#9966ff', '#ff6347', '#32cd32', '#ff1493', '#00bfff']


def class_color(cls_id):
    return QColor(_COLOR_HEX[cls_id % len(_COLOR_HEX)])


# ---------------------------------------------------------------------------
# 数据模型
# ---------------------------------------------------------------------------
class Box:
    """一个检测框, 坐标为图像像素 (x1, y1, x2, y2)。"""
    __slots__ = ('x1', 'y1', 'x2', 'y2', 'conf', 'cls')

    def __init__(self, x1, y1, x2, y2, conf, cls):
        self.x1 = float(x1)
        self.y1 = float(y1)
        self.x2 = float(x2)
        self.y2 = float(y2)
        self.conf = float(conf)
        self.cls = int(cls)

    def to_yolo(self, W, H):
        cx = (self.x1 + self.x2) / 2.0 / W
        cy = (self.y1 + self.y2) / 2.0 / H
        w = (self.x2 - self.x1) / W
        h = (self.y2 - self.y1) / H
        return f"{self.cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"


class Item:
    """一个待标注文件 (图片或视频) 及其检测结果。"""
    def __init__(self, path, kind):
        self.path = path
        self.kind = kind            # 'image' | 'video'
        self.boxes = []             # 图片: [Box]
        self.frames = {}            # 视频: {frame_idx: [Box]}
        self.frame_count = 0        # 视频总帧数
        self.cap = None             # 视频 cv2.VideoCapture 缓存
        self.cap_pos = -1           # cap 当前已读到的帧号 (顺序读取用)
        self.fps = 25.0             # 视频帧率
        self.cur_frame = 0          # 视频当前帧
        self.modified = False

    @property
    def stem(self):
        return os.path.splitext(os.path.basename(self.path))[0]


# ---------------------------------------------------------------------------
# 推理线程
# ---------------------------------------------------------------------------
class ModelLoader(QThread):
    """后台加载模型, 避免卡住界面; 加载后的模型可复用于推理。"""
    loaded = pyqtSignal(object, list)   # model, names
    error = pyqtSignal(str)

    def __init__(self, path, parent=None):
        super().__init__(parent)
        self.path = path

    def run(self):
        from ultralytics import YOLO
        try:
            m = YOLO(self.path)
            names = [str(m.names[i]) for i in range(len(m.names))]
            self.loaded.emit(m, names)
        except Exception:
            self.error.emit(traceback.format_exc())


class InferenceThread(QThread):
    progress = pyqtSignal(int, int, str)      # done, total, message
    image_result = pyqtSignal(int, list)      # 图片 index, list[Box]
    video_result = pyqtSignal(int, dict)      # 视频 index, {frame: list[Box]}
    done = pyqtSignal()
    error = pyqtSignal(str)
    warn = pyqtSignal(str)

    def __init__(self, model, images, videos, conf, iou, imgsz, vid_stride,
                 parent=None):
        super().__init__(parent)
        self.model = model
        self.images = images            # list[(index, path)]
        self.videos = videos            # list[(index, path)]
        self.conf = conf
        self.iou = iou
        self.imgsz = imgsz
        self.vid_stride = vid_stride

    def run(self):
        try:
            model = self.model
            total = len(self.images) + len(self.videos)
            done = 0

            # ---- 图片: 分批推理 ----
            # 注意: 直接把整个路径列表传给 predict 会触发 autocast_list ->
            # LoadPilAndNumpy, 把所有原图一次性载入内存导致 OOM。
            # 这里按小批 (16 张) 分块推理, 内存有界。
            if self.images:
                idxs = [i for i, _ in self.images]
                paths = [p for _, p in self.images]
                chunk = 16
                for k in range(0, len(paths), chunk):
                    batch_paths = paths[k:k + chunk]
                    batch_idxs = idxs[k:k + chunk]
                    try:
                        results = model.predict(
                            batch_paths, imgsz=self.imgsz, conf=self.conf, iou=self.iou,
                            verbose=False, save=False,
                        )
                    except Exception as e:
                        self.warn.emit(f"跳过 {len(batch_paths)} 张图片: {e}")
                        done += len(batch_paths)
                        self.progress.emit(done, total, f"图片 {done}/{len(self.images)}")
                        continue
                    for idx, r in zip(batch_idxs, results):
                        boxes = extract_boxes(r)
                        self.image_result.emit(idx, boxes)
                        done += 1
                        self.progress.emit(done, total, f"图片 {done}/{len(self.images)}")

            # ---- 视频: 逐帧 (stream) ----
            for vi, (idx, vpath) in enumerate(self.videos):
                frame_boxes = {}
                try:
                    gen = model.predict(
                        vpath, imgsz=self.imgsz, conf=self.conf, iou=self.iou,
                        verbose=False, save=False, stream=True,
                        vid_stride=self.vid_stride,
                    )
                    for fi, r in enumerate(gen):
                        boxes = extract_boxes(r)
                        if boxes:
                            frame_boxes[fi * self.vid_stride] = boxes
                except Exception as e:
                    self.error.emit(f"视频推理失败: {os.path.basename(vpath)}\n{e}")
                self.video_result.emit(idx, frame_boxes)
                done += 1
                self.progress.emit(done, total, f"视频 {os.path.basename(vpath)}")

            self.done.emit()
        except Exception:
            self.error.emit(traceback.format_exc())


def extract_boxes(result):
    """把 ultralytics 结果转成 Box 列表。"""
    out = []
    if result is None or result.boxes is None or len(result.boxes) == 0:
        return out
    xyxy = result.boxes.xyxy.detach().cpu().numpy()
    conf = result.boxes.conf.detach().cpu().numpy()
    cls = result.boxes.cls.detach().cpu().numpy().astype(int)
    for (x1, y1, x2, y2), c, k in zip(xyxy, conf, cls):
        out.append(Box(x1, y1, x2, y2, c, k))
    return out


class ExportThread(QThread):
    """后台整理数据集 (调用 export_dataset.build_dataset)。"""
    progress = pyqtSignal(int, int, str)
    done = pyqtSignal(dict)
    error = pyqtSignal(str)

    def __init__(self, src, out, val_split, parent=None):
        super().__init__(parent)
        self.src = src
        self.out = out
        self.val_split = val_split

    def run(self):
        try:
            stats = export_dataset.build_dataset(
                self.src, self.out, self.val_split,
                classes=None, seed=0,
                progress_cb=lambda d, t, m: self.progress.emit(d, t, m),
            )
            self.done.emit(stats)
        except Exception:
            self.error.emit(traceback.format_exc())


# ---------------------------------------------------------------------------
# 画布上的框 (QGraphicsItem)
# ---------------------------------------------------------------------------
HANDLE_R = 6  # 手柄命中半径 (场景像素)


class BoxItem(QGraphicsItem):
    """可移动 / 四角缩放的检测框。"""

    def __init__(self, cls_id, conf, class_name, x1, y1, x2, y2):
        super().__init__()
        self.cls = int(cls_id)
        self.conf = float(conf)
        self.class_name = class_name
        self._w = float(x2 - x1)
        self._h = float(y2 - y1)
        self.setPos(float(x1), float(y1))
        self.setFlags(QGraphicsItem.ItemIsMovable
                      | QGraphicsItem.ItemSendsGeometryChanges)
        self.setAcceptHoverEvents(True)
        self.selected = False
        self._handle = None          # 'tl','tr','br','bl' 之一
        self._resizing = False
        self._fixed_scene = None     # 缩放时固定的对角点 (场景坐标)

    # ---- 几何 ----
    def local_rect(self):
        return QRectF(0, 0, self._w, self._h)

    def scene_rect(self):
        return QRectF(self.pos().x(), self.pos().y(), self._w, self._h)

    def boundingRect(self):
        return QRectF(-HANDLE_R - 1, -HANDLE_R - 1,
                      self._w + 2 * HANDLE_R + 2, self._h + 2 * HANDLE_R + 2)

    def to_box(self):
        r = self.scene_rect()
        return Box(r.left(), r.top(), r.right(), r.bottom(), self.conf, self.cls)

    # ---- 绘制 ----
    def paint(self, painter, option, widget=None):
        color = class_color(self.cls)
        pen = QPen(color, 2)
        if self.selected:
            pen.setWidth(3)
        painter.setPen(pen)
        painter.setBrush(QBrush(QColor(color.red(), color.green(), color.blue(), 30)))
        painter.drawRect(self.local_rect())

        # 标签文字 (绘制在框内左上角)
        label = f"{self.class_name} {self.conf:.2f}" if self.conf >= 0 else f"{self.class_name} (手动)"
        font = QFont()
        font.setPointSize(9)
        painter.setFont(font)
        fm = painter.fontMetrics()
        tw = fm.horizontalAdvance(label)
        th = fm.height()
        txt_rect = QRectF(1, 1, tw + 6, th + 2)
        painter.setBrush(QBrush(color))
        painter.setPen(Qt.NoPen)
        painter.drawRect(txt_rect)
        painter.setPen(QColor(255, 255, 255))
        painter.drawText(txt_rect, Qt.AlignCenter, label)

        # 选中时绘制四角手柄
        if self.selected:
            painter.setBrush(QBrush(QColor(255, 255, 255)))
            painter.setPen(QPen(QColor(30, 30, 30), 1))
            for p in self._handle_points_local():
                painter.drawRect(QRectF(p.x() - HANDLE_R, p.y() - HANDLE_R,
                                        2 * HANDLE_R, 2 * HANDLE_R))

    def _handle_points_local(self):
        return [QPointF(0, 0), QPointF(self._w, 0),
                QPointF(self._w, self._h), QPointF(0, self._h)]

    _HANDLE_NAMES = ('tl', 'tr', 'br', 'bl')

    def _hit_handle(self, lp):
        for name, p in zip(self._HANDLE_NAMES, self._handle_points_local()):
            if (abs(lp.x() - p.x()) <= HANDLE_R + 1 and
                    abs(lp.y() - p.y()) <= HANDLE_R + 1):
                return name
        return None

    def _opposite_scene(self, name):
        r = self.scene_rect()
        return {'tl': r.bottomRight(), 'tr': r.bottomLeft(),
                'br': r.topLeft(), 'bl': r.topRight()}[name]

    # ---- 交互 ----
    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            lp = event.pos()
            h = self._hit_handle(lp)
            if self.selected and h:
                self._handle = h
                self._resizing = True
                self._fixed_scene = self._opposite_scene(h)
                event.accept()
                return
            if self.local_rect().contains(lp) or self.boundingRect().contains(lp):
                # 交给 ItemIsMovable 处理移动
                self.setCursor(Qt.ClosedHandCursor)
                super().mousePressEvent(event)
                event.accept()
                return
            event.ignore()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._resizing and self._handle:
            sp = self.mapToScene(event.pos())
            self._resize_to(sp)
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if self._resizing:
            self._resizing = False
            self._handle = None
            self._fixed_scene = None
            self.setCursor(Qt.ArrowCursor)
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def _resize_to(self, cursor_scene):
        fx, fy = self._fixed_scene.x(), self._fixed_scene.y()
        cx, cy = cursor_scene.x(), cursor_scene.y()
        x1, y1 = min(fx, cx), min(fy, cy)
        x2, y2 = max(fx, cx), max(fy, cy)
        self.prepareGeometryChange()
        self._w = max(2.0, x2 - x1)
        self._h = max(2.0, y2 - y1)
        self.setPos(x1, y1)

    def itemChange(self, change, value):
        if change == QGraphicsItem.ItemPositionHasChanged:
            pass
        return super().itemChange(change, value)


# ---------------------------------------------------------------------------
# 画布 (QGraphicsView)
# ---------------------------------------------------------------------------
class Canvas(QGraphicsView):
    box_created = pyqtSignal(object)          # 新建框 -> BoxItem
    box_selected = pyqtSignal(object, bool)   # 选中变化 -> (BoxItem|None, has_selection)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)
        self.setRenderHints(QPainter.Antialiasing | QPainter.SmoothPixmapTransform)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorViewCenter)
        self.setMouseTracking(True)
        self.setBackgroundBrush(QColor(40, 40, 40))

        self.image_item = None
        self.classes = ['object']
        self.current_class = 0

        self._drawing = False
        self._rubber = None
        self._rubber_start = None
        self._panning = False
        self._pan_last = None
        self._space = False

        self.selected_box = None
        self._img_size = (1, 1)

    # ---- 图像与框 ----
    def set_image(self, bgr_np, fit=True):
        self._scene.clear()
        self.image_item = None
        self.selected_box = None
        if bgr_np is None:
            self._img_size = (1, 1)
            return
        self._img_size = (bgr_np.shape[1], bgr_np.shape[0])
        pm = cv2_to_qpixmap(bgr_np)
        self.image_item = self._scene.addPixmap(pm)
        self.image_item.setZValue(-1)
        self._scene.setSceneRect(0, 0, pm.width(), pm.height())
        if fit:
            self.fit_to_window()

    def clear_boxes(self):
        for it in list(self._scene.items()):
            if isinstance(it, BoxItem):
                self._scene.removeItem(it)
        self.selected_box = None

    def add_box(self, box, class_name):
        bi = BoxItem(box.cls, box.conf, class_name, box.x1, box.y1, box.x2, box.y2)
        bi.setZValue(0)
        self._scene.addItem(bi)
        return bi

    def all_boxes(self):
        return [it for it in self._scene.items() if isinstance(it, BoxItem)]

    def set_selected(self, box_item):
        if self.selected_box is box_item:
            return
        if self.selected_box is not None:
            self.selected_box.selected = False
            self.selected_box.update()
        self.selected_box = box_item
        if box_item is not None:
            box_item.selected = True
            box_item.update()
        self.box_selected.emit(box_item, box_item is not None)

    def delete_selected(self):
        if self.selected_box is not None:
            self._scene.removeItem(self.selected_box)
            self.selected_box = None
            self.box_selected.emit(None, False)
            return True
        return False

    # ---- 缩放 / 平移 ----
    def fit_to_window(self):
        r = self._scene.itemsBoundingRect()
        if not r.isNull():
            self.fitInView(r, Qt.KeepAspectRatio)

    def wheelEvent(self, event):
        factor = 1.25 if event.angleDelta().y() > 0 else 1 / 1.25
        self.scale(factor, factor)

    def _box_at(self, scene_pos):
        for it in self._scene.items(scene_pos):
            if isinstance(it, BoxItem):
                return it
        return None

    # ---- 鼠标 ----
    def mousePressEvent(self, event):
        if event.button() == Qt.MiddleButton or (event.button() == Qt.LeftButton and self._space):
            self._panning = True
            self._pan_last = event.pos()
            self.setCursor(Qt.ClosedHandCursor)
            event.accept()
            return

        if event.button() == Qt.LeftButton:
            sp = self.mapToScene(event.pos())
            box = self._box_at(sp)
            if box is not None:
                self.set_selected(box)
                # 交给 item 处理移动 / 缩放
                super().mousePressEvent(event)
                return
            # 空白处: 开始画新框
            self._drawing = True
            self._rubber_start = sp
            self._rubber = QGraphicsRectItem(QRectF(sp, sp))
            pen = QPen(QColor(0, 200, 255), 1, Qt.DashLine)
            self._rubber.setPen(pen)
            self._scene.addItem(self._rubber)
            event.accept()
            return

        if event.button() == Qt.RightButton:
            sp = self.mapToScene(event.pos())
            box = self._box_at(sp)
            if box is not None:
                self._scene.removeItem(box)
                if self.selected_box is box:
                    self.selected_box = None
                    self.box_selected.emit(None, False)
            event.accept()
            return

        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._panning and self._pan_last is not None:
            delta = event.pos() - self._pan_last
            self._pan_last = event.pos()
            self.horizontalScrollBar().setValue(self.horizontalScrollBar().value() - delta.x())
            self.verticalScrollBar().setValue(self.verticalScrollBar().value() - delta.y())
            event.accept()
            return

        if self._drawing and self._rubber is not None:
            sp = self.mapToScene(event.pos())
            x1, y1 = self._rubber_start.x(), self._rubber_start.y()
            self._rubber.setRect(QRectF(x1, y1, sp.x() - x1, sp.y() - y1).normalized())
            event.accept()
            return

        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MiddleButton or self._panning:
            self._panning = False
            self._pan_last = None
            self.unsetCursor()
            event.accept()
            return

        if self._drawing and event.button() == Qt.LeftButton:
            self._drawing = False
            if self._rubber is not None:
                r = self._rubber.rect()
                self._scene.removeItem(self._rubber)
                self._rubber = None
                if r.width() > 3 and r.height() > 3:
                    cn = self.classes[self.current_class] if self.current_class < len(self.classes) else 'object'
                    box = Box(r.left(), r.top(), r.right(), r.bottom(), -1, self.current_class)
                    bi = self.add_box(box, cn)
                    bi.setZValue(0)
                    self.set_selected(bi)
                    self.box_created.emit(bi)
            event.accept()
            return

        super().mouseReleaseEvent(event)

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Space:
            self._space = True
            self.setCursor(Qt.OpenHandCursor)
        super().keyPressEvent(event)

    def keyReleaseEvent(self, event):
        if event.key() == Qt.Key_Space:
            self._space = False
            self.unsetCursor()
        super().keyReleaseEvent(event)


def cv2_to_qpixmap(bgr):
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    h, w, ch = rgb.shape
    qimg = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
    return QPixmap.fromImage(qimg).copy()


# ---------------------------------------------------------------------------
# 主窗口
# ---------------------------------------------------------------------------
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("自动标注工具 - YOLOv8 (mosquito)")
        self.resize(1280, 800)

        self.model_path = ""
        self.classes = ['mosquito']          # 默认; 载入模型后更新
        self.folder = ""
        self.items = []                       # list[Item]
        self.current_index = -1
        self._model = None

        self._build_ui()
        self._build_menu()

        # 默认模型路径
        default_model = r"D:\Data of Experiment\训练\SLM-AFPN-C2f_SCConv\weights\best.pt"
        if os.path.exists(default_model):
            self.model_path = default_model
            self.model_path_label.setText(os.path.basename(default_model))

        self._load_model()   # 提前加载模型 (后台)

        QShortcut(QKeySequence(Qt.Key_Delete), self, activated=self._delete_selected)
        QShortcut(QKeySequence(Qt.Key_F), self, activated=self.canvas.fit_to_window)

    # ------------------------------------------------------------------ UI
    def _build_ui(self):
        # 工具栏
        tb = QToolBar("主工具栏")
        tb.setMovable(False)
        self.addToolBar(tb)

        self.act_select_model = QAction("选择模型", self)
        self.act_select_model.triggered.connect(self._choose_model)
        tb.addAction(self.act_select_model)

        self.model_path_label = QLabel("未选择")
        self.model_path_label.setStyleSheet("color:#888;")
        tb.addWidget(self.model_path_label)

        tb.addSeparator()

        self.act_select_folder = QAction("选择文件夹", self)
        self.act_select_folder.triggered.connect(self._choose_folder)
        tb.addAction(self.act_select_folder)

        self.folder_label = QLabel("未选择")
        self.folder_label.setStyleSheet("color:#888;")
        tb.addWidget(self.folder_label)

        self.act_load = QAction("读取已有标签", self)
        self.act_load.triggered.connect(self._on_load_labels)
        tb.addAction(self.act_load)

        tb.addSeparator()

        tb.addWidget(QLabel("  置信度:"))
        self.conf_spin = QDoubleSpinBox()
        self.conf_spin.setRange(0.01, 0.95)
        self.conf_spin.setSingleStep(0.05)
        self.conf_spin.setValue(0.25)
        tb.addWidget(self.conf_spin)

        tb.addWidget(QLabel("  IoU:"))
        self.iou_spin = QDoubleSpinBox()
        self.iou_spin.setRange(0.05, 0.95)
        self.iou_spin.setSingleStep(0.05)
        self.iou_spin.setValue(0.5)
        tb.addWidget(self.iou_spin)

        tb.addWidget(QLabel("  视频帧间隔:"))
        self.vid_stride_spin = QSpinBox()
        self.vid_stride_spin.setRange(1, 100)
        self.vid_stride_spin.setValue(1)
        tb.addWidget(self.vid_stride_spin)

        tb.addSeparator()

        self.act_run = QAction("批量标注", self)
        self.act_run.triggered.connect(self._run_batch)
        tb.addAction(self.act_run)

        self.act_save = QAction("保存全部", self)
        self.act_save.triggered.connect(self._save_all)
        tb.addAction(self.act_save)

        self.act_export = QAction("导出可视化", self)
        self.act_export.triggered.connect(self._export_visual)
        tb.addAction(self.act_export)

        self.act_dataset = QAction("导出训练集", self)
        self.act_dataset.triggered.connect(self._export_dataset)
        tb.addAction(self.act_dataset)

        # 中央画布
        self.canvas = Canvas(self)
        self.canvas.classes = self.classes
        self.canvas.box_selected.connect(self._on_box_selected)
        self.setCentralWidget(self.canvas)

        # 左侧 dock: 文件列表
        left = QDockWidget("文件列表", self)
        left_widget = QWidget()
        lv = QVBoxLayout(left_widget)
        lv.setContentsMargins(4, 4, 4, 4)
        self.file_list = QListWidget()
        self.file_list.currentRowChanged.connect(self._on_file_changed)
        lv.addWidget(self.file_list)
        self.save_to_labels_check = QCheckBox("保存到 labels/ 子文件夹")
        self.save_to_labels_check.setChecked(True)
        lv.addWidget(self.save_to_labels_check)
        left.setWidget(left_widget)
        self.addDockWidget(Qt.LeftDockWidgetArea, left)

        # 右侧 dock: 类别 + 标注信息
        right = QDockWidget("类别与标注", self)
        rw = QWidget()
        rv = QVBoxLayout(rw)
        rv.setContentsMargins(4, 4, 4, 4)

        g1 = QGroupBox("类别")
        g1l = QVBoxLayout(g1)
        self.class_combo = QComboBox()
        self.class_combo.currentIndexChanged.connect(self._on_class_changed)
        g1l.addWidget(self.class_combo)
        rv.addWidget(g1)

        g2 = QGroupBox("选中框信息")
        g2l = QGridLayout(g2)
        self.info_label = QLabel("未选中")
        self.info_label.setWordWrap(True)
        g2l.addWidget(self.info_label, 0, 0, 1, 2)
        self.box_class_combo = QComboBox()
        self.box_class_combo.currentIndexChanged.connect(self._on_box_class_changed)
        g2l.addWidget(QLabel("类别:"), 1, 0)
        g2l.addWidget(self.box_class_combo, 1, 1)
        btn_del = QPushButton("删除选中框 (Del)")
        btn_del.clicked.connect(self._delete_selected)
        g2l.addWidget(btn_del, 2, 0, 1, 2)
        rv.addWidget(g2)

        # 视频导航
        g3 = QGroupBox("视频帧导航")
        g3l = QVBoxLayout(g3)
        nav_row = QHBoxLayout()
        self.btn_prev = QPushButton("上一帧")
        self.btn_prev.clicked.connect(lambda: self._seek_video(-1))
        self.btn_next = QPushButton("下一帧")
        self.btn_next.clicked.connect(lambda: self._seek_video(1))
        self.btn_play = QPushButton("播放")
        self.btn_play.clicked.connect(self._toggle_play)
        nav_row.addWidget(self.btn_prev)
        nav_row.addWidget(self.btn_play)
        nav_row.addWidget(self.btn_next)
        g3l.addLayout(nav_row)
        self.frame_slider = QSlider(Qt.Horizontal)
        self.frame_slider.valueChanged.connect(self._on_frame_slider)
        g3l.addWidget(self.frame_slider)
        self.frame_label = QLabel("帧: -/-")
        g3l.addWidget(self.frame_label)
        rv.addWidget(g3)
        rv.addStretch(1)

        right.setWidget(rw)
        self.addDockWidget(Qt.RightDockWidgetArea, right)

        # 状态栏
        self.status = self.statusBar()
        self.progress = QProgressBar()
        self.progress.setMaximumWidth(250)
        self.progress.setVisible(False)
        self.status.addPermanentWidget(self.progress)
        self.status_label = QLabel("就绪")
        self.status.addWidget(self.status_label)

        self._video_timer = QTimer(self)
        self._video_timer.timeout.connect(lambda: self._seek_video(1, from_timer=True))
        self._playing = False

        self._update_video_controls()

    def _build_menu(self):
        m = self.menuBar().addMenu("帮助")
        act = m.addAction("使用说明")
        act.triggered.connect(self._show_help)

    # ------------------------------------------------------------- 模型 / 文件夹
    def _choose_model(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "选择模型权重", self.model_path or ROOT, "PyTorch 模型 (*.pt *.pth);;所有文件 (*)")
        if path:
            self.model_path = path
            self.model_path_label.setText(os.path.basename(path))
            self._load_model()

    def _load_model(self):
        if not self.model_path:
            return
        self.status_label.setText("正在加载模型...")
        self._model = None
        self._model_loader = ModelLoader(self.model_path)
        self._model_loader.loaded.connect(self._on_model_loaded)
        self._model_loader.error.connect(self._on_model_error)
        self._model_loader.start()

    def _on_model_loaded(self, model, names):
        self._model = model
        if names:
            self.classes = names
        self.canvas.classes = self.classes
        self._refresh_class_combo()
        self.status_label.setText(
            f"模型已加载: {os.path.basename(self.model_path)} (类别: {', '.join(self.classes)})")

    def _on_model_error(self, msg):
        self.status_label.setText("模型加载失败")
        QMessageBox.critical(self, "错误", f"模型加载失败:\n{msg}")

    def _refresh_class_combo(self):
        cur = self.class_combo.currentText()
        self.class_combo.blockSignals(True)
        self.class_combo.clear()
        self.class_combo.addItems(self.classes)
        if cur in self.classes:
            self.class_combo.setCurrentText(cur)
        self.class_combo.blockSignals(False)

        self.box_class_combo.blockSignals(True)
        self.box_class_combo.clear()
        self.box_class_combo.addItems(self.classes)
        self.box_class_combo.blockSignals(False)

    def _choose_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "选择包含图片/视频的文件夹", self.folder or ROOT)
        if folder:
            self.folder = folder
            self.folder_label.setText(folder)
            self._scan_folder()

    def _scan_folder(self):
        self.items = []
        paths = []
        for ext in IMAGE_EXTS:
            paths += glob.glob(os.path.join(self.folder, '*' + ext))
        image_paths = sorted(paths)
        video_paths = []
        for ext in VIDEO_EXTS:
            video_paths += sorted(glob.glob(os.path.join(self.folder, '*' + ext)))

        for p in image_paths:
            self.items.append(Item(p, 'image'))
        for p in video_paths:
            it = Item(p, 'video')
            cap = cv2.VideoCapture(p)
            it.frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
            it.fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
            cap.release()
            self.items.append(it)

        # 自动读取上次保存的标签, 方便继续之前的操作
        n_loaded = self._load_saved_labels()
        self._refresh_file_list()
        msg = f"已扫描 {len(self.items)} 个文件 (图片 {len(image_paths)}, 视频 {len(video_paths)})"
        if n_loaded:
            msg += f", 已读取 {n_loaded} 个文件的已有标签"
        self.status_label.setText(msg)

    # ------------------------------------------------------------- 读取已有标签
    def _parse_label_file(self, path, W, H):
        """解析一个 YOLO 标签文件 (cls cx cy w h) 为 Box 列表。conf=-1 表示无置信度信息。"""
        boxes = []
        try:
            with open(path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    p = line.split()
                    if len(p) < 5:
                        continue
                    try:
                        cls = int(float(p[0]))
                        cx, cy, w, h = float(p[1]), float(p[2]), float(p[3]), float(p[4])
                    except ValueError:
                        continue
                    x1 = (cx - w / 2.0) * W
                    y1 = (cy - h / 2.0) * H
                    x2 = (cx + w / 2.0) * W
                    y2 = (cy + h / 2.0) * H
                    boxes.append(Box(x1, y1, x2, y2, -1.0, cls))
        except OSError:
            return []
        return boxes

    def _load_saved_labels(self):
        """读取 labels 目录里的标签, 恢复到 self.items, 返回恢复的文件数。"""
        if not self.folder or not self.items:
            return 0
        labels_dir = os.path.join(self.folder, 'labels')
        if not os.path.isdir(labels_dir):
            labels_dir = self.folder  # 兼容"保存到同目录"的情况

        # 恢复类别名
        ctxt = os.path.join(labels_dir, 'classes.txt')
        if os.path.exists(ctxt):
            names = []
            try:
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
                        while len(names) <= idx:
                            names.append(str(len(names)))
                        names[idx] = parts[1] if len(parts) > 1 else str(idx)
            except OSError:
                names = []
            if names:
                self.classes = names
                self.canvas.classes = self.classes
                self._refresh_class_combo()

        restored = 0
        for it in self.items:
            if it.kind == 'image':
                lab = os.path.join(labels_dir, it.stem + '.txt')
                if not os.path.exists(lab):
                    continue
                W, H = export_dataset.image_size_unicode(it.path)
                if W <= 0 or H <= 0:
                    continue
                boxes = self._parse_label_file(lab, W, H)
                if boxes:
                    it.boxes = boxes
                    restored += 1
            else:
                vdir = os.path.join(labels_dir, it.stem)
                if not os.path.isdir(vdir):
                    continue
                cap = cv2.VideoCapture(it.path)
                W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                cap.release()
                if W <= 0 or H <= 0:
                    continue
                n_frames = 0
                for ftxt in sorted(glob.glob(os.path.join(vdir, 'frame_*.txt'))):
                    fn = os.path.basename(ftxt)[len('frame_'):-len('.txt')]
                    try:
                        fidx = int(fn)
                    except ValueError:
                        continue
                    boxes = self._parse_label_file(ftxt, W, H)
                    if boxes:
                        it.frames[fidx] = boxes
                        n_frames += 1
                if n_frames:
                    restored += 1
        return restored

    def _on_load_labels(self):
        if not self.folder or not self.items:
            QMessageBox.information(self, "提示", "请先选择文件夹")
            return
        n = self._load_saved_labels()
        self._refresh_file_list(
            select=self.current_index if self.current_index >= 0 else None)
        if 0 <= self.current_index < len(self.items):
            self._load_item_to_canvas(self.items[self.current_index])
        self.status_label.setText(
            f"已读取 {n} 个文件的已有标签" if n else "未找到已有标签 (labels/)")

    # ------------------------------------------------------------- 文件列表
    def _refresh_file_list(self, select=None):
        self.file_list.blockSignals(True)
        self.file_list.clear()
        for it in self.items:
            if it.kind == 'image':
                n = len(it.boxes)
                tag = f"[{n}]" if n else "   "
            else:
                n = sum(len(v) for v in it.frames.values())
                tag = f"[{n}]" if n else "   "
            label = f"{tag} {'🎬' if it.kind == 'video' else '🖼'} {os.path.basename(it.path)}"
            item = QListWidgetItem(label)
            item.setData(Qt.UserRole, it.path)
            self.file_list.addItem(item)
        self.file_list.blockSignals(False)
        if select is not None and 0 <= select < len(self.items):
            self.file_list.setCurrentRow(select)
        elif self.items:
            self.file_list.setCurrentRow(0)

    def _on_file_changed(self, row):
        if row < 0 or row >= len(self.items):
            return
        self._stop_play()  # 切换文件前先停掉播放, 否则定时器会疯狂重载新文件
        self._save_current_to_item()
        self.current_index = row
        self._load_item_to_canvas(self.items[row])
        self._update_video_controls()

    # ------------------------------------------------------------- 画布加载 / 同步
    def _load_item_to_canvas(self, it, fit=True):
        self.canvas.clear_boxes()
        if it.kind == 'image':
            img = export_dataset.imread_unicode(it.path)
            self.canvas.set_image(img, fit=fit)
            cn = self.classes[0] if self.classes else 'object'
            for b in it.boxes:
                name = self.classes[b.cls] if b.cls < len(self.classes) else 'object'
                self.canvas.add_box(b, name)
        else:  # video
            img = self._read_frame(it, getattr(it, 'cur_frame', 0))
            self.canvas.set_image(img, fit=fit)
            for b in it.frames.get(getattr(it, 'cur_frame', 0), []):
                name = self.classes[b.cls] if b.cls < len(self.classes) else 'object'
                self.canvas.add_box(b, name)
            self._sync_video_slider()

    def _read_frame(self, it, idx):
        if it.cap is None:
            it.cap = cv2.VideoCapture(it.path)
            it.cap_pos = -1
        # 顺序读取: 目标帧正好是下一帧时直接 read, 避免昂贵的 seek (4K H.264 seek 极慢)
        if idx == it.cap_pos + 1:
            ok, frame = it.cap.read()
            if ok:
                it.cap_pos = idx
                return frame
            # 顺序读取失败 (到末尾/文件异常) 回退到 seek
        it.cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = it.cap.read()
        if ok:
            it.cap_pos = idx
        return frame if ok else None

    def _save_current_to_item(self):
        """把画布上的框写回当前 item。"""
        if self.current_index < 0:
            return
        it = self.items[self.current_index]
        boxes = [bi.to_box() for bi in self.canvas.all_boxes()]
        if it.kind == 'image':
            it.boxes = boxes
        else:
            frame = getattr(it, 'cur_frame', 0)
            it.frames[frame] = boxes

    # ------------------------------------------------------------- 推理
    def _run_batch(self):
        if not self.items:
            QMessageBox.information(self, "提示", "请先选择文件夹")
            return
        if not self.model_path:
            QMessageBox.information(self, "提示", "请先选择模型权重")
            return

        if self._model is None:
            QMessageBox.information(self, "提示", "模型尚未加载完成, 请稍候")
            return

        images = [(i, it.path) for i, it in enumerate(self.items) if it.kind == 'image']
        videos = [(i, it.path) for i, it in enumerate(self.items) if it.kind == 'video']

        self.act_run.setEnabled(False)
        self.progress.setVisible(True)
        self.progress.setValue(0)

        self.thread = InferenceThread(
            self._model, images, videos,
            self.conf_spin.value(), self.iou_spin.value(),
            640, self.vid_stride_spin.value(),
        )
        self.thread.image_result.connect(self._on_image_result)
        self.thread.video_result.connect(self._on_video_result)
        self.thread.progress.connect(self._on_progress)
        self.thread.done.connect(self._on_inference_done)
        self.thread.error.connect(self._on_inference_error)
        self.thread.warn.connect(self._on_inference_warn)
        self.thread.start()

    def _on_inference_warn(self, msg):
        self.status_label.setText(msg)

    def _on_image_result(self, idx, boxes):
        self.items[idx].boxes = boxes
        self._update_list_item(idx)

    def _on_video_result(self, idx, frame_boxes):
        self.items[idx].frames = frame_boxes
        self._update_list_item(idx)

    def _update_list_item(self, idx):
        it = self.items[idx]
        n = len(it.boxes) if it.kind == 'image' else sum(len(v) for v in it.frames.values())
        tag = f"[{n}]" if n else "   "
        label = f"{tag} {'🎬' if it.kind == 'video' else '🖼'} {os.path.basename(it.path)}"
        item = self.file_list.item(idx)
        if item:
            item.setText(label)

    def _on_progress(self, done, total, msg):
        self.progress.setMaximum(max(1, total))
        self.progress.setValue(done)
        self.status_label.setText(f"推理中 {done}/{total} - {msg}")

    def _on_inference_done(self):
        self.progress.setVisible(False)
        self.act_run.setEnabled(True)
        self.status_label.setText("批量标注完成")
        if self.current_index >= 0:
            self._load_item_to_canvas(self.items[self.current_index])
        self._refresh_file_list(select=self.current_index)

    def _on_inference_error(self, msg):
        self.progress.setVisible(False)
        self.act_run.setEnabled(True)
        self.status_label.setText("推理出错")
        QMessageBox.critical(self, "错误", msg)

    # ------------------------------------------------------------- 编辑
    def _on_box_selected(self, box, has):
        if not has or box is None:
            self.info_label.setText("未选中")
            self.box_class_combo.setEnabled(False)
            return
        r = box.scene_rect()
        self.info_label.setText(
            f"类别: {self.classes[box.cls] if box.cls < len(self.classes) else box.cls}\n"
            f"置信度: {box.conf:.3f}\n"
            f"x1={r.left():.1f} y1={r.top():.1f}\n"
            f"x2={r.right():.1f} y2={r.bottom():.1f}\n"
            f"w={r.width():.1f} h={r.height():.1f}"
        )
        self.box_class_combo.setEnabled(True)
        self.box_class_combo.blockSignals(True)
        self.box_class_combo.setCurrentIndex(box.cls)
        self.box_class_combo.blockSignals(False)

    def _on_class_changed(self, idx):
        if 0 <= idx < len(self.classes):
            self.canvas.current_class = idx

    def _on_box_class_changed(self, idx):
        if self.canvas.selected_box is not None and 0 <= idx < len(self.classes):
            self.canvas.selected_box.cls = idx
            self.canvas.selected_box.class_name = self.classes[idx]
            self.canvas.selected_box.update()
            self._on_box_selected(self.canvas.selected_box, True)

    def _delete_selected(self):
        if self.canvas.delete_selected():
            self.status_label.setText("已删除框")

    # ------------------------------------------------------------- 视频导航
    def _update_video_controls(self):
        is_video = (0 <= self.current_index < len(self.items)
                    and self.items[self.current_index].kind == 'video')
        for w in (self.btn_prev, self.btn_next, self.btn_play, self.frame_slider):
            w.setEnabled(is_video)
        if is_video:
            self._sync_video_slider()
        else:
            self.frame_label.setText("帧: -/-")

    def _sync_video_slider(self):
        it = self.items[self.current_index]
        cur = getattr(it, 'cur_frame', 0)
        self.frame_slider.blockSignals(True)
        self.frame_slider.setRange(0, max(0, it.frame_count - 1))
        self.frame_slider.setValue(cur)
        self.frame_slider.blockSignals(False)
        self.frame_label.setText(f"帧: {cur}/{it.frame_count - 1}")

    def _on_frame_slider(self, v):
        it = self.items[self.current_index]
        self._save_current_to_item()
        it.cur_frame = v
        self._load_item_to_canvas(it, fit=False)
        self.frame_label.setText(f"帧: {v}/{it.frame_count - 1}")

    def _stop_play(self):
        if self._playing:
            self._playing = False
            self._video_timer.stop()
            self.btn_play.setText("播放")

    def _seek_video(self, delta, from_timer=False):
        it = self.items[self.current_index]
        if it.kind != 'video':
            self._stop_play()
            return
        cur = getattr(it, 'cur_frame', 0)
        nxt = max(0, min(it.frame_count - 1, cur + delta))
        # 播放到最后一帧时自动停止, 避免定时器反复重载最后一帧
        if from_timer and nxt == cur:
            self._stop_play()
            return
        self._save_current_to_item()
        it.cur_frame = nxt
        self._load_item_to_canvas(it, fit=False)

    def _toggle_play(self):
        if self._playing:
            self._stop_play()
        else:
            self._playing = True
            self.btn_play.setText("暂停")
            it = self.items[self.current_index]
            fps = getattr(it, 'fps', 25.0) or 25.0
            self._video_timer.start(max(1, int(round(1000.0 / fps))))

    # ------------------------------------------------------------- 保存
    def _labels_dir(self):
        base = self.folder
        if self.save_to_labels_check.isChecked():
            return os.path.join(base, 'labels')
        return base

    def _save_all(self):
        if not self.items:
            QMessageBox.information(self, "提示", "没有文件可保存")
            return
        self._save_current_to_item()
        labels_dir = self._labels_dir()
        os.makedirs(labels_dir, exist_ok=True)

        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            n_img = n_vid = 0
            for it in self.items:
                if it.kind == 'image':
                    self._write_image_labels(it, labels_dir)
                    n_img += 1
                else:
                    self._write_video_labels(it, labels_dir)
                    n_vid += 1
                QApplication.processEvents()

            # classes.txt
            with open(os.path.join(labels_dir, 'classes.txt'), 'w', encoding='utf-8') as f:
                for i, c in enumerate(self.classes):
                    f.write(f"{i} {c}\n")
        finally:
            QApplication.restoreOverrideCursor()

        self.status_label.setText(f"已保存: 图片 {n_img} 张, 视频 {n_vid} 个 -> {labels_dir}")

    def _write_image_labels(self, it, labels_dir):
        if not it.boxes:
            return
        W, H = export_dataset.image_size_unicode(it.path)
        if W <= 0 or H <= 0:
            return
        txt = os.path.join(labels_dir, it.stem + '.txt')
        with open(txt, 'w', encoding='utf-8') as f:
            for b in it.boxes:
                f.write(b.to_yolo(W, H) + '\n')

    def _write_video_labels(self, it, labels_dir):
        if not it.frames:
            return
        vdir = os.path.join(labels_dir, it.stem)
        os.makedirs(vdir, exist_ok=True)
        # 用视频头里的宽高, 不必逐帧解码取尺寸
        cap = cv2.VideoCapture(it.path)
        W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        if W <= 0 or H <= 0:
            for idx in it.frames:
                frame = self._read_frame(it, idx)
                if frame is not None:
                    H, W = frame.shape[:2]
                    break
        if W <= 0 or H <= 0:
            return
        for idx, boxes in it.frames.items():
            if not boxes:
                continue
            with open(os.path.join(vdir, f"frame_{idx:06d}.txt"), 'w', encoding='utf-8') as f:
                for b in boxes:
                    f.write(b.to_yolo(W, H) + '\n')

    # ------------------------------------------------------------- 导出可视化
    def _export_visual(self):
        if not self.items:
            return
        self._save_current_to_item()
        out_dir = os.path.join(self.folder, 'output')
        os.makedirs(out_dir, exist_ok=True)
        self.status_label.setText("正在导出可视化...")
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            for it in self.items:
                if it.kind == 'image':
                    self._export_image(it, out_dir)
                else:
                    self._export_video(it, out_dir)
                QApplication.processEvents()
        finally:
            QApplication.restoreOverrideCursor()
        self.status_label.setText(f"导出完成 -> {out_dir}")

    def _draw_boxes(self, img, boxes):
        for b in boxes:
            c = class_color(b.cls)
            color = (c.red(), c.green(), c.blue())
            cv2.rectangle(img, (int(b.x1), int(b.y1)), (int(b.x2), int(b.y2)), color, 2)
            name = self.classes[b.cls] if b.cls < len(self.classes) else 'object'
            txt = f"{name} {b.conf:.2f}" if b.conf >= 0 else f"{name}"
            cv2.putText(img, txt, (int(b.x1), max(15, int(b.y1) - 4)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        return img

    def _export_image(self, it, out_dir):
        img = export_dataset.imread_unicode(it.path)
        if img is None:
            return
        self._draw_boxes(img, it.boxes)
        export_dataset.imwrite_unicode(os.path.join(out_dir, it.stem + '.jpg'), img)

    def _export_video(self, it, out_dir):
        if not it.frames or it.frame_count <= 0:
            return
        cap = cv2.VideoCapture(it.path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        writer = cv2.VideoWriter(os.path.join(out_dir, it.stem + '.mp4'),
                                 cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
        for i in range(it.frame_count):
            ok, frame = cap.read()
            if not ok:
                break
            if i in it.frames:
                self._draw_boxes(frame, it.frames[i])
            writer.write(frame)
        writer.release()
        cap.release()

    # ------------------------------------------------------------- 导出训练集
    def _export_dataset(self):
        if not self.folder or not self.items:
            QMessageBox.information(self, "提示", "请先选择并标注好文件夹")
            return
        # 先把当前修正结果落盘
        self._save_all()

        out = QFileDialog.getExistingDirectory(
            self, "选择输出数据集目录 (建议新建空目录)", self.folder)
        if not out:
            return
        # 避免直接写回源目录造成混乱
        if os.path.abspath(out) == os.path.abspath(self.folder):
            out = os.path.join(out, 'yolo_dataset')

        val_split, ok = QInputDialog.getDouble(
            self, "验证集比例", "验证集占比 (0~1):", 0.1, 0.0, 0.9, 2)
        if not ok:
            return

        self.act_dataset.setEnabled(False)
        self.progress.setVisible(True)
        self.progress.setValue(0)

        self._export_thread = ExportThread(self.folder, out, val_split)
        self._export_thread.progress.connect(self._on_export_progress)
        self._export_thread.done.connect(self._on_export_done)
        self._export_thread.error.connect(self._on_export_error)
        self._export_thread.start()

    def _on_export_progress(self, done, total, msg):
        self.progress.setMaximum(max(1, total))
        self.progress.setValue(done)
        self.status_label.setText(f"导出训练集 {done}/{total}")

    def _on_export_done(self, stats):
        self.progress.setVisible(False)
        self.act_dataset.setEnabled(True)
        self.status_label.setText("训练集导出完成")
        QMessageBox.information(
            self, "完成",
            f"训练集已导出到:\n{stats['out']}\n\n"
            f"图片 {stats['images']} 张, 视频帧 {stats['video_frames']} 帧\n"
            f"train {stats['train']}, val {stats['val']}\n\n"
            f"dataset.yaml:\n{stats['yaml']}")

    def _on_export_error(self, msg):
        self.progress.setVisible(False)
        self.act_dataset.setEnabled(True)
        self.status_label.setText("导出失败")
        QMessageBox.critical(self, "错误", msg)

    # ------------------------------------------------------------- 其它
    def _show_help(self):
        QMessageBox.information(self, "使用说明",
            "操作:\n"
            "  1. 选择模型 (默认已选 best.pt) 与文件夹。\n"
            "  2. 点击【批量标注】自动检测所有图片/视频。\n"
            "  3. 左侧列表切换文件, 画布查看效果。\n\n"
            "编辑:\n"
            "  左键拖动空白处 = 新建框\n"
            "  左键拖动框体   = 移动框\n"
            "  拖动四角手柄   = 调整大小\n"
            "  Delete / 右键  = 删除框\n"
            "  右侧类别下拉框  = 修改类别\n\n"
            "视图:\n"
            "  滚轮 = 缩放    Space+左键 或 中键 = 平移\n"
            "  F = 适应窗口\n\n"
            "保存:\n"
            "  【保存全部】写为 YOLO 标签 (labels/xxx.txt);\n"
            "  【导出可视化】生成标注图/标注视频到 output/。")

    def closeEvent(self, event):
        self._save_current_to_item()
        for it in self.items:
            if it.cap is not None:
                it.cap.release()
        event.accept()


# ---------------------------------------------------------------------------
def main():
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
