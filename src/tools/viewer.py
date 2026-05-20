import os
import cv2
import numpy as np
import re

from PyQt5.QtWidgets import (
    QApplication,
    QWidget,
    QLabel,
    QPushButton,
    QHBoxLayout,
    QVBoxLayout,
    QComboBox,
    QGraphicsView,
    QGraphicsScene,
    QGraphicsPixmapItem
)

from PyQt5.QtGui import QPixmap, QImage, QPainter
from PyQt5.QtCore import Qt

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


class ZoomableGraphicsView(QGraphicsView):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.scene = QGraphicsScene(self)
        self.setScene(self.scene)
        self.pixmap_item = QGraphicsPixmapItem()
        self.scene.addItem(self.pixmap_item)

        # 启用平滑缩放/渲染以保证放大后字迹清晰
        self.setRenderHint(QPainter.Antialiasing)
        self.setRenderHint(QPainter.SmoothPixmapTransform)

        # 启用鼠标左键拖拽平移
        self.setDragMode(QGraphicsView.ScrollHandDrag)

        # 设置缩放中心和调整中心都以鼠标光标为锚点
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorUnderMouse)

        # 隐藏滚动条让界面更清爽美观
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        # 缩放比例跟踪
        self.current_zoom = 1.0

    def setPixmap(self, pixmap):
        self.pixmap_item.setPixmap(pixmap)
        self.scene.setSceneRect(self.pixmap_item.boundingRect())
        
        # 初始加载时自适应大小
        self.fitInView(self.pixmap_item, Qt.KeepAspectRatio)
        self.current_zoom = self.transform().m11()

    def wheelEvent(self, event):
        zoom_in_factor = 1.15
        zoom_out_factor = 1.0 / zoom_in_factor

        if event.angleDelta().y() > 0:
            scale_factor = zoom_in_factor
        else:
            scale_factor = zoom_out_factor

        new_zoom = self.current_zoom * scale_factor

        # 限制放大倍数范围，避免过小或过大
        if 0.15 <= new_zoom <= 10.0:
            self.scale(scale_factor, scale_factor)
            self.current_zoom = new_zoom

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self.pixmap_item.pixmap() and not self.pixmap_item.pixmap().isNull():
            self.fitInView(self.pixmap_item, Qt.KeepAspectRatio)
            self.current_zoom = self.transform().m11()


class Viewer(QWidget):

    def __init__(self, root_dir):

        super().__init__()

        self.root_dir = root_dir
        self.splits = ["train", "val", "test"]
        self.current_split = "train"

        def extract_num(x):
            nums = re.findall(r"\d+", x)
            return int(nums[-1]) if nums else -1

        self.extract_num = extract_num

        self.i = 0

        self.init_ui()
        self.load_images()

    # ==================================================
    # load images for current split
    # ==================================================
    def load_images(self):
        img_dir = os.path.join(self.root_dir, "images", self.current_split)
        if os.path.exists(img_dir):
            self.imgs = sorted(os.listdir(img_dir), key=self.extract_num)
        else:
            self.imgs = []
        self.i = 0

    # ==================================================
    # UI
    # ==================================================
    def init_ui(self):

        self.setWindowTitle("Dataset Viewer")
        self.resize(1400, 800)

        self.img_label = ZoomableGraphicsView()
        self.mask_label = ZoomableGraphicsView()
        self.stats_label = QLabel()

        self.stats_label.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        self.stats_label.setWordWrap(True)

        self.split_combo = QComboBox()
        self.split_combo.addItems(self.splits)
        self.split_combo.setCurrentText(self.current_split)
        self.split_combo.currentTextChanged.connect(self.on_split_changed)

        self.prev_btn = QPushButton("Previous")
        self.next_btn = QPushButton("Next")

        self.prev_btn.clicked.connect(self.prev)
        self.next_btn.clicked.connect(self.next)

        top_layout = QHBoxLayout()
        top_layout.addWidget(self.img_label)
        top_layout.addWidget(self.mask_label)

        control_layout = QHBoxLayout()
        control_layout.addWidget(QLabel("Split:"))
        control_layout.addWidget(self.split_combo)
        control_layout.addStretch()
        control_layout.addWidget(self.prev_btn)
        control_layout.addWidget(self.next_btn)
        control_layout.addStretch()

        main_layout = QVBoxLayout()
        main_layout.addLayout(top_layout)
        main_layout.addWidget(self.stats_label)
        main_layout.addLayout(control_layout)

        self.setLayout(main_layout)

    def on_split_changed(self, split):
        self.current_split = split
        self.load_images()
        self.load_current()

    # ==================================================
    # load
    # ==================================================
    def load_current(self):

        if not self.imgs:
            self.stats_label.setText("No images available in current split.")
            return

        name = self.imgs[self.i]

        img_path = os.path.join(self.root_dir, "images", self.current_split, name)
        mask_name = os.path.splitext(name)[0] + ".png"
        mask_path = os.path.join(self.root_dir, "labels", self.current_split, mask_name)

        img = cv2.imread(img_path)
        mask = cv2.imread(mask_path, 0)

        if img is None or mask is None:
            self.stats_label.setText("Failed to load image or mask.")
            return

        unique, counts = np.unique(mask, return_counts=True)
        total_pixels = mask.size

        stats_lines = [
            f"Index: {self.i + 1}/{len(self.imgs)}",
            f"Name: {name}",
            f"Image shape: {img.shape}",
            f"Mask shape: {mask.shape}",
            "Mask values:"
        ]
        for value, count in zip(unique, counts):
            stats_lines.append(
                f"  value={value}: {count} pixels, ratio={count / total_pixels:.6f}"
            )
        invalid_values = [v for v in unique if v not in (0, 1, 2)]
        if invalid_values:
            stats_lines.append(f"Invalid values: {invalid_values}")
        else:
            stats_lines.append("Invalid values: none")

        self.stats_label.setText("\n".join(stats_lines))

        # ==================================================
        # mask visualization
        # ==================================================
        mask_color = np.zeros_like(img)

        # 类别 0：背景
        mask_color[mask == 0] = [0, 0, 0]
        # 类别 1：印刷体文字
        mask_color[mask == 1] = [0, 255, 0]
        # 类别 2：手写体文字
        mask_color[mask == 2] = [0, 0, 255]

        self.show_img(self.img_label, img)
        self.show_img(self.mask_label, mask_color)

    # ==================================================
    # cv2 -> Qt
    # ==================================================
    def show_img(self, view, img):

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        h, w, c = img.shape

        qimg = QImage(
            img.data,
            w,
            h,
            c * w,
            QImage.Format_RGB888
        )

        pix = QPixmap.fromImage(qimg)

        view.setPixmap(pix)

    # ==================================================
    # control
    # ==================================================
    def prev(self):
        self.i = max(0, self.i - 1)
        self.load_current()

    def next(self):
        self.i = min(len(self.imgs) - 1, self.i + 1)
        self.load_current()

    def keyPressEvent(self, event):

        if event.key() == Qt.Key_A:
            self.prev()

        elif event.key() == Qt.Key_D:
            self.next()


# ==================================================
# main
# ==================================================
if __name__ == "__main__":

    dataset_dir = os.path.join(PROJECT_ROOT, "output/both/dataset")

    if not os.path.exists(dataset_dir):
        print(f"Error: Dataset directory not found: {dataset_dir}")
        exit(1)

    app = QApplication([])

    viewer = Viewer(dataset_dir)

    viewer.show()

    app.exec_()