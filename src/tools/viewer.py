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
    QVBoxLayout
)

from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt


class Viewer(QWidget):

    def __init__(self):

        super().__init__()

        # ==================================================
        # 🚀 自动定位项目根目录
        # ==================================================
        base_dir = os.path.abspath(
            os.path.join(
                os.path.dirname(__file__),
                "../../"
            )
        )

        self.img_dir = os.path.join(base_dir, "output/v3/images")
        self.mask_dir = os.path.join(base_dir, "output/v3/masks")

        def extract_num(x):
            nums = re.findall(r"\d+", x)
            return int(nums[-1]) if nums else -1

        self.imgs = sorted(
            os.listdir(self.img_dir),
            key=extract_num
        )

        self.i = 0

        self.init_ui()
        self.load_current()

    # ==================================================
    # UI
    # ==================================================
    def init_ui(self):

        self.setWindowTitle("Dataset Viewer")
        self.resize(1400, 800)

        self.img_label = QLabel()
        self.mask_label = QLabel()
        self.stats_label = QLabel()

        self.img_label.setAlignment(Qt.AlignCenter)
        self.mask_label.setAlignment(Qt.AlignCenter)
        self.stats_label.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        self.stats_label.setWordWrap(True)

        self.prev_btn = QPushButton("Previous")
        self.next_btn = QPushButton("Next")

        self.prev_btn.clicked.connect(self.prev)
        self.next_btn.clicked.connect(self.next)

        top_layout = QHBoxLayout()
        top_layout.addWidget(self.img_label)
        top_layout.addWidget(self.mask_label)

        btn_layout = QHBoxLayout()
        btn_layout.addStretch()
        btn_layout.addWidget(self.prev_btn)
        btn_layout.addWidget(self.next_btn)
        btn_layout.addStretch()

        main_layout = QVBoxLayout()
        main_layout.addLayout(top_layout)
        main_layout.addWidget(self.stats_label)
        main_layout.addLayout(btn_layout)

        self.setLayout(main_layout)

    # ==================================================
    # load
    # ==================================================
    def load_current(self):

        name = self.imgs[self.i]

        img_path = os.path.join(self.img_dir, name)
        # Convert image extension to .png for mask lookup
        mask_name = os.path.splitext(name)[0] + ".png"
        mask_path = os.path.join(self.mask_dir, mask_name)

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
    def show_img(self, label, img):

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

        pix = pix.scaled(
            650,
            700,
            Qt.KeepAspectRatio
        )

        label.setPixmap(pix)

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

    app = QApplication([])

    viewer = Viewer()

    viewer.show()

    app.exec_()