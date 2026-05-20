import os
import multiprocessing
import platform


# ==================================================
# 🚀 project root
# ==================================================
PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)


class Config:

    # ==================================================
    # 🚀 system info
    # ==================================================
    CPU_COUNT = multiprocessing.cpu_count()

    SYSTEM = platform.system()

    # ==================================================
    # 🚀 worker auto config
    # ==================================================
    # 数据生成以 OpenCV/Pillow/磁盘 IO 为主，适合用接近 CPU 核数的 worker
    NUM_WORKERS = int(os.getenv(
        "NUM_WORKERS",
        str(min(12, max(1, CPU_COUNT)))
    ))

    # ==================================================
    # 🚀 gpu config（后续可扩展）
    # ==================================================
    USE_GPU = False

    # ==================================================
    # 🚀 debug mode
    # ==================================================
    DEBUG = True

    # ==================================================
    # 🚀 dataset version
    # ==================================================
    DATASET_NAME = "dataset"

    # ==================================================
    # 🚀 output
    # ==================================================
        # 数据集模式：printed_only, handwriting_only, both
    DATASET_MODE = "both"
    OUTPUT_ROOT = os.path.join(PROJECT_ROOT, f"output/{DATASET_MODE}")

    OUTPUT_IMG = os.path.join(OUTPUT_ROOT, "images")
    OUTPUT_MASK = os.path.join(OUTPUT_ROOT, "masks")
    OUTPUT_MASK_VIS = os.path.join(OUTPUT_ROOT, "mask_vis")

    OUTPUT_DIR = os.path.join(
        OUTPUT_ROOT,
        DATASET_NAME
    )

    # ==================================================
    # 🚀 handwriting data
    # ==================================================
    HANDWRITING_ROOT = os.path.join(PROJECT_ROOT, "handwriting")

    IAM_DIR = os.path.join(HANDWRITING_ROOT, "iam")
    CASIA_DIR = os.path.join(HANDWRITING_ROOT, "casia")

    IAM_RGBA_DIR = os.path.join(HANDWRITING_ROOT, "iam_rgba")
    CASIA_RGBA_DIR = os.path.join(HANDWRITING_ROOT, "casia_rgba")
    REBUILD_HANDWRITING_RGBA = os.getenv("REBUILD_HANDWRITING_RGBA", "0") == "1"

    # ==================================================
    # 🚀 fonts
    # ==================================================
    FONT_ROOT = os.path.join(PROJECT_ROOT, "fonts")

    FONT_PATH = os.path.join(
        FONT_ROOT,
        "SourceHanSerifCN-Regular.otf"
    )

    # ==================================================
    # 🚀 open text corpus
    # ==================================================
    TEXT_ROOT = os.path.join(PROJECT_ROOT, "data", "text")
    TEXT_FILE_EXTENSIONS = (".txt",)

    # ==================================================
    # 🚀 image config
    # ==================================================
    # A4 比例参考，纵向纸张更符合文档场景
    WIDTH = 960
    HEIGHT = 1358

    CHANNELS = 3

    OUTPUT_IMAGE_EXT = ".jpg"
    OUTPUT_MASK_EXT = ".png"
    IMAGE_JPEG_QUALITY = int(os.getenv("IMAGE_JPEG_QUALITY", "95"))
    SAVE_MASK_VIS = os.getenv("SAVE_MASK_VIS", "1") == "1"

    # ==================================================
    # 🚀 dataset config
    # ==================================================
    NUM_SAMPLES = 100

    TRAIN_RATIO = 0.7
    VAL_RATIO = 0.2
    TEST_RATIO = 0.1

    RANDOM_SEED = 42

    # 重新生成时清理旧图片/标签，避免 PaddleSeg 训练到上一轮残留样本
    CLEAN_OUTPUT = True

    # 前景太稀会让 printed/handwriting 的 IoU 很不稳定
    HANDWRITING_OVERLAYS_PER_IMAGE = (2, 4)
    MIN_FOREGROUND_RATIO = 0.018
    MIN_PRINTED_RATIO = 0.006
    MIN_HANDWRITING_RATIO = 0.004
    MAX_REBUILD_ATTEMPTS = 8
    PRINTED_LABEL_ALPHA_THRESHOLD = 32
    PRINTED_OCCUPIED_ALPHA_THRESHOLD = 20

    # 手写体白底扣图与去噪。按背景亮度提取暗笔画，避免灰纸底变成整块 alpha。
    HANDWRITING_MIN_STROKE_CONTRAST = 28
    HANDWRITING_LABEL_DILATE_ITERATIONS = 0
    HANDWRITING_MIN_COMPONENT_AREA = 10
    HANDWRITING_MIN_COMPONENT_SIDE = 2
    HANDWRITING_FEATHER_BLUR = 3

    # 版式不要过于模板化，否则模型会记住表格位置而不是文字外观
    HEADLINE_LAYOUT_PROB = 0.18
    FORM_LAYOUT_PROB = 0.40
    RECEIPT_LAYOUT_PROB = 0.18
    FREE_LAYOUT_PROB = 0.24

    # 合成手机扫描纸质文档的真实性增强
    ENABLE_PAPER_TEXTURE = True
    ENABLE_FORM_LAYOUT = True
    ENABLE_PHONE_EFFECTS = True

    # 手写体靠近印刷体/线条的概率，用于解决邻近识别问题
    HANDWRITING_PROXIMITY_PROB = 0.5

    # ==================================================
    # 🚀 augmentation
    # ==================================================
    ENABLE_ROTATE = True
    ENABLE_PERSPECTIVE = True
    ENABLE_SCAN_NOISE = True

    # ==================================================
    # 🚀 logging
    # ==================================================
    LOG_INTERVAL = 100

    # ==================================================
    # 🚀 auto create dirs
    # ==================================================
    def init_dirs(self):

        dirs = [
            getattr(self, 'OUTPUT_ROOT', type(self).OUTPUT_ROOT),
            getattr(self, 'OUTPUT_IMG', type(self).OUTPUT_IMG),
            getattr(self, 'OUTPUT_MASK', type(self).OUTPUT_MASK),
            getattr(self, 'OUTPUT_MASK_VIS', type(self).OUTPUT_MASK_VIS),
            getattr(self, 'OUTPUT_DIR', type(self).OUTPUT_DIR),

            self.HANDWRITING_ROOT,
            self.IAM_DIR,
            self.CASIA_DIR,

            self.IAM_RGBA_DIR,
            self.CASIA_RGBA_DIR,

            self.FONT_ROOT,
            self.TEXT_ROOT,
        ]

        for d in dirs:
            os.makedirs(d, exist_ok=True)

    @classmethod
    def get_dataset_classes(cls):
        mode = cls.DATASET_MODE
        if mode == "printed_only":
            return ["background", "printed_text"]
        elif mode == "handwriting_only":
            return ["background", "handwriting"]
        else:  # both, both_overlap, or rotate
            return ["background", "printed_text", "handwriting"]

    def get_dataset_classes_instance(self):
        mode = getattr(self, 'DATASET_MODE', type(self).DATASET_MODE)
        if mode == "printed_only":
            return ["background", "printed_text"]
        elif mode == "handwriting_only":
            return ["background", "handwriting"]
        else:  # both, both_overlap, or rotate
            return ["background", "printed_text", "handwriting"]

    # ==================================================
    # 🚀 print config
    # ==================================================
    @classmethod
    def print_config(cls):

        print("\n========== CONFIG ==========")

        attrs = [
            "SYSTEM",
            "CPU_COUNT",
            "NUM_WORKERS",
            "USE_GPU",
            "DEBUG",
            "DATASET_NAME",
            "NUM_SAMPLES",
            "RANDOM_SEED",
            "CLEAN_OUTPUT",
            "DATASET_MODE",
            "ENABLE_PAPER_TEXTURE",
            "ENABLE_FORM_LAYOUT",
            "ENABLE_PHONE_EFFECTS",
            "WIDTH",
            "HEIGHT",
        ]

        for a in attrs:
            print(f"{a}: {getattr(cls, a)}")

        print("============================\n")
