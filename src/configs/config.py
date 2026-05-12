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
    OUTPUT_ROOT = os.path.join(PROJECT_ROOT, "output/v3")

    OUTPUT_IMG = os.path.join(OUTPUT_ROOT, "images")
    OUTPUT_MASK = os.path.join(OUTPUT_ROOT, "masks")

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

    # ==================================================
    # 🚀 dataset config
    # ==================================================
    NUM_SAMPLES = int(os.getenv("NUM_SAMPLES", "60"))

    TRAIN_RATIO = 0.7
    VAL_RATIO = 0.2
    TEST_RATIO = 0.1

    RANDOM_SEED = 42

    # 重新生成时清理旧图片/标签，避免 PaddleSeg 训练到上一轮残留样本
    CLEAN_OUTPUT = True

    # 前景太稀会让 printed/handwriting 的 IoU 很不稳定
    HANDWRITING_OVERLAYS_PER_IMAGE = (1, 3)
    MIN_FOREGROUND_RATIO = 0.08
    MIN_HANDWRITING_RATIO = 0.02

    # 版式不要过于模板化，否则模型会记住表格位置而不是文字外观
    FORM_LAYOUT_PROB = 0.45
    RECEIPT_LAYOUT_PROB = 0.20
    FREE_LAYOUT_PROB = 0.35

    # 合成手机扫描纸质文档的真实性增强
    ENABLE_PAPER_TEXTURE = True
    ENABLE_FORM_LAYOUT = True
    ENABLE_PHONE_EFFECTS = True

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
    @classmethod
    def init_dirs(cls):

        dirs = [
            cls.OUTPUT_ROOT,
            cls.OUTPUT_IMG,
            cls.OUTPUT_MASK,
            cls.OUTPUT_DIR,

            cls.HANDWRITING_ROOT,
            cls.IAM_DIR,
            cls.CASIA_DIR,

            cls.IAM_RGBA_DIR,
            cls.CASIA_RGBA_DIR,

            cls.FONT_ROOT,
            cls.TEXT_ROOT,
        ]

        for d in dirs:
            os.makedirs(d, exist_ok=True)

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
            "ENABLE_PAPER_TEXTURE",
            "ENABLE_FORM_LAYOUT",
            "ENABLE_PHONE_EFFECTS",
            "WIDTH",
            "HEIGHT",
        ]

        for a in attrs:
            print(f"{a}: {getattr(cls, a)}")

        print("============================\n")
