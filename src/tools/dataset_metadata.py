import os
import json

from tools.logger import Logger


# ==================================================
# 🚀 create labels.txt
# ==================================================
def create_labels_txt(out_dir):

    labels_path = os.path.join(
        out_dir,
        "labels.txt"
    )

    labels = [
        "background",
        "printed_text",
        "handwriting"
    ]

    with open(labels_path, "w") as f:
        for label in labels:
            f.write(label + "\n")

    Logger.info(f"labels.txt created: {labels_path}")


# ==================================================
# 🚀 create dataset_info.json
# ==================================================
def create_dataset_info(cfg):

    info = {
        "dataset_name": cfg.DATASET_NAME,

        "num_classes": 3,

        "classes": [
            "background",
            "printed_text",
            "handwriting"
        ],

        "image_size": {
            "width": cfg.WIDTH,
            "height": cfg.HEIGHT
        },

        "num_samples": cfg.NUM_SAMPLES,

        "train_ratio": cfg.TRAIN_RATIO,
        "val_ratio": cfg.VAL_RATIO,
        "test_ratio": cfg.TEST_RATIO,
    }

    save_path = os.path.join(
        cfg.OUTPUT_DIR,
        "dataset_info.json"
    )

    with open(save_path, "w") as f:
        json.dump(
            info,
            f,
            indent=2,
            ensure_ascii=False
        )

    Logger.info(f"dataset_info.json created: {save_path}")


# ==================================================
# 🚀 build all metadata
# ==================================================
def build_dataset_metadata(cfg):

    create_labels_txt(cfg.OUTPUT_DIR)

    create_dataset_info(cfg)