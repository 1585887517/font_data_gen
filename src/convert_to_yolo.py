import os
import cv2
import yaml
import shutil
import argparse
import numpy as np

from PIL import Image
from tqdm import tqdm


def load_classes(labels_txt):
    with open(labels_txt, 'r', encoding='utf-8') as f:
        classes = [line.strip() for line in f.readlines()]

    # 去掉 background
    classes = [c for c in classes if c.lower() != 'background']

    return classes


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def simplify_contour(contour):
    epsilon = 0.002 * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, True)
    return approx


def mask_to_yolo_seg(mask_path, txt_output_path, num_classes):

    mask = np.array(Image.open(mask_path))

    h, w = mask.shape[:2]

    lines = []

    # 跳过 background(0)
    for cls_id in range(1, num_classes):

        binary = (mask == cls_id).astype(np.uint8)

        contours, _ = cv2.findContours(
            binary,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        for contour in contours:

            area = cv2.contourArea(contour)

            # 过滤太小区域
            if area < 10:
                continue

            contour = simplify_contour(contour)

            contour = contour.squeeze(1)

            if len(contour.shape) != 2:
                continue

            if contour.shape[0] < 3:
                continue

            points = []

            for point in contour:
                x, y = point

                x = x / w
                y = y / h

                points.append(f"{x:.6f}")
                points.append(f"{y:.6f}")

            # YOLO类别从0开始
            yolo_cls = cls_id - 1

            line = f"{yolo_cls} " + " ".join(points)

            lines.append(line)

    with open(txt_output_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(lines))


def process_split(src_root, dst_root, split, num_classes):

    img_src_dir = os.path.join(src_root, "images", split)
    mask_src_dir = os.path.join(src_root, "labels", split)

    img_dst_dir = os.path.join(dst_root, "images", split)
    label_dst_dir = os.path.join(dst_root, "labels", split)

    ensure_dir(img_dst_dir)
    ensure_dir(label_dst_dir)

    image_files = []

    for file in os.listdir(img_src_dir):

        if not file.lower().endswith(".jpg"):
            continue

        image_files.append(file)

    for img_name in tqdm(image_files, desc=f"Processing {split}"):

        base_name = os.path.splitext(img_name)[0]

        img_path = os.path.join(img_src_dir, img_name)

        mask_path = os.path.join(mask_src_dir, base_name + ".png")

        if not os.path.exists(mask_path):
            print(f"Mask not found: {mask_path}")
            continue

        # copy image
        shutil.copy(img_path, os.path.join(img_dst_dir, img_name))

        # generate yolo txt
        txt_output_path = os.path.join(
            label_dst_dir,
            base_name + ".txt"
        )

        mask_to_yolo_seg(
            mask_path,
            txt_output_path,
            num_classes
        )


def generate_yaml(dst_root, class_names):

    yaml_data = {
        "path": os.path.abspath(dst_root),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": {
            idx: name
            for idx, name in enumerate(class_names)
        }
    }

    yaml_path = os.path.join(dst_root, "data.yaml")

    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.dump(
            yaml_data,
            f,
            allow_unicode=True,
            sort_keys=False
        )

    print(f"Saved yaml: {yaml_path}")


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--src",
        type=str,
        required=True,
        help="source dataset root"
    )

    parser.add_argument(
        "--dst",
        type=str,
        required=True,
        help="output yolo dataset root"
    )

    args = parser.parse_args()

    src_root = args.src
    dst_root = args.dst

    ensure_dir(dst_root)

    labels_txt = os.path.join(src_root, "labels.txt")

    class_names = load_classes(labels_txt)

    print("Classes:", class_names)

    num_classes = len(class_names) + 1

    for split in ["train", "val", "test"]:

        process_split(
            src_root,
            dst_root,
            split,
            num_classes
        )

    generate_yaml(dst_root, class_names)

    print("Done.")


if __name__ == "__main__":
    main()