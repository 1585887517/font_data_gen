#!/usr/bin/env python3
import argparse
import os
import shutil
from pathlib import Path


ALLOWED_IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.bmp'}


def copy_or_link(src: Path, dst: Path):
    if dst.exists():
        dst.unlink()

    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def path_is_image(path: Path) -> bool:
    return path.suffix.lower() in ALLOWED_IMAGE_EXTS


def collect_split_files(root: Path, split: str):
    images_dir = root / 'images' / split
    labels_dir = root / 'labels' / split
    if images_dir.is_dir() and labels_dir.is_dir():
        images = sorted([p for p in images_dir.iterdir() if p.is_file() and path_is_image(p)])
        return [(img, labels_dir / f"{img.stem}.png") for img in images]

    txt_path = root / f'{split}.txt'
    if txt_path.is_file():
        pairs = []
        with txt_path.open('r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                image_path, label_path = line.split(maxsplit=1)
                pairs.append((root / image_path, root / label_path))
        return pairs

    raise FileNotFoundError(
        f"Cannot find split data for '{split}'. Expected either "
        f"'{root}/images/{split}' + '{root}/labels/{split}', or '{root}/{split}.txt'."
    )


def validate_mask(mask_path: Path, max_classes: int):
    import cv2
    mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
    if mask is None:
        raise ValueError(f"Unable to read mask: {mask_path}")

    if mask.ndim == 3:
        if mask.shape[2] == 4:
            mask = cv2.cvtColor(mask, cv2.COLOR_BGRA2GRAY)
        else:
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

    values = sorted(set(int(v) for v in mask.reshape(-1)))
    if values and values[-1] >= max_classes:
        raise ValueError(
            f"Mask {mask_path} contains class id {values[-1]} >= nc={max_classes}. "
            f"Valid values must be in [0, {max_classes - 1}]"
        )
    return values


def build_yolo_dataset(source: Path, destination: Path, nc: int, names: list[str]):
    destination.mkdir(parents=True, exist_ok=True)
    yaml_path = destination / 'dataset.yaml'
    available_splits = []

    for split in ['train', 'val', 'test']:
        has_split = False
        try:
            pairs = collect_split_files(source, split)
            if pairs:
                has_split = True
        except FileNotFoundError:
            pairs = []

        if not has_split:
            continue

        available_splits.append((split, pairs))

    if not available_splits:
        raise ValueError('No valid train/val/test split found in source dataset.')

    for split, pairs in available_splits:
        split_images = destination / 'images' / split
        split_labels = destination / 'labels' / split
        split_images.mkdir(parents=True, exist_ok=True)
        split_labels.mkdir(parents=True, exist_ok=True)

        for img_src, mask_src in pairs:
            if not img_src.exists():
                raise FileNotFoundError(f"Image file not found: {img_src}")
            if not mask_src.exists():
                raise FileNotFoundError(f"Mask file not found: {mask_src}")

            validate_mask(mask_src, nc)

            dst_img = split_images / img_src.name
            dst_mask = split_labels / f"{img_src.stem}.png"
            copy_or_link(img_src, dst_img)
            copy_or_link(mask_src, dst_mask)

        print(f"[{split}] copied {len(pairs)} samples")

    yaml_content = [f"path: ."]
    for split, _ in available_splits:
        yaml_content.append(f"{split}: images/{split}")
    yaml_content.extend([f"nc: {nc}", "names:"])
    for name in names:
        yaml_content.append(f"  - {name}")

    yaml_path.write_text("\n".join(yaml_content) + "\n", encoding='utf-8')
    print(f"Generated YOLO segmentation dataset: {destination}")
    print(f"Generated dataset YAML: {yaml_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Convert an existing generated dataset to YOLO segmentation format.'
    )
    parser.add_argument(
        '--source', '-s',
        default='output/v3/dataset',
        help='Source dataset root containing images/ and labels/ or train/val/test split files.'
    )
    parser.add_argument(
        '--dest', '-d',
        default='output/v3/yolo_seg_dataset',
        help='Destination root for the YOLO segmentation dataset.'
    )
    parser.add_argument(
        '--nc',
        type=int,
        default=3,
        help='Number of segmentation classes in the dataset.'
    )
    parser.add_argument(
        '--names',
        nargs='+',
        default=['background', 'printed_text', 'handwriting'],
        help='Names of the segmentation classes.'
    )
    args = parser.parse_args()

    if args.nc != len(args.names):
        raise SystemExit('Error: --nc must match the number of --names entries.')

    source = Path(args.source)
    destination = Path(args.dest)

    if not source.exists():
        raise SystemExit(f"Source dataset not found: {source}")

    build_yolo_dataset(source, destination, args.nc, args.names)


if __name__ == '__main__':
    main()
