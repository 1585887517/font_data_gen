import argparse
import os
import random
import re
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from configs.config import Config
from generators.handwriting_loader import HandwritingLoader
from tools.io_utils import IOUtils
from tools.split_dataset import split_dataset


def import_pypdf():
    try:
        from pypdf import PdfReader
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "Missing dependency: pypdf. Run `uv sync` first, then retry this tool."
        ) from exc
    return PdfReader


def slugify(value):
    stem = Path(value).stem
    stem = re.sub(r"[^0-9A-Za-z\u4e00-\u9fff_-]+", "_", stem)
    return stem.strip("_") or "pdf"


def iter_pdf_page_images(pdf_path, start_page=1, end_page=None):
    PdfReader = import_pypdf()
    reader = PdfReader(str(pdf_path))
    total = len(reader.pages)
    end = total if end_page is None else min(end_page, total)

    for page_index in range(max(1, start_page) - 1, end):
        page = reader.pages[page_index]
        images = list(page.images)
        if not images:
            continue

        image_file = max(
            images,
            key=lambda item: item.image.size[0] * item.image.size[1]
        )
        image = image_file.image.convert("RGB")
        yield page_index + 1, image


def resize_if_needed(image, max_side):
    if not max_side or max(image.size) <= max_side:
        return image

    scale = max_side / max(image.size)
    size = (
        max(1, int(round(image.size[0] * scale))),
        max(1, int(round(image.size[1] * scale))),
    )
    return image.resize(size, Image.Resampling.LANCZOS)


def remove_small_components(binary, min_area, max_area_ratio):
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary,
        connectivity=8
    )
    h, w = binary.shape
    page_area = h * w
    cleaned = np.zeros_like(binary)

    for label in range(1, num_labels):
        x, y, comp_w, comp_h, area = stats[label]
        if area < min_area:
            continue
        if area > page_area * max_area_ratio:
            continue
        if comp_w > w * 0.98 and comp_h > h * 0.05:
            continue
        if comp_h > h * 0.35 and comp_w > w * 0.35:
            continue
        cleaned[labels == label] = 255

    return cleaned


def extract_printed_mask(
    rgb,
    min_contrast=18,
    min_area=4,
    max_area_ratio=0.04,
):
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    h, w = gray.shape
    blur_size = max(31, (min(h, w) // 18) | 1)
    background = cv2.GaussianBlur(gray, (blur_size, blur_size), 0)
    contrast = np.clip(
        background.astype(np.int16) - gray.astype(np.int16),
        0,
        255
    ).astype(np.uint8)

    positive = contrast[contrast > 0]
    if positive.size == 0:
        return np.zeros_like(gray, dtype=np.uint8)

    otsu_threshold, _ = cv2.threshold(
        contrast,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    threshold = max(int(otsu_threshold), int(min_contrast))
    binary = (contrast >= threshold).astype(np.uint8) * 255

    binary = remove_small_components(
        binary,
        min_area=min_area,
        max_area_ratio=max_area_ratio
    )

    return (binary > 0).astype(np.uint8)


def make_cfg(output_root, save_mask_vis=True):
    cfg = Config()
    cfg.OUTPUT_ROOT = str(output_root)
    cfg.OUTPUT_IMG = str(output_root / "images")
    cfg.OUTPUT_MASK = str(output_root / "masks")
    cfg.OUTPUT_MASK_VIS = str(output_root / "mask_vis")
    cfg.OUTPUT_DIR = str(output_root / "dataset")
    cfg.SAVE_MASK_VIS = save_mask_vis
    cfg.OUTPUT_IMAGE_EXT = ".jpg"
    cfg.OUTPUT_MASK_EXT = ".png"
    return cfg


def save_sample(rgb, mask, name, cfg):
    os.makedirs(cfg.OUTPUT_IMG, exist_ok=True)
    os.makedirs(cfg.OUTPUT_MASK, exist_ok=True)
    os.makedirs(cfg.OUTPUT_MASK_VIS, exist_ok=True)
    IOUtils.save(rgb, mask, name, cfg)


def maybe_add_handwriting(rgb, mask, handwriting_loader, overlays, rng):
    for _ in range(overlays):
        source = "iam" if rng.random() < 0.5 else "casia"
        rgb, mask = handwriting_loader.overlay_by_source(
            rgb,
            mask,
            source,
            allow_overlap=False
        )
    return rgb, mask


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert scanned book PDFs into PaddleSeg-style segmentation data."
    )
    parser.add_argument("--pdf", nargs="+", required=True, help="Input scanned PDF file(s).")
    parser.add_argument("--out", required=True, help="Output root directory.")
    parser.add_argument(
        "--mode",
        choices=["printed_only", "both_with_synthetic_handwriting"],
        default="printed_only",
    )
    parser.add_argument("--start-page", type=int, default=1)
    parser.add_argument("--end-page", type=int, default=None)
    parser.add_argument("--max-side", type=int, default=1800)
    parser.add_argument("--min-contrast", type=int, default=18)
    parser.add_argument("--min-component-area", type=int, default=4)
    parser.add_argument("--max-component-area-ratio", type=float, default=0.04)
    parser.add_argument("--handwriting-overlays", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--no-split", action="store_true")
    parser.add_argument("--no-mask-vis", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    rng = random.Random(args.seed)
    np.random.seed(args.seed % (2**32))

    output_root = Path(args.out).resolve()
    cfg = make_cfg(output_root, save_mask_vis=not args.no_mask_vis)

    handwriting_loader = None
    if args.mode == "both_with_synthetic_handwriting":
        handwriting_loader = HandwritingLoader(cfg)

    sample_count = 0
    for pdf in args.pdf:
        pdf_path = Path(pdf).expanduser().resolve()
        pdf_slug = slugify(pdf_path)

        for page_number, image in iter_pdf_page_images(
            pdf_path,
            start_page=args.start_page,
            end_page=args.end_page
        ):
            image = resize_if_needed(image, args.max_side)
            rgb = np.array(image)
            printed = extract_printed_mask(
                rgb,
                min_contrast=args.min_contrast,
                min_area=args.min_component_area,
                max_area_ratio=args.max_component_area_ratio,
            )
            mask = np.zeros(printed.shape, dtype=np.uint8)
            mask[printed > 0] = 1

            if handwriting_loader is not None:
                rgb, mask = maybe_add_handwriting(
                    rgb,
                    mask,
                    handwriting_loader,
                    overlays=args.handwriting_overlays,
                    rng=rng
                )

            name = f"{pdf_slug}_p{page_number:04d}"
            save_sample(rgb, mask, name, cfg)
            sample_count += 1

    if not args.no_split and sample_count > 0:
        split_dataset(
            img_dir=cfg.OUTPUT_IMG,
            mask_dir=cfg.OUTPUT_MASK,
            out_dir=cfg.OUTPUT_DIR,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            seed=args.seed,
            clean=True
        )

    print(f"PDF dataset generated: {sample_count} samples -> {output_root}")
    print(f"Training labels: {cfg.OUTPUT_MASK}")
    if cfg.SAVE_MASK_VIS:
        print(f"Visual masks: {cfg.OUTPUT_MASK_VIS}")
    if not args.no_split:
        print(f"PaddleSeg dataset: {cfg.OUTPUT_DIR}")


if __name__ == "__main__":
    main()
