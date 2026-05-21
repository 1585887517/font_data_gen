import argparse
import os
import random
import re
import sys
import tempfile
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


def import_text_detection():
    try:
        from paddleocr import TextDetection
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "Missing dependency: paddleocr. Run `uv sync` first, then retry this tool."
        ) from exc
    return TextDetection


def extract_dt_polys(result):
    data = getattr(result, "json", None)

    if data is None:
        if isinstance(result, dict):
            data = result
        else:
            raise RuntimeError(f"Cannot extract json from OCR result: {type(result)}")

    if "res" in data:
        data = data["res"]

    polys = data.get("dt_polys", [])
    clean_polys = []

    for poly in polys:
        pts = poly.tolist() if hasattr(poly, "tolist") else poly
        if pts and len(pts) >= 3:
            clean_polys.append([[float(x), float(y)] for x, y in pts])

    return clean_polys


def slugify(value):
    stem = Path(value).stem
    stem = re.sub(r"[^0-9A-Za-z\u4e00-\u9fff_-]+", "_", stem)
    return stem.strip("_") or "pdf"


def clamp_poly(poly, width, height):
    out = []
    for x, y in poly:
        out.append([
            float(np.clip(x, 0, width - 1)),
            float(np.clip(y, 0, height - 1)),
        ])
    return out


def make_ocr_model(model_name, device):
    TextDetection = import_text_detection()
    model_kwargs = {"model_name": model_name}
    if device:
        model_kwargs["device"] = device
    return TextDetection(**model_kwargs)


def detect_text_polys(rgb, model, batch_size, tmp_dir, image_name):
    tmp_path = Path(tmp_dir) / f"{image_name}.jpg"
    cv2.imwrite(
        str(tmp_path),
        cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
        [int(cv2.IMWRITE_JPEG_QUALITY), 95]
    )

    output = model.predict(str(tmp_path), batch_size=batch_size)
    polys = []
    for res in output:
        polys.extend(extract_dt_polys(res))

    h, w = rgb.shape[:2]
    return [clamp_poly(poly, w, h) for poly in polys]


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


def polygon_mask(shape, polys, expand_px=3):
    h, w = shape
    mask = np.zeros((h, w), dtype=np.uint8)

    for poly in polys:
        pts = np.array(poly, dtype=np.float32)
        if pts.shape[0] < 3:
            continue
        pts = np.round(pts).astype(np.int32)
        cv2.fillPoly(mask, [pts], 255)

    if expand_px > 0:
        kernel_size = max(1, int(expand_px) * 2 + 1)
        kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (kernel_size, kernel_size)
        )
        mask = cv2.dilate(mask, kernel, iterations=1)

    return mask


def odd_at_least(value, minimum=15):
    value = max(minimum, int(value))
    return value if value % 2 == 1 else value + 1


def extract_single_polarity(gray_roi, roi_text_mask, min_contrast, polarity):
    h, w = gray_roi.shape
    morph_size = odd_at_least(min(h, w) // 2, minimum=15)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (morph_size, morph_size))
    
    if polarity == "dark":
        bg_base = cv2.dilate(gray_roi, kernel)
    else:
        bg_base = cv2.erode(gray_roi, kernel)

    blur_size = odd_at_least(min(h, w) // 4, minimum=7)
    background = cv2.GaussianBlur(bg_base, (blur_size, blur_size), 0)

    if polarity == "light":
        contrast = np.clip(gray_roi.astype(np.int16) - background.astype(np.int16), 0, 255).astype(np.uint8)
    else:
        contrast = np.clip(background.astype(np.int16) - gray_roi.astype(np.int16), 0, 255).astype(np.uint8)

    values = contrast[roi_text_mask > 0]
    values = values[values > 0]
    if values.size == 0:
        return np.zeros_like(gray_roi, dtype=np.uint8)

    otsu_threshold, _ = cv2.threshold(values.reshape(-1, 1), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    threshold = max(int(otsu_threshold), int(min_contrast))
    strokes = ((contrast >= threshold) & (roi_text_mask > 0)).astype(np.uint8) * 255
    return strokes


def extract_roi_strokes(gray_roi, roi_text_mask, min_contrast):
    """Extract stroke pixels using dual-polarity estimation and fallback heuristics."""
    if not np.any(roi_text_mask):
        return np.zeros_like(gray_roi, dtype=np.uint8)
        
    strokes_dark = extract_single_polarity(gray_roi, roi_text_mask, min_contrast, "dark")
    strokes_light = extract_single_polarity(gray_roi, roi_text_mask, min_contrast, "light")
    
    mask_area = max(1, np.count_nonzero(roi_text_mask))
    fill_dark = np.count_nonzero(strokes_dark) / mask_area
    fill_light = np.count_nonzero(strokes_light) / mask_area
    
    # If one polarity yields a nearly solid block (>60% fill), it's likely extracting a background band.
    # We should choose the opposite polarity.
    if fill_dark > 0.60 and fill_light < 0.60:
        return strokes_light
    if fill_light > 0.60 and fill_dark < 0.60:
        return strokes_dark
        
    # Fallback: check the boundary pixels of the bounding box
    top = gray_roi[0, :]
    bottom = gray_roi[-1, :]
    left = gray_roi[:, 0]
    right = gray_roi[:, -1]
    boundary_pixels = np.concatenate([top, bottom, left, right])
    if boundary_pixels.size == 0:
        return strokes_dark
        
    boundary_median = np.median(boundary_pixels)
    if boundary_median > 127:  # Bright local background implies dark text
        return strokes_dark
    else:                      # Dark local background implies light text
        return strokes_light


def _threshold_single_direction(contrast, min_contrast):
    """Apply Otsu + min_contrast threshold to a single-direction contrast map.

    Returns a binary uint8 image (0 / 255) or *None* when no signal is found.
    """
    positive = contrast[contrast > 0]
    if positive.size == 0:
        return None

    otsu_threshold, _ = cv2.threshold(
        contrast,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    threshold = max(int(otsu_threshold), int(min_contrast))
    return (contrast >= threshold).astype(np.uint8) * 255


def extract_printed_mask(
    rgb,
    min_contrast=18,
    min_area=4,
    max_area_ratio=0.05,
):
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    h, w = gray.shape
    blur_size = max(31, (min(h, w) // 18) | 1)
    background = cv2.GaussianBlur(gray, (blur_size, blur_size), 0)

    diff = background.astype(np.int16) - gray.astype(np.int16)

    # Dark-on-light direction (original)
    contrast_dark = np.clip(diff, 0, 255).astype(np.uint8)
    # Light-on-dark direction (new)
    contrast_light = np.clip(-diff, 0, 255).astype(np.uint8)

    bin_dark = _threshold_single_direction(contrast_dark, min_contrast)
    bin_light = _threshold_single_direction(contrast_light, min_contrast)

    if bin_dark is None and bin_light is None:
        return np.zeros((h, w), dtype=np.uint8)

    binary = np.zeros((h, w), dtype=np.uint8)
    if bin_dark is not None:
        binary = np.maximum(binary, bin_dark)
    if bin_light is not None:
        binary = np.maximum(binary, bin_light)

    binary = remove_small_components(
        binary,
        min_area=min_area,
        max_area_ratio=max_area_ratio
    )

    return (binary > 0).astype(np.uint8)


def extract_printed_mask_ocr_guided(
    rgb,
    polys,
    min_contrast=14,
    min_area=3,
    max_area_ratio=0.05,
    expand_px=4,
):
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    h, w = gray.shape
    text_region = polygon_mask((h, w), polys, expand_px=expand_px)
    if not np.any(text_region):
        return np.zeros((h, w), dtype=np.uint8)

    raw = np.zeros((h, w), dtype=np.uint8)

    for poly in polys:
        pts = np.array(poly, dtype=np.float32)
        if pts.shape[0] < 3:
            continue

        x, y, box_w, box_h = cv2.boundingRect(np.round(pts).astype(np.int32))
        # Use a generous padding so the border region is well represented
        # for polarity detection and background estimation.
        pad = max(4, int(expand_px) + 2)
        x0 = max(0, x - pad)
        y0 = max(0, y - pad)
        x1 = min(w, x + box_w + pad)
        y1 = min(h, y + box_h + pad)
        if x1 <= x0 or y1 <= y0:
            continue

        roi_mask = np.zeros((y1 - y0, x1 - x0), dtype=np.uint8)
        local_pts = np.round(pts - np.array([x0, y0], dtype=np.float32)).astype(np.int32)
        cv2.fillPoly(roi_mask, [local_pts], 255)
        if expand_px > 0:
            kernel_size = max(1, int(expand_px) * 2 + 1)
            kernel = cv2.getStructuringElement(
                cv2.MORPH_RECT,
                (kernel_size, kernel_size)
            )
            roi_mask = cv2.dilate(roi_mask, kernel, iterations=1)

        gray_roi = gray[y0:y1, x0:x1]

        # Extract strokes using the smart dual-polarity logic
        strokes = extract_roi_strokes(gray_roi, roi_mask, min_contrast)
        raw[y0:y1, x0:x1] = np.maximum(raw[y0:y1, x0:x1], strokes)

    raw = remove_small_components(
        raw,
        min_area=min_area,
        max_area_ratio=max_area_ratio
    )
    raw[text_region == 0] = 0

    return (raw > 0).astype(np.uint8)


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
    parser.add_argument(
        "--mask-method",
        choices=["ocr_guided", "threshold"],
        default="ocr_guided",
        help="ocr_guided confines printed masks to PaddleOCR text detection regions.",
    )
    parser.add_argument("--ocr-model-name", default="PP-OCRv5_server_det")
    parser.add_argument("--ocr-device", default=None, help='For example "cpu", "gpu", "gpu:0".')
    parser.add_argument("--ocr-batch-size", type=int, default=1)
    parser.add_argument("--ocr-expand-px", type=int, default=4)
    parser.add_argument("--min-contrast", type=int, default=18)
    parser.add_argument("--min-component-area", type=int, default=3)
    parser.add_argument("--max-component-area-ratio", type=float, default=0.05)
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

    ocr_model = None
    if args.mask_method == "ocr_guided":
        ocr_model = make_ocr_model(args.ocr_model_name, args.ocr_device)

    handwriting_loader = None
    if args.mode == "both_with_synthetic_handwriting":
        handwriting_loader = HandwritingLoader(cfg)

    sample_count = 0
    with tempfile.TemporaryDirectory(prefix="pdf_ocr_") as tmp_dir:
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
                name = f"{pdf_slug}_p{page_number:04d}"

                if args.mask_method == "ocr_guided":
                    polys = detect_text_polys(
                        rgb,
                        ocr_model,
                        batch_size=args.ocr_batch_size,
                        tmp_dir=tmp_dir,
                        image_name=name
                    )
                    printed = extract_printed_mask_ocr_guided(
                        rgb,
                        polys,
                        min_contrast=args.min_contrast,
                        min_area=args.min_component_area,
                        max_area_ratio=args.max_component_area_ratio,
                        expand_px=args.ocr_expand_px,
                    )
                else:
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
