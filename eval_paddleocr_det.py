# eval_paddleocr_det.py
import argparse
import csv
import json
from pathlib import Path

from paddleocr import TextDetection
from shapely.geometry import Polygon
from tqdm import tqdm


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def polygon_iou(poly1, poly2):
    """poly: [[x,y], ...]. Return IoU of two polygons."""
    p1 = Polygon(poly1)
    p2 = Polygon(poly2)

    if not p1.is_valid:
        p1 = p1.buffer(0)
    if not p2.is_valid:
        p2 = p2.buffer(0)

    if p1.is_empty or p2.is_empty or p1.area <= 0 or p2.area <= 0:
        return 0.0

    inter = p1.intersection(p2).area
    union = p1.union(p2).area
    return float(inter / union) if union > 0 else 0.0


def load_paddleocr_det_labels(label_file, image_dir):
    """
    PaddleOCR det label format:
    img_path<TAB>[{"transcription":"text","points":[[x,y],...]}]
    """
    labels = {}
    image_dir = Path(image_dir)

    with open(label_file, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue

            try:
                rel_path, anno_json = line.split("\t", 1)
            except ValueError:
                raise ValueError(f"Line {line_no} is not valid: expected image_path<TAB>json")

            annos = json.loads(anno_json)
            gt_polys = []

            for obj in annos:
                # 忽略 don't care 区域；如果你的数据不用 ###，可以删掉这段
                if obj.get("transcription") == "###":
                    continue
                pts = obj.get("points")
                if pts and len(pts) >= 3:
                    gt_polys.append([[float(x), float(y)] for x, y in pts])

            img_path = image_dir / rel_path
            labels[str(img_path.resolve())] = gt_polys

    return labels


def extract_dt_polys(result):
    """
    PaddleOCR TextDetection result usually exposes result.json:
    {"res": {"dt_polys": ..., "dt_scores": ...}}
    This function is defensive for minor version differences.
    """
    data = getattr(result, "json", None)

    if data is None:
        # 有些版本 Result 对象行为像 dict
        if isinstance(result, dict):
            data = result
        else:
            raise RuntimeError(f"Cannot extract json from result: {type(result)}")

    if "res" in data:
        data = data["res"]

    polys = data.get("dt_polys", [])
    clean_polys = []

    for poly in polys:
        # poly may be numpy array
        pts = poly.tolist() if hasattr(poly, "tolist") else poly
        if pts and len(pts) >= 3:
            clean_polys.append([[float(x), float(y)] for x, y in pts])

    return clean_polys


def greedy_match(gt_polys, pred_polys, iou_thresh):
    """
    One-to-one greedy matching by IoU.
    Returns: tp, fp, fn, matched_ious
    """
    candidates = []

    for gi, gt in enumerate(gt_polys):
        for pi, pred in enumerate(pred_polys):
            iou = polygon_iou(gt, pred)
            if iou >= iou_thresh:
                candidates.append((iou, gi, pi))

    candidates.sort(reverse=True, key=lambda x: x[0])

    matched_gt = set()
    matched_pred = set()
    matched_ious = []

    for iou, gi, pi in candidates:
        if gi in matched_gt or pi in matched_pred:
            continue
        matched_gt.add(gi)
        matched_pred.add(pi)
        matched_ious.append(iou)

    tp = len(matched_ious)
    fp = len(pred_polys) - tp
    fn = len(gt_polys) - tp

    return tp, fp, fn, matched_ious


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_dir", required=True, help="验证集图片目录")
    parser.add_argument("--label_file", required=True, help="PaddleOCR det 格式标注文件")
    parser.add_argument("--model_name", default="PP-OCRv5_server_det")
    parser.add_argument("--device", default=None, help='例如 "cpu", "gpu", "gpu:0"')
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--iou_thresh", type=float, default=0.5)
    parser.add_argument("--output_csv", default="det_eval_detail.csv")
    parser.add_argument("--summary_json", default="det_eval_summary.json")
    args = parser.parse_args()

    labels = load_paddleocr_det_labels(args.label_file, args.image_dir)

    model_kwargs = {"model_name": args.model_name}
    if args.device:
        model_kwargs["device"] = args.device

    model = TextDetection(**model_kwargs)

    total_tp = total_fp = total_fn = 0
    all_matched_ious = []
    rows = []

    for img_path, gt_polys in tqdm(labels.items(), desc="Evaluating"):
        if Path(img_path).suffix.lower() not in IMG_EXTS:
            continue

        output = model.predict(img_path, batch_size=args.batch_size)
        pred_polys = []

        for res in output:
            pred_polys.extend(extract_dt_polys(res))

        tp, fp, fn, matched_ious = greedy_match(
            gt_polys=gt_polys,
            pred_polys=pred_polys,
            iou_thresh=args.iou_thresh,
        )

        total_tp += tp
        total_fp += fp
        total_fn += fn
        all_matched_ious.extend(matched_ious)

        precision = tp / (tp + fp) if tp + fp > 0 else 0.0
        recall = tp / (tp + fn) if tp + fn > 0 else 0.0
        mean_iou = sum(matched_ious) / len(matched_ious) if matched_ious else 0.0

        rows.append({
            "image": img_path,
            "gt_count": len(gt_polys),
            "pred_count": len(pred_polys),
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": precision,
            "recall": recall,
            "mean_matched_iou": mean_iou,
        })

    dataset_precision = total_tp / (total_tp + total_fp) if total_tp + total_fp > 0 else 0.0
    dataset_recall = total_tp / (total_tp + total_fn) if total_tp + total_fn > 0 else 0.0
    dataset_mean_iou = (
        sum(all_matched_ious) / len(all_matched_ious)
        if all_matched_ious else 0.0
    )

    summary = {
        "model_name": args.model_name,
        "iou_thresh": args.iou_thresh,
        "tp": total_tp,
        "fp": total_fp,
        "fn": total_fn,
        "precision": dataset_precision,
        "recall": dataset_recall,
        "mean_matched_iou": dataset_mean_iou,
        "num_images": len(rows),
    }

    with open(args.output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [])
        if rows:
            writer.writeheader()
            writer.writerows(rows)

    with open(args.summary_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()