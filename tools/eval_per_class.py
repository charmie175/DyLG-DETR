"""Export COCO-style AP and AP50 for every category.

This evaluator intentionally has no pycocotools dependency. It accepts a COCO
annotation JSON and a COCO detection-result JSON (bbox in [x, y, w, h]) and
implements the standard 101-point interpolated AP at IoU 0.50:0.95.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path


def iou(a, b):
    ax1, ay1, aw, ah = a
    bx1, by1, bw, bh = b
    ax2, ay2 = ax1 + max(0.0, aw), ay1 + max(0.0, ah)
    bx2, by2 = bx1 + max(0.0, bw), by1 + max(0.0, bh)
    ix1, iy1, ix2, iy2 = max(ax1, bx1), max(ay1, by1), min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = max(0.0, aw) * max(0.0, ah) + max(0.0, bw) * max(0.0, bh) - inter
    return inter / union if union > 0 else 0.0


def ap_for_class(gt_by_image, dt_by_image, threshold):
    npos = sum(len(v) for v in gt_by_image.values())
    if npos == 0:
        return math.nan
    detections = []
    for image_id, values in dt_by_image.items():
        detections.extend((float(d.get("score", 0.0)), image_id, d["bbox"]) for d in values)
    detections.sort(key=lambda x: x[0], reverse=True)
    matched = {image_id: [False] * len(boxes) for image_id, boxes in gt_by_image.items()}
    tp, fp = [], []
    for _, image_id, box in detections:
        gt_boxes = gt_by_image.get(image_id, [])
        best, best_iou = -1, threshold
        for idx, gt_box in enumerate(gt_boxes):
            if not matched[image_id][idx]:
                overlap = iou(box, gt_box)
                if overlap >= best_iou:
                    best, best_iou = idx, overlap
        if best >= 0:
            matched[image_id][best] = True
            tp.append(1.0)
            fp.append(0.0)
        else:
            tp.append(0.0)
            fp.append(1.0)
    if not tp:
        return 0.0
    recalls, precisions = [], []
    cumulative_tp = cumulative_fp = 0.0
    for t, f in zip(tp, fp):
        cumulative_tp += t
        cumulative_fp += f
        recalls.append(cumulative_tp / npos)
        precisions.append(cumulative_tp / (cumulative_tp + cumulative_fp))
    # COCO uses max precision for each of 101 recall points.
    result = 0.0
    for recall_point in [x / 100.0 for x in range(101)]:
        candidates = [p for r, p in zip(recalls, precisions) if r >= recall_point]
        result += max(candidates) if candidates else 0.0
    return result / 101.0


def evaluate(annotation_file, detection_file):
    with open(annotation_file, "r", encoding="utf-8") as f:
        annotations = json.load(f)
    with open(detection_file, "r", encoding="utf-8") as f:
        detections = json.load(f)
    categories = sorted(annotations.get("categories", []), key=lambda x: x["id"])
    valid_ids = {c["id"] for c in categories}
    gt = {c["id"]: {} for c in categories}
    for ann in annotations.get("annotations", []):
        if ann.get("iscrowd", 0) or ann.get("category_id") not in valid_ids:
            continue
        gt[ann["category_id"]].setdefault(ann["image_id"], []).append(ann["bbox"])
    dt = {c["id"]: {} for c in categories}
    for det in detections:
        if det.get("category_id") in valid_ids and "bbox" in det:
            dt[det["category_id"]].setdefault(det["image_id"], []).append(det)
    # Match COCOeval's maxDets=100 setting for each image/category.
    for by_image in dt.values():
        for image_id in by_image:
            by_image[image_id].sort(key=lambda x: float(x.get("score", 0.0)), reverse=True)
            del by_image[image_id][100:]
    thresholds = [0.50 + 0.05 * i for i in range(10)]
    rows = []
    for category in categories:
        cid = category["id"]
        aps = [ap_for_class(gt[cid], dt[cid], t) for t in thresholds]
        rows.append({"category_id": cid, "category": category["name"],
                     "AP": sum(aps) / len(aps), "AP50": aps[0], "num_gt": sum(map(len, gt[cid].values()))})
    valid_ap = [r["AP"] for r in rows if not math.isnan(r["AP"])]
    valid_ap50 = [r["AP50"] for r in rows if not math.isnan(r["AP50"])]
    return rows, sum(valid_ap) / len(valid_ap), sum(valid_ap50) / len(valid_ap50)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--anno", required=True, type=Path)
    parser.add_argument("--pred", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--method", default="DyLG-DETR")
    args = parser.parse_args()
    rows, mean_ap, mean_ap50 = evaluate(args.anno, args.pred)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["method", "category_id", "category", "AP", "AP50", "num_gt"])
        writer.writeheader()
        for row in rows:
            writer.writerow({"method": args.method, **row, "AP": f"{row['AP'] * 100:.4f}", "AP50": f"{row['AP50'] * 100:.4f}"})
        writer.writerow({"method": args.method, "category_id": "", "category": "mAP", "AP": f"{mean_ap * 100:.4f}", "AP50": f"{mean_ap50 * 100:.4f}", "num_gt": sum(r["num_gt"] for r in rows)})
    print(f"{args.method}: mAP={mean_ap * 100:.4f}, AP50={mean_ap50 * 100:.4f}")
    for row in rows:
        print(f"{row['category_id']:>2} {row['category']:<24} AP={row['AP'] * 100:7.4f} AP50={row['AP50'] * 100:7.4f} GT={row['num_gt']}")


if __name__ == "__main__":
    main()
