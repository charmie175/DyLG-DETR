import os
import json
import argparse
from PIL import Image
from collections import Counter

CLASSES = [
    'plane',
    'baseball-diamond',
    'bridge',
    'ground-track-field',
    'small-vehicle',
    'large-vehicle',
    'ship',
    'tennis-court',
    'basketball-court',
    'storage-tank',
    'soccer-ball-field',
    'roundabout',
    'harbor',
    'swimming-pool',
    'helicopter',
]

CAT2ID = {name: i + 1 for i, name in enumerate(CLASSES)}

def convert(img_dir, label_dir, out_json):
    images = []
    annotations = []

    class_count = Counter()

    image_id = 1
    ann_id = 1

    files = sorted([
        x for x in os.listdir(img_dir)
        if x.lower().endswith(('.png', '.jpg', '.jpeg'))
    ])

    for idx, name in enumerate(files):
        img_path = os.path.join(img_dir, name)

        with Image.open(img_path) as im:
            width, height = im.size

        images.append({
            "id": image_id,
            "file_name": name,
            "width": width,
            "height": height
        })

        label_path = os.path.join(
            label_dir,
            os.path.splitext(name)[0] + ".txt"
        )

        if os.path.exists(label_path):
            with open(label_path, "r") as f:
                for line in f:
                    parts = line.strip().split()

                    if len(parts) < 9:
                        continue

                    try:
                        coords = list(map(float, parts[:8]))
                    except ValueError:
                        continue

                    cls_name = parts[8]

                    if cls_name not in CAT2ID:
                        continue

                    difficult = 0
                    if len(parts) >= 10:
                        try:
                            difficult = int(parts[9])
                        except ValueError:
                            difficult = 0

                    # DOTA difficult=2 usually ignored
                    if difficult == 2:
                        continue

                    xs = coords[0::2]
                    ys = coords[1::2]

                    xmin = max(0.0, min(xs))
                    ymin = max(0.0, min(ys))
                    xmax = min(float(width), max(xs))
                    ymax = min(float(height), max(ys))

                    bw = xmax - xmin
                    bh = ymax - ymin

                    if bw <= 1 or bh <= 1:
                        continue

                    annotations.append({
                        "id": ann_id,
                        "image_id": image_id,
                        "category_id": CAT2ID[cls_name],
                        "bbox": [xmin, ymin, bw, bh],
                        "area": bw * bh,
                        "iscrowd": 0
                    })

                    class_count[cls_name] += 1
                    ann_id += 1

        image_id += 1

        if idx % 2000 == 0:
            print("processed:", idx, "/", len(files), flush=True)

    categories = [
        {
            "id": i + 1,
            "name": name,
            "supercategory": "object"
        }
        for i, name in enumerate(CLASSES)
    ]

    data = {
        "images": images,
        "annotations": annotations,
        "categories": categories
    }

    with open(out_json, "w") as f:
        json.dump(data, f)

    print("\n===== DONE =====")
    print("images      =", len(images))
    print("annotations =", len(annotations))
    print("categories  =", len(categories))

    print("\n===== CLASS COUNTS =====")
    for cls in CLASSES:
        print(f"{cls:22s} {class_count[cls]}")

    print("\nsaved:", out_json)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--img-dir", required=True)
    parser.add_argument("--label-dir", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    convert(args.img_dir, args.label_dir, args.out)
