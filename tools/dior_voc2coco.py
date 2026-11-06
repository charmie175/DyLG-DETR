"""Convert the official DIOR VOC-style annotations to COCO JSON."""
from __future__ import annotations

import argparse
import json
import xml.etree.ElementTree as ET
from pathlib import Path


CATEGORIES = [
    "aeroplane", "airport", "baseball stadium", "basketball court", "bridge",
    "chimney", "dam", "motorway service area", "motorway toll booth", "golf course",
    "athletics track", "port", "flyover", "vessels", "stadiums", "storage tanks",
    "tennis courts", "railway stations", "vehicles", "windmills",
]
ALIASES = {
    "airplane": "aeroplane", "baseballfield": "baseball stadium",
    "baseball field": "baseball stadium", "basketballcourt": "basketball court",
    "expressway-service-area": "motorway service area", "expressway service area": "motorway service area",
    "expressway-toll-station": "motorway toll booth", "expressway toll station": "motorway toll booth",
    "golffield": "golf course", "golf field": "golf course", "groundtrackfield": "athletics track",
    "ground track field": "athletics track", "harbor": "port", "overpass": "flyover",
    "ship": "vessels", "stadium": "stadiums", "storagetank": "storage tanks",
    "storage tank": "storage tanks", "tenniscourt": "tennis courts", "trainstation": "railway stations",
    "vehicle": "vehicles", "windmill": "windmills",
}


def find_split_file(root, split):
    candidates = [root / "ImageSets" / "Main" / f"{split}.txt", root / "ImageSets" / "Main" / f"{split}val.txt"]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"DIOR split file not found for {split}: {candidates}")


def convert(root, split, output):
    split_file = find_split_file(root, split)
    names = [line.strip().split()[0] for line in split_file.read_text(encoding="utf-8").splitlines() if line.strip()]
    categories = [{"id": i + 1, "name": name} for i, name in enumerate(CATEGORIES)]
    category_ids = {name: i + 1 for i, name in enumerate(CATEGORIES)}
    images, annotations = [], []
    ann_id = 1
    for image_id, name in enumerate(names, 1):
        xml_path = root / "Annotations" / f"{name}.xml"
        if not xml_path.exists():
            xml_path = root / "Annotations" / "Horizontal Bounding Boxes" / f"{name}.xml"
        if not xml_path.exists():
            raise FileNotFoundError(xml_path)
        xml = ET.parse(xml_path).getroot()
        size = xml.find("size")
        width = int(float(size.findtext("width", "0"))) if size is not None else 0
        height = int(float(size.findtext("height", "0"))) if size is not None else 0
        images.append({"id": image_id, "file_name": f"{name}.jpg", "width": width, "height": height})
        for obj in xml.findall("object"):
            raw_name = " ".join((obj.findtext("name") or "").lower().replace("_", "-").split())
            class_name = ALIASES.get(raw_name, raw_name)
            if class_name not in category_ids:
                raise ValueError(f"Unknown DIOR category {raw_name!r} in {xml_path}")
            box = obj.find("bndbox")
            x1 = float(box.findtext("xmin")) - 1.0
            y1 = float(box.findtext("ymin")) - 1.0
            x2 = float(box.findtext("xmax"))
            y2 = float(box.findtext("ymax"))
            w, h = max(0.0, x2 - x1), max(0.0, y2 - y1)
            annotations.append({"id": ann_id, "image_id": image_id, "category_id": category_ids[class_name],
                                "bbox": [x1, y1, w, h], "area": w * h, "iscrowd": 0})
            ann_id += 1
    result = {"images": images, "annotations": annotations, "categories": categories}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {output}: {len(images)} images, {len(annotations)} instances")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dior-root", required=True, type=Path)
    parser.add_argument("--split", choices=["train", "val", "test", "trainval"], required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    convert(args.dior_root, args.split, args.output)
