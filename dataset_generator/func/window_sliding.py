import os
import ujson as json
from PIL import Image
from calculate_IOU import cal_iou
from pathlib import Path
from dotenv import load_dotenv

class_count = {0: 0, 1: 0, 2: 0}


def window_sliding(
    imageset_name: str,
    model_name: str,
    image: Image.Image,
    labels: list[list[int]],
    window_size: int,
    stride: int = 2,
):
    load_dotenv()
    root_dir_path = Path(os.getenv("dataset_path")) / model_name / imageset_name

    w, h = image.size

    for y in range(0, h - window_size + 1, stride):
        for x in range(0, w - window_size + 1, stride):
            max_iou = 0
            target = [0, 0, 0, 0]
            for label in labels:
                iou = cal_iou(x, y, label, window_size)
                if max_iou < iou:
                    max_iou = iou
                    target = label

            class_label = 0
            if iou >= 0.65:
                class_label = 2
            elif iou >= 0.4:
                class_label = 1

            if class_label != 2:
                if class_count[2] * 2 < class_count[class_label]:
                    continue

            class_count[class_label] += 1

            window_save_path = (
                root_dir_path
                / str(class_label)
                / "images"
                / f"{class_label}_{class_count[class_label]}.jpg"
            )
            label_save_path = (
                root_dir_path
                / str(class_label)
                / "labels"
                / f"{class_label}_{class_count[class_label]}.json"
            )

            bbox_x1 = max(0, target[0] - x) / window_size
            bbox_y1 = max(0, target[1] - y) / window_size
            bbox_x2 = (
                max(min(x + window_size, target[0] + target[2]) - x, 0.0) / window_size
            )
            bbox_y2 = (
                max(min(y + window_size, target[1] + target[3]) - y, 0.0) / window_size
            )
            bbox = [bbox_x1, bbox_y1, bbox_x2, bbox_y2]

            label_json = {
                "class": class_label,
                "bbox": bbox,
            }

            with open(label_save_path, "w", encoding="utf-8") as file:
                json.dump(label_json, file, ensure_ascii=False, indent=4)

            window = image.crop([x, y, x + window_size, y + window_size])
            window.save(window_save_path, "JPEG")
