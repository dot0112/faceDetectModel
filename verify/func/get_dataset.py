import os
import concurrent.futures
import ujson as json
from dotenv import load_dotenv
from pathlib import Path


def get_image_paths(
    imageset_name: str, model_name: str, class_lable: int
) -> tuple[str]:
    load_dotenv()
    image_dir = (
        Path(os.environ["dataset_path"])
        / model_name
        / imageset_name
        / str(class_lable)
        / "images"
    )
    return [str(path) for path in image_dir.glob("*.jpg")]


def get_label(image_path: str, idx: int):
    label_path = image_path.replace("images", "labels")[:-3] + "json"
    with open(label_path, "r", encoding="utf-8") as file:
        loaded_data = json.load(file)
    return loaded_data["class"], loaded_data["bbox"], idx


def get_labels(
    image_paths: list[str],
) -> tuple[list, list]:
    bbox_labels = [None] * len(image_paths)
    class_labels = [None] * len(image_paths)

    with concurrent.futures.ThreadPoolExecutor() as executor:
        future_to_path = {
            executor.submit(get_label, path, idx): (path, idx)
            for idx, path in enumerate(image_paths)
        }

        for future in concurrent.futures.as_completed(future_to_path):
            class_label, bbox_label, index = future.result()
            class_labels[index] = class_label
            bbox_labels[index] = bbox_label

    return class_labels, bbox_labels


def get_dataset(imageset_name: str, model_name: str) -> tuple[list, list, list]:
    positive_image_paths = get_image_paths(imageset_name, model_name, 2)
    partitial_image_paths = get_image_paths(imageset_name, model_name, 1)
    negative_image_paths = get_image_paths(imageset_name, model_name, 0)

    positive_class_labels, positive_bbox_labels = get_labels(positive_image_paths)
    partitial_class_labels, partitial_bbox_labels = get_labels(partitial_image_paths)
    negative_class_labels, negative_bbox_labels = get_labels(negative_image_paths)

    return [
        [positive_image_paths, positive_class_labels, positive_bbox_labels],
        [partitial_image_paths, partitial_class_labels, partitial_bbox_labels],
        [negative_image_paths, negative_class_labels, negative_bbox_labels],
    ]
