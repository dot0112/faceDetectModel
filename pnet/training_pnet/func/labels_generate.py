import os
import concurrent.futures
import ujson as json
from pathlib import Path
from tqdm import tqdm
from dotenv import load_dotenv
from .lists import shuffle_lists, split_lists


def get_label(
    image_path: str, part: int, dataset_name: str, root_dir: str, index: int
) -> tuple:
    file_name = Path(image_path).stem
    file_name_split = file_name.split("_")
    label_name = f"{(file_name_split[0])}_{file_name_split[-1]}"

    label_path = root_dir / dataset_name / str(part) / "labels" / f"{label_name}.json"

    with open(label_path, "r", encoding="utf-8") as file:
        loaded_data = json.load(file)

    return loaded_data["class"], loaded_data["bbox"], index


def get_labels(
    image_paths: list[str], part: int, dataset_name: str, root_dir: str
) -> tuple[list, list]:
    bbox_labels = [None] * len(image_paths)
    class_labels = [None] * len(image_paths)
    part_type = {2: "positive", 1: "partitial", 0: "negative"}

    with concurrent.futures.ThreadPoolExecutor() as executor:
        future_to_path = {
            executor.submit(get_label, path, part, dataset_name, root_dir, idx): (
                path,
                idx,
            )
            for idx, path in enumerate(image_paths)
        }

        for future in tqdm(
            concurrent.futures.as_completed(future_to_path),
            total=len(image_paths),
            desc=f"[get_labels: {part_type[part]}]",
            unit="image",
        ):
            class_label, bbox_label, index = future.result()
            class_labels[index] = class_label
            bbox_labels[index] = bbox_label

    return class_labels, bbox_labels


def labels_generate(dataset_name: str) -> tuple[list]:
    load_dotenv()
    root_dir = Path(os.environ["dataset_path"]) / "pnet"
    positive_image_paths = [
        str(path) for path in (root_dir / dataset_name / "2" / "images").glob("*.jpg")
    ]
    partitial_image_paths = [
        str(path) for path in (root_dir / dataset_name / "1" / "images").glob("*.jpg")
    ]
    negative_image_paths = [
        str(path) for path in (root_dir / dataset_name / "0" / "images").glob("*.jpg")
    ]
    positive_class_labels, positive_bbox_labels = get_labels(
        positive_image_paths, 2, dataset_name, root_dir
    )
    partitial_class_labels, partitial_bbox_labels = get_labels(
        partitial_image_paths, 1, dataset_name, root_dir
    )
    negative_class_labels, negative_bbox_labels = get_labels(
        negative_image_paths, 0, dataset_name, root_dir
    )
    positive_image_paths, positive_class_labels, positive_bbox_labels = shuffle_lists(
        [positive_image_paths, positive_class_labels, positive_bbox_labels]
    )
    partitial_image_paths, partitial_class_labels, partitial_bbox_labels = (
        shuffle_lists(
            [partitial_image_paths, partitial_class_labels, partitial_bbox_labels]
        )
    )
    negative_image_paths, negative_class_labels, negative_bbox_labels = shuffle_lists(
        [negative_image_paths, negative_class_labels, negative_bbox_labels]
    )
    (
        train_positive_image_paths,
        val_positive_image_paths,
        train_positive_class_labels,
        val_positive_class_labels,
        train_positive_bbox_labels,
        val_positive_bbox_labels,
    ) = split_lists(
        [positive_image_paths, positive_class_labels, positive_bbox_labels],
        split_ratio=0.9,
    )

    (
        train_partitial_image_paths,
        val_partitial_image_paths,
        train_partitial_class_labels,
        val_partitial_class_labels,
        train_partitial_bbox_labels,
        val_partitial_bbox_labels,
    ) = split_lists(
        [partitial_image_paths, partitial_class_labels, partitial_bbox_labels],
        split_ratio=0.9,
    )

    (
        train_negative_image_paths,
        val_negative_image_paths,
        train_negative_class_labels,
        val_negative_class_labels,
        train_negative_bbox_labels,
        val_negative_bbox_labels,
    ) = split_lists(
        [negative_image_paths, negative_class_labels, negative_bbox_labels],
        split_ratio=0.9,
    )

    train_image_paths = (
        train_positive_image_paths
        + train_partitial_image_paths
        + train_negative_image_paths
    )
    train_class_labels = (
        train_positive_class_labels
        + train_partitial_class_labels
        + train_negative_class_labels
    )
    train_bbox_labels = (
        train_positive_bbox_labels
        + train_partitial_bbox_labels
        + train_negative_bbox_labels
    )
    train_image_paths, train_class_labels, train_bbox_labels = shuffle_lists(
        [train_image_paths, train_class_labels, train_bbox_labels]
    )

    val_image_paths = (
        val_positive_image_paths + val_partitial_image_paths + val_negative_image_paths
    )
    val_class_labels = (
        val_positive_class_labels
        + val_partitial_class_labels
        + val_negative_class_labels
    )
    val_bbox_labels = (
        val_positive_bbox_labels + val_partitial_bbox_labels + val_negative_bbox_labels
    )
    val_image_paths, val_class_labels, val_bbox_labels = shuffle_lists(
        [val_image_paths, val_class_labels, val_bbox_labels]
    )

    return (
        train_image_paths,
        train_class_labels,
        train_bbox_labels,
        val_image_paths,
        val_class_labels,
        val_bbox_labels,
    )
