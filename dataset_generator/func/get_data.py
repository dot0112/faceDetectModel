import os
from pathlib import Path
from tqdm import tqdm
from dotenv import load_dotenv


def get_data(imageset_name: str) -> tuple[list[str], list[list[int]]]:
    match imageset_name:
        case "celeba":
            return get_data_celeba()
        case "wider":
            return get_data_wider()

        case _:
            error_message = f"Unsupported imageset: {imageset_name}"
            raise ValueError(f"\033[1;5;31m{error_message}\033[0m")


def get_data_celeba() -> tuple[list[str], list[list[int]]]:
    load_dotenv()
    root_dir_path = Path(os.getenv("imageset_path")) / "img_celeba"
    image_path = root_dir_path / "img_celeba"
    label_path = root_dir_path / "list_bbox_celeba.txt"

    image_paths = []
    labels = []

    with open(label_path, "r", encoding="utf-8") as file:
        total_lines = int(file.readline())
        for line in tqdm(file, total=total_lines, desc="Get Data From: [CELEBA]"):
            split_line = line.strip().split()
            if split_line[0][-1] != "g":
                continue
            image_paths.append(str(image_path / split_line[0]))
            labels.append(list(map(float, split_line[1:])))

    return image_paths, labels


def get_data_wider():
    load_dotenv()
    root_dir_path = Path(os.getenv("imageset_path")) / "wider_dataset"
    image_path = root_dir_path / "WIDER_train" / "images"
    label_path_1 = root_dir_path / "wider_face_split" / "wider_face_train_bbx_gt.txt"
    label_path_2 = root_dir_path / "wider_face_split" / "wider_face_val_bbx_gt.txt"

    image_paths = []
    labels = []

    for label_path in (label_path_1, label_path_2):
        with open(label_path, "r", encoding="utf-8") as file:
            total_lines = len(file.readlines())
            file.seek(0)
            count = 0
            image_name = True
            temp_labels = []
            for line in tqdm(file, total=total_lines, desc="Get Data From: [WIDER]"):
                if count != 0:  # bbox label
                    count -= 1
                    label = line.strip().split()
                    temp_labels.append([int(n) for n in label[:4]])
                    if count == 0:
                        image_name = True
                        labels.append(temp_labels)
                        temp_labels = []
                elif image_name:  # image name
                    image_paths.append(line)
                    image_name = False
                else:  # label count
                    count = max(1, int(line))

    return image_paths, labels
