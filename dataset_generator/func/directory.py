import os
from pathlib import Path
from dotenv import load_dotenv


def create_directory(imageset_name: str, model_name: str):
    load_dotenv()
    root_dir_path = Path(os.getenv("dataset_path")) / model_name / imageset_name
    image_path = root_dir_path / "images"
    label_path = root_dir_path / "labels"
    for path in (root_dir_path, image_path, label_path):
        path.mkdir(parents=True, exist_ok=True)
