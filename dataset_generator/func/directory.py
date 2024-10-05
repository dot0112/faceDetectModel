import os
from pathlib import Path
from dotenv import load_dotenv


def create_directory(imageset_name: str, model_name: str):
    load_dotenv()
    root_dir_path = Path(os.getenv("dataset_path")) / model_name / imageset_name
    root_dir_path.mkdir(parents=True, exist_ok=True)
    for i in range(3):
        (root_dir_path / str(i) / "images").mkdir(parents=True, exist_ok=True)
        (root_dir_path / str(i) / "labels").mkdir(parents=True, exist_ok=True)
