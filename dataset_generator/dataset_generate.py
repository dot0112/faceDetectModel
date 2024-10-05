from func.get_data import get_data
from func.directory import create_directory
from func.list import shuffle_lists
from func.create_dataset import create_dataset


def dataset_generate(
    imageset_name: str,
    model_name: str = "pnet",
    window_size: int = 12,
    resize_factor: float = 0.709,
    batch_size: int = 5000,
):
    image_paths, labels = get_data(imageset_name)
    image_paths, labels = shuffle_lists(image_paths, labels)
    create_directory(imageset_name, model_name)
    create_dataset(
        imageset_name,
        model_name,
        image_paths,
        labels,
        window_size,
        resize_factor,
        batch_size,
    )
