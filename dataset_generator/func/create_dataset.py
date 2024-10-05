from create_pyramid import create_pyramid
from window_sliding import window_sliding


def create_dataset(
    imageset_name: str,
    model_name: str,
    image_paths: list[str],
    labels: list[list[int]],
    window_size: int,
    resize_factor: float,
    batch_size: int,
):
    for image_path, label in zip(image_paths, labels):
        pyramid_images, pyramid_labels = create_pyramid(
            image_path, label, window_size, resize_factor
        )
        for p_image, p_label in zip(pyramid_images, pyramid_labels):
            window_sliding(imageset_name, model_name, p_image, p_label, window_size)
