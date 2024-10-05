from create_pyramid import create_pyramid
from window_sliding import window_sliding, class_count
from tqdm import tqdm


def create_dataset(
    imageset_name: str,
    model_name: str,
    image_paths: list[str],
    labels: list[list[int]],
    window_size: int,
    resize_factor: float,
    batch_size: int,
):
    for batch_num in range(0, len(image_paths), batch_size):
        batch = [
            image_paths[batch_num : batch_num + batch_size],
            labels[batch_num : batch_num + batch_size],
        ]
        for image_path, label in tqdm(
            zip(*batch),
            total=batch_size,
            desc=f"create dataset",
        ):
            pyramid_images, pyramid_labels = create_pyramid(
                image_path, label, window_size, resize_factor
            )
            for p_image, p_label in zip(pyramid_images, pyramid_labels):
                window_sliding(imageset_name, model_name, p_image, p_label, window_size)
        print(f"0: {class_count[0]}\t1: {class_count[1]}\t2: {class_count[2]}")
        c = input("Enter '0' to stop, or press Enter to continue: ")
        if c == "0":
            break
