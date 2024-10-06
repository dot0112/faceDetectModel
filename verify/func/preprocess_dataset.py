import tensorflow as tf
import concurrent.futures


def preprocess_image(image_path: str, idx: int) -> tf.Tensor:
    image_tensor = tf.io.read_file(image_path)
    image_tensor = tf.image.decode_jpeg(image_tensor, channels=3)
    image_tensor = tf.cast(image_tensor, tf.float32) / 255.0
    return image_tensor, idx


def preprocess_images(image_paths: list[str]) -> list[tf.Tensor]:
    image_tensors = [None] * len(image_paths)
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future_to_tensor = {
            executor.submit(preprocess_image, path, idx): (path, idx)
            for idx, path in enumerate(image_paths)
        }

        for future in concurrent.futures.as_completed(future_to_tensor):
            image_tensor, idx = future.result()
            image_tensors[idx] = image_tensor
    return image_tensors


def preprocess_label(
    tensor_depth: int, class_label: int, bbox_label: list[int], idx: int
) -> tf.Tensor:
    class_label = tf.convert_to_tensor(class_label, dtype=tf.int32)
    class_label = tf.cast(class_label, dtype=tf.float32)
    class_label = tf.reshape(class_label, [1] * (tensor_depth - 1) + [1])

    bbox_label = tf.convert_to_tensor(bbox_label, dtype=tf.float32)
    bbox_label = tf.reshape(bbox_label, [1] * (tensor_depth - 1) + [4])

    combined_label = tf.concat([class_label, bbox_label], axis=-1)
    return combined_label, idx


def preprocess_labels(
    tensor_depth: int, class_labels: list[int], bbox_labels: list[list[int]]
) -> list[tf.Tensor]:
    label_tensors = [None] * len(class_labels)

    with concurrent.futures.ThreadPoolExecutor() as executor:
        future_to_tensor = {
            executor.submit(
                preprocess_label, tensor_depth, class_label, bbox_label, idx
            ): (
                class_label,
                bbox_label,
                idx,
            )
            for idx, (class_label, bbox_label) in enumerate(
                zip(class_labels, bbox_labels)
            )
        }

        for future in concurrent.futures.as_completed(future_to_tensor):
            label_tensor, idx = future.result()
            label_tensors[idx] = label_tensor
    return label_tensors


def preprocess_dataset(
    model_name: str,
    dataset: tuple[list[str], list[int], list[list[int]]],
) -> tuple[list[tf.Tensor], list[tf.Tensor]]:
    if model_name == "pnet":
        tensor_depth = 3
    elif model_name in ["rnet", "onet"]:
        tensor_depth = 1
    else:
        error_message = f"Unsupported model: {model_name}"
        raise ValueError(f"\033[1;5;31m{error_message}\033[0m")

    image_paths = dataset[0]
    class_labels = dataset[1]
    bbox_labels = dataset[2]

    image_tensors = preprocess_images(image_paths)
    label_tensors = preprocess_labels(tensor_depth, class_labels, bbox_labels)

    return image_tensors, label_tensors
