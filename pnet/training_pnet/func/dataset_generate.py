import tensorflow as tf
import math
from tqdm import tqdm


def preprocess_image(image_path: str) -> tf.Tensor:
    image_path = image_path.numpy().decode("utf-8")

    # image_tensor = shape(12, 12, 3)
    image_tensor = tf.io.read_file(image_path)
    image_tensor = tf.image.decode_jpeg(image_tensor, channels=3)
    image_tensor = tf.image.resize(image_tensor, [12, 12])
    image_tensor = tf.cast(image_tensor, tf.float32) / 255.0

    return image_tensor


def preprocess_class_label(class_label: int) -> tf.Tensor:

    # class_label = shape(1, 1, 1)
    class_label = tf.convert_to_tensor(class_label, dtype=tf.int32)
    class_label = tf.cast(class_label, dtype=tf.float32)
    class_label = tf.reshape(class_label, [-1, 1, 1])

    return class_label


def preprocess_bbox_label(bbox_label: list[float]) -> tf.Tensor:

    # bbox_label = shape(1, 1, 4)
    bbox_label = tf.convert_to_tensor(bbox_label, dtype=tf.float32)
    bbox_label = tf.reshape(bbox_label, [-1, 1, 4])

    return bbox_label


def preprocess_data(
    image_path: str, class_label: int, bbox_label: list[float]
) -> tuple:
    try:
        image_tensor = preprocess_image(image_path)
        class_label = preprocess_class_label(class_label)
        bbox_label = preprocess_bbox_label(bbox_label)

        return image_tensor, class_label, bbox_label

    except Exception as e:
        tf.print(f"Error processing {image_path}, {class_label}, {bbox_label}: {e}")
        return [
            tf.zeros((12, 12, 3), dtype=tf.float32),
            tf.zeros((1, 1, 1), dtype=tf.float32),
            tf.zeros((1, 1, 4), dtype=tf.float32),
        ]


def preprocess_wrapper(
    image_path: str, class_label: int, bbox_label: list[float]
) -> tuple:
    image_tensor, class_label, bbox_label = tf.py_function(
        preprocess_data,
        [image_path, class_label, bbox_label],
        [tf.float32, tf.float32, tf.float32],
    )
    image_tensor.set_shape((12, 12, 3))
    class_label.set_shape((1, 1, 1))
    bbox_label.set_shape((1, 1, 4))
    combined_label = tf.concat([class_label, bbox_label], axis=-1)
    return image_tensor, {"combined_outputs": combined_label}


def create_dataset(
    image_paths: list[str],
    class_labels: list[int],
    bbox_labels: list[list[float]],
    batch_size: int,
    split_count: int,
) -> list[tf.data.Dataset]:
    data_length = len(image_paths)
    chunk_size = math.ceil(data_length / split_count)

    split_image_paths = [
        image_paths[i : i + chunk_size] for i in range(0, data_length, chunk_size)
    ]
    split_class_labels = [
        class_labels[i : i + chunk_size] for i in range(0, data_length, chunk_size)
    ]
    split_bbox_labels = [
        bbox_labels[i : i + chunk_size] for i in range(0, data_length, chunk_size)
    ]

    split_datasets = []

    for _image_paths, _class_labels, _bbox_labels in tqdm(
        zip(split_image_paths, split_class_labels, split_bbox_labels),
        total=split_count,
        desc="Creating datasets",
    ):
        dataset = tf.data.Dataset.from_tensor_slices(
            (_image_paths, _class_labels, _bbox_labels)
        )
        dataset = dataset.map(preprocess_wrapper)
        dataset = dataset.batch(batch_size).prefetch(buffer_size=tf.data.AUTOTUNE)
        split_datasets.append(dataset)

    return split_datasets


def dataset_generate(image_paths, class_labels, bbox_labels, batch_size, split_count):
    datasets = create_dataset(
        image_paths, class_labels, bbox_labels, batch_size, split_count
    )
    return datasets
