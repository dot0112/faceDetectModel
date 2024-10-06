import tensorflow as tf
import math
from keras import models


def binary_loss(true, pred):
    true = tf.clip_by_value(
        true, tf.keras.backend.epsilon(), 1 - tf.keras.backend.epsilon()
    )
    pred = tf.clip_by_value(
        pred, tf.keras.backend.epsilon(), 1 - tf.keras.backend.epsilon()
    )
    loss = true * -1 * tf.math.log(pred) + (1 - true) * -1 * tf.math.log(1 - pred)
    return tf.reduce_mean(loss)


def bbox_loss(true, pred):
    true_x1, true_y1, true_x2, true_y2 = (
        true[:, 0],
        true[:, 1],
        true[:, 2],
        true[:, 3],
    )
    pred_x1, pred_y1, pred_x2, pred_y2 = (
        pred[:, 0],
        pred[:, 1],
        pred[:, 2],
        pred[:, 3],
    )

    true_x_center, true_y_center = (true_x1 + true_x2) / 2, (true_y1 + true_y2) / 2
    pred_x_center, pred_y_center = (pred_x1 + pred_x2) / 2, (pred_y1 + pred_y2) / 2

    inter_x1 = tf.maximum(true_x1, pred_x1)
    inter_y1 = tf.maximum(true_y1, pred_y1)
    inter_x2 = tf.minimum(true_x2, pred_x2)
    inter_y2 = tf.minimum(true_y2, pred_y2)

    inter_width = tf.maximum(inter_x2 - inter_x1, 0.0)
    inter_height = tf.maximum(inter_y2 - inter_y1, 0.0)
    inter_area = inter_width * inter_height

    true_area = (true_x2 - true_x1) * (true_y2 - true_y1)
    pred_area = (pred_x2 - pred_x1) * (pred_y2 - pred_y1)
    union_area = true_area + pred_area - inter_area

    iou = inter_area / (union_area + 1e-7)

    center_dist = (pred_x_center - true_x_center) ** 2 + (
        pred_y_center - true_y_center
    ) ** 2

    c = (tf.maximum(true_x2, pred_x2) - tf.minimum(true_x1, pred_x1)) ** 2 + (
        tf.maximum(true_y2, pred_y2) - tf.minimum(true_y1, pred_y1)
    ) ** 2

    v = (4.0 / (tf.constant(math.pi, dtype=true.dtype) ** 2)) * tf.square(
        tf.atan((true_x2 - true_x1) / (true_y2 - true_y1 + 1e-7))
        - tf.atan((pred_x2 - pred_x1) / (pred_y2 - pred_y1 + 1e-7))
    )

    alpha = v / ((1.0 - iou) + v + 1e-7)
    ciou = iou - (center_dist / (c + 1e-7)) - alpha * v

    loss = 1.0 - ciou
    return loss


def extract_labels(model_name, true_labels, pred_labels):
    if model_name == "pnet":
        true_class_labels = true_labels[:, 0, 0, 0]
        true_bbox_labels = true_labels[:, 0, 0, 1:]
        pred_class_labels = pred_labels[:, 0, 0, 0]
        pred_bbox_labels = pred_labels[:, 0, 0, 1:]
    elif model_name in ["rnet", "onet"]:
        true_class_labels = true_labels[:, 0]
        true_bbox_labels = true_labels[:, 1:]
        pred_class_labels = pred_labels[:, 0]
        pred_bbox_labels = pred_labels[:, 1:]
    else:
        raise ValueError(f"Unsupported model: {model_name}")
    return true_class_labels, true_bbox_labels, pred_class_labels, pred_bbox_labels


def compare(
    model_name: str,
    model: models.Model,
    image_tensors: list[tf.Tensor],
    label_tensors: list[tf.Tensor],
) -> tuple[any]:
    prediction = model.predict(tf.stack(image_tensors), verbose=1)
    true_labels = tf.stack(label_tensors, axis=0)
    pred_labels = tf.stack(prediction["combined_outputs"])

    try:
        true_class, true_bbox, pred_class, pred_bbox = extract_labels(
            model_name, true_labels, pred_labels
        )
    except ValueError as e:
        print(f"\033[1;5;31m{str(e)}\033[0m")

    class_label_loss = binary_loss(true_class, pred_class)
    bbox_label_loss = bbox_loss(true_bbox, pred_bbox)
    return (tf.reduce_mean(class_label_loss), tf.reduce_mean(bbox_label_loss))


def verify(
    model_name: str,
    model_path: str,
    image_tensors: list[tf.Tensor],
    label_tensors: list[tf.Tensor],
) -> tuple[any]:
    model = models.load_model(model_path, compile=False)
    results = compare(model_name, model, image_tensors, label_tensors)
    return results
