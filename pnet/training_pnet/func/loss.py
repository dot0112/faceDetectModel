import tensorflow as tf
import math


def binary_loss(y_true, y_pred):
    y_true = tf.clip_by_value(y_true, 0, 1)
    y_pred = tf.clip_by_value(
        y_pred, tf.keras.backend.epsilon(), 1 - tf.keras.backend.epsilon()
    )
    loss = y_true * -1 * tf.math.log(y_pred) + (1 - y_true) * -1 * tf.math.log(
        1 - y_pred
    )
    return tf.reduce_mean(loss)


# def bbox_loss(y_true, y_pred):
#     true_x_min, true_y_min, true_width, true_height = (
#         y_true[:, 0],
#         y_true[:, 1],
#         y_true[:, 2],
#         y_true[:, 3],
#     )
#     pred_x_min, pred_y_min, pred_width, pred_height = (
#         y_pred[:, 0],
#         y_pred[:, 1],
#         y_pred[:, 2],
#         y_pred[:, 3],
#     )

#     true_x_center, true_y_center = (
#         tf.add(true_x_min, tf.divide(true_width, 2.0)),
#         tf.add(true_y_min, tf.divide(true_height, 2.0)),
#     )
#     pred_x_center, pred_y_center = (
#         tf.add(pred_x_min, tf.divide(pred_width, 2.0)),
#         tf.add(pred_y_min, tf.divide(pred_height, 2.0)),
#     )
#     true_x_max, true_y_max = tf.add(true_x_min, true_width), tf.add(
#         true_y_min, true_height
#     )
#     pred_x_max, pred_y_max = tf.add(pred_x_min, pred_width), tf.add(
#         pred_y_min, pred_height
#     )

#     inter_x_min, inter_y_min, inter_x_max, inter_y_max = (
#         tf.maximum(pred_x_min, true_x_min),
#         tf.maximum(pred_y_min, true_y_min),
#         tf.minimum(pred_x_max, true_x_max),
#         tf.minimum(pred_y_max, true_y_max),
#     )
#     inter_width, inter_height = tf.maximum(inter_x_max - inter_x_min, 0.0), tf.maximum(
#         inter_y_max - inter_y_min, 0.0
#     )
#     inter_area = tf.multiply(inter_width, inter_height)

#     true_area = tf.multiply(true_width, true_height)
#     pred_area = tf.multiply(pred_width, pred_height)
#     union_area = tf.subtract(tf.add(true_area, pred_area), inter_area)

#     iou = tf.divide(inter_area, union_area)

#     center_dist = tf.square(tf.subtract(pred_x_center, true_x_center)) + tf.square(
#         tf.subtract(pred_y_center, true_y_center)
#     )

#     c = tf.square(
#         tf.subtract(
#             tf.maximum(pred_x_max, true_x_max), tf.minimum(pred_x_min, true_x_min)
#         )
#     ) + tf.square(
#         tf.subtract(
#             tf.maximum(pred_y_max, true_y_max), tf.minimum(pred_y_min, true_y_min)
#         )
#     )

#     v = (
#         tf.divide(8.0, (tf.square(tf.constant(math.pi, dtype=y_true.dtype))))
#     ) * tf.square(
#         tf.atan(tf.divide(true_width, true_height))
#         - tf.atan(tf.divide(pred_width, pred_height))
#     )

#     # v / (1 - iou + v)
#     alpha = tf.divide(v, tf.add(tf.subtract(1.0, iou), v))

#     # iou - (center_dist / c) - alpha * v
#     ciou = tf.subtract(
#         tf.subtract(iou, tf.divide(center_dist, c)), tf.multiply(alpha, v)
#     )

#     loss = tf.subtract(1.0, ciou)

#     return loss


def bbox_loss(y_true, y_pred):
    true_x1, true_y1, true_x2, true_y2 = (
        y_true[:, 0],
        y_true[:, 1],
        y_true[:, 2],
        y_true[:, 3],
    )
    pred_x1, pred_y1, pred_x2, pred_y2 = (
        y_pred[:, 0],
        y_pred[:, 1],
        y_pred[:, 2],
        y_pred[:, 3],
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

    v = (4.0 / (tf.constant(math.pi, dtype=y_true.dtype) ** 2)) * tf.square(
        tf.atan((true_x2 - true_x1) / (true_y2 - true_y1 + 1e-7))
        - tf.atan((pred_x2 - pred_x1) / (pred_y2 - pred_y1 + 1e-7))
    )

    alpha = v / ((1.0 - iou) + v + 1e-7)
    ciou = iou - (center_dist / (c + 1e-7)) - alpha * v

    loss = 1.0 - ciou
    return loss


def combined_loss(y_true_class, y_pred_class, y_true_bbox, y_pred_bbox):
    loss_class = binary_loss(y_true_class, y_pred_class)
    loss_bbox = bbox_loss(y_true_bbox, y_pred_bbox)

    weight_for_class = tf.where(
        tf.logical_or(tf.equal(y_true_class, 2.0), tf.equal(y_true_class, 0.0)),
        1.0,
        0.0,
    )

    weight_for_bbox = tf.where(
        tf.logical_or(tf.equal(y_true_class, 2.0), tf.equal(y_true_class, 1.0)),
        0.5,
        0.0,
    )

    total_loss = tf.add(
        tf.multiply(weight_for_class, loss_class),
        tf.multiply(weight_for_bbox, loss_bbox),
    )
    return tf.reduce_mean(total_loss)


def combined_loss_wrapper(y_true, y_pred):
    y_true_class = y_true[:, 0, 0, 0]
    y_true_bbox = y_true[:, 0, 0, 1:]

    y_pred_class = y_pred[:, 0, 0, 0]
    y_pred_bbox = y_pred[:, 0, 0, 1:]

    return combined_loss(y_true_class, y_pred_class, y_true_bbox, y_pred_bbox)
