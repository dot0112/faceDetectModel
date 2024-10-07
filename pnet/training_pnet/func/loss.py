import tensorflow as tf
import math


def binary_loss(y_true, y_pred):
    y_true = tf.clip_by_value(
        y_true, tf.keras.backend.epsilon(), tf.subtract(1.0, tf.keras.backend.epsilon())
    )
    y_pred = tf.clip_by_value(
        y_pred, tf.keras.backend.epsilon(), tf.subtract(1.0, tf.keras.backend.epsilon())
    )
    loss = y_true * -1.0 * tf.math.log(y_pred) + (1.0 - y_true) * -1.0 * tf.math.log(
        1.0 - y_pred
    )
    return tf.reduce_mean(loss)


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

    true_width, true_height = (
        tf.add(tf.subtract(true_x2, true_x1), tf.keras.backend.epsilon()),
        tf.add(tf.subtract(true_y2, true_y1), tf.keras.backend.epsilon()),
    )
    pred_width, pred_height = (
        tf.add(tf.subtract(pred_x2, pred_x1), tf.keras.backend.epsilon()),
        tf.add(tf.subtract(pred_y2, pred_y1), tf.keras.backend.epsilon()),
    )

    true_x_center, true_y_center = (
        tf.divide(tf.add(true_x1, true_x2), 2),
        tf.divide(tf.add(true_y1, true_y2), 2),
    )
    pred_x_center, pred_y_center = (
        tf.divide(tf.add(pred_x1, pred_x2), 2),
        tf.divide(tf.add(pred_y1, pred_y2), 2),
    )

    inter_x1 = tf.maximum(true_x1, pred_x1)
    inter_y1 = tf.maximum(true_y1, pred_y1)
    inter_x2 = tf.minimum(true_x2, pred_x2)
    inter_y2 = tf.minimum(true_y2, pred_y2)

    inter_width = tf.maximum(tf.subtract(inter_x2, inter_x1), 0.0)
    inter_height = tf.maximum(tf.subtract(inter_y2, inter_y1), 0.0)
    inter_area = tf.multiply(inter_width, inter_height)

    true_area = tf.multiply(true_width, true_height)
    pred_area = tf.multiply(pred_width, pred_height)
    union_area = tf.subtract(tf.add(true_area, pred_area), inter_area)

    iou = tf.divide(inter_area, tf.add(union_area, tf.keras.backend.epsilon()))

    center_dist = tf.add(
        tf.square(tf.subtract(pred_x_center, true_x_center)),
        tf.square(tf.subtract(pred_y_center, true_y_center)),
    )

    c = tf.add(
        tf.square(
            tf.subtract(tf.maximum(pred_x2, true_x2), tf.minimum(pred_x1, true_x1))
        ),
        tf.square(
            tf.subtract(tf.maximum(pred_y2, true_y2), tf.minimum(pred_y1, true_y1))
        ),
    )

    v = tf.multiply(
        tf.divide(8.0, (tf.square(tf.constant(math.pi, dtype=y_true.dtype)))),
        tf.square(
            tf.subtract(
                tf.atan(tf.divide(true_width, true_height)),
                tf.atan(tf.divide(pred_width, pred_height)),
            )
        ),
    )

    alpha = tf.divide(v, tf.add(tf.subtract(1.0, iou), v))
    ciou = tf.subtract(
        tf.subtract(iou, tf.divide(center_dist, tf.add(c, tf.keras.backend.epsilon()))),
        tf.multiply(alpha, v),
    )

    loss = tf.subtract(1.0, ciou)
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

    loss_class = tf.multiply(weight_for_class, loss_class)
    loss_bbox = tf.multiply(weight_for_bbox, loss_bbox)

    total_loss = tf.add(loss_class, loss_bbox)
    return tf.reduce_mean(total_loss)


def combined_loss_wrapper(y_true, y_pred):
    y_true_class = y_true[:, 0, 0, 0]
    y_true_bbox = y_true[:, 0, 0, 1:]

    y_pred_class = y_pred[:, 0, 0, 0]
    y_pred_bbox = y_pred[:, 0, 0, 1:]

    return combined_loss(y_true_class, y_pred_class, y_true_bbox, y_pred_bbox)
