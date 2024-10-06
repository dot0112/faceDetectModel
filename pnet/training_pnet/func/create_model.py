from keras import layers, models, Input
from loss import binary_loss, bbox_loss, combined_loss, combined_loss_wrapper
import tensorflow as tf


def create_mtcnn_pnet_functional(input_shape):
    inputs = Input(shape=input_shape)
    x = layers.Conv2D(10, (3, 3), strides=(1, 1), padding="valid")(inputs)
    x = layers.PReLU()(x)
    x = layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)
    x = layers.Conv2D(16, (3, 3), strides=(1, 1), padding="valid")(x)
    x = layers.PReLU()(x)
    x = layers.Conv2D(32, (3, 3), strides=(1, 1), padding="valid")(x)
    x = layers.PReLU()(x)

    class_output = layers.Conv2D(
        1,
        (1, 1),
        strides=(1, 1),
        activation="sigmoid",
        padding="valid",
        name="class_output",
    )(x)

    bbox_output = layers.Conv2D(
        4, (1, 1), strides=(1, 1), padding="valid", name="bbox_output"
    )(x)

    combined_outputs = layers.concatenate(
        [class_output, bbox_output], name="combined_outputs"
    )

    model = models.Model(
        inputs=inputs,
        outputs={
            "combined_outputs": combined_outputs,
        },
    )

    return model


def create_model(input_shape, model_path=""):
    if model_path != "auto":
        model = models.load_model(
            model_path,
            custom_objects={
                "binary_loss": binary_loss,
                "bbox_loss": bbox_loss,
                "combined_loss": combined_loss,
                "combined_loss_wrapper": combined_loss_wrapper,
            },
        )
    else:
        model = create_mtcnn_pnet_functional(input_shape)
        model.compile(optimizer="adam", loss=combined_loss_wrapper)

    return model
