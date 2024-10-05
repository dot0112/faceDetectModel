import tensorflow as tf
from datetime import datetime


class DisplayTimeCallback(tf.keras.callbacks.Callback):
    def __init__(
        self,
    ):
        super(DisplayTimeCallback, self).__init__()

    def on_epoch_begin(self, epoch, logs=None):
        print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))


def get_callback():
    display_time = DisplayTimeCallback()
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        filepath=f"./pnet/pnet_{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}.keras",
        monitor="val_loss",
        save_best_only=True,
        save_weights_only=False,
        mode="min",
        verbose=1,
    )
    return [display_time, checkpoint]
