# Clinical Loss Functions
import tensorflow as tf


def version():
    # wait until python 3.11
    # with open("pyproject.toml", "rb") as f:
    #     data = tomllib.load(f)
    #     print(data["tool.poetry"])
    with open("pyproject.toml", encoding="utf-8") as f:
        read_data = f.readline()
        read_data = f.readline()
        read_data = f.readline()
        return read_data.split(" ")[2][1:-2]


VERSION = version()


def clinical_loss_fn(y_true, y_pred):
    """
    TODO: actually do it
    """
    squared_difference = tf.square(y_true - y_pred)
    return tf.reduce_mean(squared_difference, axis=-1)  # Note the `axis=-1`
