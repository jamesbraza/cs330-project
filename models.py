import os
from typing import TypeAlias

import tensorflow as tf

# Directory where models will be saved
MODEL_SAVE_DIR = os.path.join(os.path.dirname(__file__), "saved_models")

VGG_IMAGE_SIZE = (224, 224)
VGG_IMAGE_SHAPE = (*VGG_IMAGE_SIZE, 3)  # RGB
TopFCUnits: TypeAlias = tuple[int, ...]
VGG_TOP_FC_UNITS: TopFCUnits = (4096, 4096, 1000)  # From the paper to match ImageNet


def get_model(
    top_fc_units: TopFCUnits = VGG_TOP_FC_UNITS,
    image_shape: tuple[int, int, int] = VGG_IMAGE_SHAPE,
    base_model: tf.keras.Model | None = None,
) -> tf.keras.Model:
    """
    Make a VGG16 model given info on the top FC units.

    Args:
        top_fc_units: Number of units to use in each of the top FC layers.
            Default is three FC layers per the VGGNet paper.
            Last value in the tuple should match your number of classes.
        image_shape: Shape of the images input to the model.
            Default is the size of images per the VGGNet paper.
        base_model: Optional base model to use for transfer learning.
            If left as default of None, use Keras VGG16 trained on ImageNet.

    Returns:
        VGG16 model created.
    """
    if base_model is None:
        base_model = tf.keras.applications.VGG16(weights=None, include_top=False)
    dense_layers = [
        tf.keras.layers.Dense(units=units, activation="relu", name=f"fc{i+1}")
        for i, units in enumerate(top_fc_units[:-1])
    ]
    return tf.keras.Sequential(
        [
            tf.keras.Input(shape=image_shape),
            tf.keras.layers.Lambda(
                function=tf.keras.applications.vgg16.preprocess_input,
                name="preprocess_images",
            ),
            base_model,
            tf.keras.layers.Flatten(),
            *dense_layers,
            # Last layer matches number of classes
            tf.keras.layers.Dense(
                units=top_fc_units[-1], activation="softmax", name="predictions"
            ),
        ],
        name="tl_vgg16",
    )
