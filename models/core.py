from __future__ import annotations

import math
import tempfile
from collections.abc import Sequence

import tensorflow as tf

from data.dataset import Shape


class Conv2D(tf.keras.Model):
    def __init__(self, **conv2d_kwargs):
        super().__init__()
        self.conv = tf.keras.layers.Conv2D(**conv2d_kwargs)
        self.bn = tf.keras.layers.BatchNormalization()
        self.relu = tf.keras.layers.ReLU()

    def call(self, inputs, training=None, mask=None):
        output = self.conv(inputs)
        output = self.bn(output)
        return self.relu(output)


class TransferModel(tf.keras.Model):

    LAYER_3_NUM_FILTERS = 128
    LAYER_3_KERNEL_SIZE = (3, 3)

    def __init__(self, input_shape: Shape = (28, 28, 3), num_classes: int = 10):
        super().__init__()
        # Include input_layer so we can infer input shape for copy
        self.input_layer = tf.keras.layers.InputLayer(input_shape)
        self.layer1 = Conv2D(filters=64, kernel_size=(3, 3), strides=2, padding="valid")
        self.layer2 = Conv2D(
            filters=128, kernel_size=(3, 3), strides=2, padding="valid"
        )
        self.layer3 = Conv2D(
            filters=self.LAYER_3_NUM_FILTERS,
            kernel_size=self.LAYER_3_KERNEL_SIZE,
            strides=2,
            padding="valid",
        )
        self.pooling = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))
        self.flatten = tf.keras.layers.Flatten()
        self.dropout = tf.keras.layers.Dropout(rate=0.3)
        self.dense = tf.keras.layers.Dense(units=num_classes, activation="softmax")

    def call(self, inputs, training=None, mask=None):
        x = self.input_layer(inputs)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.pooling(x)
        x = self.flatten(x)
        x = self.dropout(x, training=training)
        return self.dense(x)

    def clone(
        self,
        new_num_classes: int,
        new_dense_weights: Sequence[float] | None = None,
    ) -> TransferModel:
        """
        Deep copy the model and change the number of units in the Dense layer.

        Sadly, tf.keras.models.clone_model doesn't support model subclasses yet per
        https://github.com/keras-team/keras/issues/17261.
        """
        with tempfile.NamedTemporaryFile(suffix=".h5") as fp:
            self.save_weights(fp.name)
            input_shape = self.input_layer.input_shape[0]
            copied_model = type(self)(
                input_shape=input_shape,
                num_classes=self.dense.units,
            )
            copied_model.build(input_shape=input_shape)
            copied_model.load_weights(fp.name)
        copied_model.dense = tf.keras.layers.Dense(
            units=new_num_classes, activation="softmax"
        )
        if new_dense_weights is not None:
            copied_model.dense.set_weights(new_dense_weights)
        return copied_model


class ReduceMatrix(tf.keras.Model):
    def __init__(self, conv_filter_dim: int, reduced_dim: int):
        super().__init__()
        self.A = tf.Variable(
            initial_value=tf.random_normal_initializer()(
                shape=(1, conv_filter_dim, reduced_dim), dtype="float32"
            ),
            trainable=True,
        )

    def call(self, inputs, training=None, mask=None):
        return tf.matmul(inputs, self.A)


class ChoiceNetSimple(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.layer_reduce = ReduceMatrix(
            conv_filter_dim=(
                TransferModel.LAYER_3_NUM_FILTERS
                * math.prod(TransferModel.LAYER_3_KERNEL_SIZE)
            ),
            reduced_dim=16,
        )
        self.flat = self.flatten = tf.keras.layers.Flatten()
        self.layer1 = tf.keras.layers.Dense(units=128, activation="relu")
        self.dropout = tf.keras.layers.Dropout(rate=0.3)
        self.dense = tf.keras.layers.Dense(units=1)

    def call(self, inputs, training=None, mask=None):
        tl_weights, ft_weights = inputs
        x = self.layer_reduce(tl_weights)
        x = self.flat(x)
        concat = tf.concat([x, ft_weights], axis=1)
        x = self.layer1(concat)
        x = self.dropout(x, training=training)
        return self.dense(x)
