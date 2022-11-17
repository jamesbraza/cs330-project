import tensorflow as tf


class Conv2D(tf.keras.Model):
    def __init__(self, filters, kernel_size, strides, padding):
        super().__init__()
        self.conv = tf.keras.layers.Conv2D(
            filters=filters, kernel_size=kernel_size, strides=strides, padding=padding
        )
        self.bn = tf.keras.layers.BatchNormalization()
        self.relu = tf.keras.layers.ReLU()

    def call(self, inputs, training=None, mask=None):
        output = self.conv(inputs)
        output = self.bn(output)
        return self.relu(output)


class TransferModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.layer1 = Conv2D(filters=64, kernel_size=(3, 3), strides=2, padding="valid")
        self.layer2 = Conv2D(
            filters=128, kernel_size=(3, 3), strides=2, padding="valid"
        )
        self.layer3 = Conv2D(
            filters=128, kernel_size=(3, 3), strides=2, padding="valid"
        )
        self.pooling = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))
        self.flatten = tf.keras.layers.Flatten()
        self.dropout = tf.keras.layers.Dropout(rate=0.3)
        self.dense = tf.keras.layers.Dense(units=10, activation="softmax")

    def call(self, inputs, training=None, mask=None):
        x = self.layer1(inputs)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.pooling(x)
        x = self.flatten(x)
        x = self.dropout(x, training=training)
        return self.dense(x)


class ChoiceNetSimple(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.layer1 = tf.keras.layers.Dense(units=512, activation="relu")
        self.layer2 = tf.keras.layers.Dense(units=256, activation="relu")
        self.dropout = tf.keras.layers.Dropout(rate=0.1)
        self.dense = tf.keras.layers.Dense(units=1)

    def call(self, inputs, training=None, mask=None):
        x = self.layer1(inputs)
        x = self.layer2(x)
        x = self.dropout(x, training=training)
        return self.dense(x)
