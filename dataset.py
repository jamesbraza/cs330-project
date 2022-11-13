import tensorflow as tf
import tensorflow_datasets as tfds

ds = tfds.load("cifar10", split="train", as_supervised=True)
ds = ds.prefetch(10).take(5)
for image, label in ds:
    _ = 0
