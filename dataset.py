from collections.abc import Callable
from typing import NamedTuple

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds


class DatasetMetadata(NamedTuple):
    name: str
    num_classes: int
    image_shape: tuple[int, int, int]


# Available datasets: https://www.tensorflow.org/datasets/catalog/overview
CIFAR10 = DatasetMetadata("cifar10", 10, (32, 32, 3))
CIFAR100 = DatasetMetadata("cifar100", 100, (32, 32, 3))

DEFAULT_NUM_DATASETS = 8
DEFAULT_NUM_CLASSES = 10
DEFAULT_NUM_EXAMPLES = -1


def get_random_datasets(
    dataset: str = CIFAR100.name,
    num_ds: int = DEFAULT_NUM_DATASETS,
    num_classes: int = DEFAULT_NUM_CLASSES,
    num_ex: int = DEFAULT_NUM_EXAMPLES,
    subset: str = "train",
) -> list[tuple[tf.data.Dataset, np.ndarray]]:
    """
    Get list of dataset, label pairings.

    Args:
        dataset: Name of the TensorFlow dataset to load, default is CIFAR 100.
        num_ds: Number of datasets to fetch, default is 8.
        num_classes: Number of classes per dataset, default is 10.
        num_ex: Number of examples within a dataset, default fetches all.
        subset: Subset to sample from, either train (default) or test.

    Returns:
        List of tuples of dataset, labels (int).
    """
    full_ds = tfds.load(name=dataset, split=subset, as_supervised=True)
    full_labels = np.fromiter(set(int(label) for _, label in full_ds), dtype=int)
    labels = np.random.choice(full_labels, size=(num_ds, num_classes), replace=False)
    return [
        (
            full_ds.filter(lambda x, y, row=i: tf.reduce_any(y == labels[row])).take(
                num_ex
            ),
            labels[i],
        )
        for i in range(num_ds)
    ]


def split(
    dataset: tf.data.Dataset, fraction: float = 0.1
) -> tuple[tf.data.Dataset, tf.data.Dataset]:
    """
    Split a dataset into two datasets.

    Args:
        dataset: Dataset to split into two datasets.
        fraction: Split fraction to split in [0, 1], in increments of 10%.
            Default split is 10%.

    Returns:
        Two-tuple of datasets.
    """
    fraction_decimal = int(fraction * 10)
    if fraction_decimal != fraction * 10:  # Validate input
        raise ValueError(f"Split fraction {fraction} must be a multiple of 10%.")

    def is_val(x, _) -> bool:
        return x % 10 < fraction_decimal

    is_train = lambda x, y: not is_val(x, y)  # noqa: E731
    recover = lambda _, y: y  # noqa: E731
    train_ds = dataset.enumerate().filter(is_train).map(recover)
    val_ds = dataset.enumerate().filter(is_val).map(recover)
    return train_ds, val_ds


def preprocess_dataset(
    dataset: tf.data.Dataset,
    num_classes: int = DEFAULT_NUM_CLASSES,
    image_preprocessor: Callable[[tf.Tensor], tf.Tensor] | None = None,
) -> tf.data.Dataset:
    """
    Preprocess the input dataset for training.

    Args:
        dataset: Dataset to preprocess.
        num_classes: Number of classes present in the dataset.
        image_preprocessor: Function to pre-process images per a model's requirements.
            Default of None will not try to pre-process images.
            Examples:
            - VGG16: tf.keras.applications.vgg16.preprocess_input.

    Returns:
        Preprocessed dataset.
    """

    def _preprocess(x: tf.Tensor, y: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        if image_preprocessor is not None:
            x = image_preprocessor(x)
        # NOTE: one_hot is a transformation step for Tensors, so we use it here
        # over to_categorical
        # pylint: disable=no-value-for-parameter
        return x, tf.one_hot(y, depth=num_classes)

    return dataset.map(_preprocess)


def main() -> None:
    """Play around with code here."""
    labelled_datasets = get_random_datasets(num_ex=200)
    for ds, labels in labelled_datasets:
        for image, label in ds.prefetch(10):
            _ = 0  # Debug here


if __name__ == "__main__":
    main()
