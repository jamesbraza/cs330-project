import os
from collections.abc import Callable, Sequence
from typing import Literal, NamedTuple, TypeAlias

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

from data import DATA_DIR

Shape: TypeAlias = Sequence[int | None]


class DatasetMetadata(NamedTuple):
    """Simple class to house metadata on TensorFlow datasets from tfds."""

    name: str
    num_classes: int
    image_shape: Shape


DATASET_CONFIGS: dict[str, DatasetMetadata] = {
    "cifar10": DatasetMetadata("cifar10", 10, (32, 32, 3)),
    "cifar100": DatasetMetadata("cifar100", 100, (32, 32, 3)),
}

DEFAULT_SEED = 42
DEFAULT_NUM_DATASETS = 8
DEFAULT_NUM_CLASSES = 10
DEFAULT_BATCH_SIZE = 32
ALL = -1


def get_random_datasets(
    dataset: str = "cifar100",
    num_ds: int = DEFAULT_NUM_DATASETS,
    num_classes: int = DEFAULT_NUM_CLASSES,
    batch_size: int = DEFAULT_BATCH_SIZE,
    num_batches: int = ALL,
    subset: str = "train",
    seed: int = DEFAULT_SEED,
) -> list[tuple[tf.data.Dataset, np.ndarray]]:
    """
    Get list of dataset, label pairings.

    Args:
        dataset: Name of the TensorFlow dataset to load, default is CIFAR 100.
        num_ds: Number of datasets to fetch, default is 8.
        num_classes: Number of classes per dataset, default is 10.
        batch_size: Batch size to use in the random datasets.
        num_batches: Number of batches to take, default fetches all.
        subset: Subset to sample from, either train (default) or test.
        seed: Seed to use for getting random datasets.

    Returns:
        List of tuples of dataset, labels (int).
    """
    full_ds = tfds.load(name=dataset, split=subset, as_supervised=True)
    full_labels = np.fromiter(set(int(label) for _, label in full_ds), dtype=int)
    rng = np.random.default_rng(seed)
    labels = rng.choice(full_labels, size=(num_ds, num_classes), replace=False)
    return [
        (
            full_ds.filter(lambda x, y, row=i: tf.reduce_any(y == labels[row]))
            .batch(batch_size)
            .take(num_batches),
            labels[i],
        )
        for i in range(num_ds)
    ]


def get_dataset_subset(
    labels: Sequence[int],
    dataset: str = "cifar100",
    batch_size: int = DEFAULT_BATCH_SIZE,
    num_batches: int = ALL,
    subset: str = "train",
) -> tf.data.Dataset:
    """
    Get the subset of a dataset corresponding to the input labels.

    Args:
        labels: Sequence of labels to fetch
        dataset: Name of the TensorFlow dataset to load, default is CIFAR 100.
        batch_size: Batch size of the dataset.
        num_batches: Number of batches to take, default fetches all.
        subset: Subset to sample from, either train (default) or test.

    Returns:
        Matching dataset.
    """
    full_ds = tfds.load(name=dataset, split=subset, as_supervised=True)
    return (
        full_ds.filter(lambda x, y: tf.reduce_any(y == labels))
        .batch(batch_size)
        .take(num_batches)
    )


TRAIN_VAL_BASE_REL_PATH = "plant-diseases/New Plant Diseases Dataset(Augmented)/New Plant Diseases Dataset(Augmented)"
TRAIN_REL_PATH_PARTICLES: list[str] = [*TRAIN_VAL_BASE_REL_PATH.split("/"), "train"]
VAL_REL_PATH_PARTICLES: list[str] = [*TRAIN_VAL_BASE_REL_PATH.split("/"), "valid"]
PLANT_DISEASES_TRAIN = os.path.join(DATA_DIR, *TRAIN_REL_PATH_PARTICLES)
PLANT_DISEASES_VAL = os.path.join(DATA_DIR, *VAL_REL_PATH_PARTICLES)


class PlantLabel(NamedTuple):
    """Simple data structure to house the labels of the plants dataset."""

    raw: str
    coarse: str
    fine: str


def get_plant_diseases_datasets(
    num_train_batch: int = ALL,
    num_val_batch: int = ALL,
    seed: int = DEFAULT_SEED,
    subset: Literal["training", "validation"] = "train",
    **from_dir_kwargs,
) -> tuple[tf.data.Dataset, tf.data.Dataset, list[PlantLabel]]:
    """
    Get the training and validation subsets of the plants dataset.

    SEE: https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset

    Args:
        num_train_batch: Number of batches in the training subset, default fetches all.
        num_val_batch: Number of batches in the validation subset, default fetches all.
        seed: Seed to use for train-validation split.
        subset: Subset (default is training) of the plant diseases dataset use.
        from_dir_kwargs: Override keyword arguments to pass through to both
            tf.keras.utils.image_dataset_from_directory calls.

    Returns:
        Tuple of training dataset, validation dataset, labels.
            NOTE: for labels, indices correspond with ID, values correspond
            with string labels.
    """
    from_dir_kwargs = {"seed": seed, "validation_split": 0.1} | from_dir_kwargs
    directory: str = PLANT_DISEASES_TRAIN if subset == "train" else PLANT_DISEASES_VAL
    train_ds, val_ds = tf.keras.utils.image_dataset_from_directory(
        directory, subset="both", **from_dir_kwargs
    )
    # NOTE: indices correspond with ID, values correspond with string
    labels: list[PlantLabel] = [
        PlantLabel(raw_label, *raw_label.split("___"))
        for raw_label in set().union(train_ds.class_names, val_ds.class_names)
    ]
    # NOTE: .take() wipes away the ephemeral class_names attribute
    return train_ds.take(num_train_batch), val_ds.take(num_val_batch), labels


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

    # pylint: disable=unnecessary-lambda-assignment
    is_train = lambda x, y: not is_val(x, y)  # noqa: E731
    recover = lambda _, y: y  # noqa: E731
    train_ds = dataset.enumerate().filter(is_train).map(recover)
    val_ds = dataset.enumerate().filter(is_val).map(recover)
    return train_ds, val_ds


def preprocess(
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
        return x, tf.one_hot(y, depth=num_classes)

    return dataset.map(_preprocess)


def main() -> None:
    """Play around with code here."""
    labelled_datasets = get_random_datasets(num_batches=7)
    for ds, labels in labelled_datasets:
        for image, label in ds:
            _ = 0  # Debug here


if __name__ == "__main__":
    main()
