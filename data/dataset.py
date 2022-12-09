import itertools
import os
from collections.abc import Callable, Sequence
from typing import NamedTuple, TypeAlias

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
    "bird-species": DatasetMetadata("bird-species", 450, (256, 256, 3)),
    "plant_village": DatasetMetadata("plant_village", 38, (256, 256, 3)),
    "plant-leaves": DatasetMetadata("plant-leaves", 22, (6000, 4000, 3)),
    "imagenet_resized/32x32": DatasetMetadata(
        "imagenet_resized/32x32", 1000, (32, 32, 3)
    ),
}

DEFAULT_SEED = 42
DEFAULT_NUM_DATASETS = 10
DEFAULT_NUM_CLASSES = 10
DEFAULT_BATCH_SIZE = 32
ALL = -1


def _batch_take(
    dataset: tf.data.Dataset,
    batch_size: int | None = DEFAULT_BATCH_SIZE,
    num_batches: int = ALL,
) -> tf.data.Dataset:
    if batch_size is not None:
        dataset = dataset.batch(batch_size)
    return dataset.take(num_batches)


def get_all_labels(dataset: tf.data.Dataset) -> np.ndarray:
    """
    Get all labels in a dataset.

    NOTE: labels should be integer values.
    NOTE: this traverses the entire dataset, so it's pretty lame.
    """
    return np.fromiter(set(int(label) for _, label in dataset), dtype=int)


def get_random_datasets(
    dataset: tf.data.Dataset,
    num_ds: int = DEFAULT_NUM_DATASETS,
    num_classes: int = DEFAULT_NUM_CLASSES,
    batch_size: int | None = DEFAULT_BATCH_SIZE,
    num_batches: int = ALL,
    seed: int = DEFAULT_SEED,
) -> list[tuple[tf.data.Dataset, np.ndarray]]:
    """
    Get list of dataset, label pairings.

    Args:
        dataset: Base dataset to randomly sample from.
        num_ds: Number of datasets to fetch, default is 8.
        num_classes: Number of classes per dataset, default is 10.
        batch_size: Batch size to use in the random datasets.
            If None, don't batch.
        num_batches: Number of batches to take, default fetches all.
        seed: Seed to use for getting random datasets.

    Returns:
        List of tuples of dataset, labels (int).
    """
    full_labels = get_all_labels(dataset)
    rng = np.random.default_rng(seed)
    labels = rng.choice(full_labels, size=(num_ds, num_classes), replace=False)
    return [
        (
            _batch_take(
                dataset.filter(lambda x, y, row=i: tf.reduce_any(y == labels[row])),
                batch_size,
                num_batches,
            ),
            labels[i],
        )
        for i in range(num_ds)
    ]


def get_dataset_subset(
    dataset: tf.data.Dataset,
    labels: Sequence[int],
    batch_size: int | None = DEFAULT_BATCH_SIZE,
    num_batches: int = ALL,
) -> tf.data.Dataset:
    """
    Get the subset of a dataset corresponding to the input labels.

    Args:
        dataset: Dataset to get a subset from.
        labels: Sequence of labels to fetch
        batch_size: Batch size of the dataset.
            If None, don't batch.
        num_batches: Number of batches to take, default fetches all.

    Returns:
        Matching dataset.
    """
    return _batch_take(
        dataset.filter(lambda x, y: tf.reduce_any(y == labels)), batch_size, num_batches
    )


PLANT_DISEASES_REL_PATH = "plant-diseases/New Plant Diseases Dataset(Augmented)/New Plant Diseases Dataset(Augmented)"
PLANT_DISEASES_TRAIN = os.path.join(
    DATA_DIR, *PLANT_DISEASES_REL_PATH.split("/"), "train"
)
PLANT_DISEASES_VAL = os.path.join(
    DATA_DIR, *PLANT_DISEASES_REL_PATH.split("/"), "valid"
)


def get_image_dataset_from_directory(
    train_dir: str,
    val_dir: str,
    num_train_batch: int = ALL,
    num_val_batch: int = ALL,
    seed: int = DEFAULT_SEED,
    **from_dir_kwargs,
) -> tuple[tf.data.Dataset, tf.data.Dataset, list[str]]:
    """
    Get the training and validation subsets of a dataset.

    Args:
        train_dir: Directory of the training subset.
        val_dir: Directory of the validation subset.
        num_train_batch: Number of batches in the training subset, default fetches all.
        num_val_batch: Number of batches in the validation subset, default fetches all.
        seed: Seed to use for train-validation split.
        from_dir_kwargs: Override keyword arguments to pass through to both
            tf.keras.utils.image_dataset_from_directory calls.

    Returns:
        Tuple of training dataset, validation dataset, labels.
    """
    from_dir_kwargs = {"seed": seed} | from_dir_kwargs
    train_ds = tf.keras.utils.image_dataset_from_directory(train_dir, **from_dir_kwargs)
    val_ds = tf.keras.utils.image_dataset_from_directory(val_dir, **from_dir_kwargs)
    # Use insertion-ordered dict for ordered set union hack
    union_labels: dict[str, None] = {
        cn: None for cn in itertools.chain(train_ds.class_names, val_ds.class_names)
    }
    # NOTE: .take() wipes away the ephemeral class_names attribute
    return (
        train_ds.take(num_train_batch),
        val_ds.take(num_val_batch),
        list(union_labels.keys()),
    )


class PlantLabel(NamedTuple):
    """Simple data structure to house the labels of the plants dataset."""

    raw: str
    coarse: str
    fine: str


def get_plant_diseases_datasets(
    num_train_batch: int = ALL,
    num_val_batch: int = ALL,
    seed: int = DEFAULT_SEED,
    **from_dir_kwargs,
) -> tuple[tf.data.Dataset, tf.data.Dataset, list[PlantLabel]]:
    """
    Get the training and validation subsets of the plant diseases dataset.

    SEE: https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset

    Returns:
        Tuple of training dataset, validation dataset, labels.
            NOTE: for labels, indices correspond with ID, values correspond
            with string labels.
    """
    train_ds, val_ds, raw_labels = get_image_dataset_from_directory(
        train_dir=PLANT_DISEASES_TRAIN,
        val_dir=PLANT_DISEASES_VAL,
        num_train_batch=num_train_batch,
        num_val_batch=num_val_batch,
        seed=seed,
        **from_dir_kwargs,
    )
    # NOTE: indices correspond with ID, values correspond with string
    labels: list[PlantLabel] = [
        PlantLabel(raw_label, *raw_label.split("___")) for raw_label in raw_labels
    ]
    return train_ds, val_ds, labels


PLANT_LEAVES_REL_PATH = "plant-leaves/Plants_2"
PLANT_LEAVES_TRAIN = os.path.join(DATA_DIR, *PLANT_LEAVES_REL_PATH.split("/"), "train")
PLANT_LEAVES_VAL = os.path.join(DATA_DIR, *PLANT_LEAVES_REL_PATH.split("/"), "valid")


def get_plant_leaves_datasets(
    num_train_batch: int = ALL,
    num_val_batch: int = ALL,
    seed: int = DEFAULT_SEED,
    **from_dir_kwargs,
) -> tuple[tf.data.Dataset, tf.data.Dataset, list[str]]:
    """
    Get the training and validation subsets of the plant leaves dataset.

    SEE: https://www.kaggle.com/datasets/csafrit2/plant-leaves-for-image-classification

    Returns:
        Tuple of training dataset, validation dataset, labels.
            NOTE: for labels, indices correspond with ID, values correspond
            with string labels.
    """
    return get_image_dataset_from_directory(
        train_dir=PLANT_LEAVES_TRAIN,
        val_dir=PLANT_LEAVES_VAL,
        num_train_batch=num_train_batch,
        num_val_batch=num_val_batch,
        seed=seed,
        **from_dir_kwargs,
    )


BIRD_SPECIES_REL_PATH = "bird-species"
BIRD_SPECIES_TRAIN = os.path.join(DATA_DIR, *BIRD_SPECIES_REL_PATH.split("/"), "train")
BIRD_SPECIES_VAL = os.path.join(DATA_DIR, *BIRD_SPECIES_REL_PATH.split("/"), "valid")


def get_bird_species_datasets(
    num_train_batch: int = ALL,
    num_val_batch: int = ALL,
    seed: int = DEFAULT_SEED,
    **from_dir_kwargs,
) -> tuple[tf.data.Dataset, tf.data.Dataset, list[str]]:
    """
    Get the training and validation subsets of the bird species dataset.

    SEE: https://www.kaggle.com/datasets/gpiosenka/100-bird-species

    Returns:
        Tuple of training dataset, validation dataset, labels.
            NOTE: for labels, indices correspond with ID, values correspond
            with string labels.
    """
    return get_image_dataset_from_directory(
        train_dir=BIRD_SPECIES_TRAIN,
        val_dir=BIRD_SPECIES_VAL,
        num_train_batch=num_train_batch,
        num_val_batch=num_val_batch,
        seed=seed,
        **from_dir_kwargs,
    )


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
    image_preprocessor: Callable[[tf.Tensor], tf.Tensor] | None = None,
    label_preprocessor: Callable[[tf.Tensor], tf.Tensor] | None = None,
) -> tf.data.Dataset:
    """
    Preprocess the input dataset for training.

    Args:
        dataset: Dataset to preprocess.
        image_preprocessor: Function to pre-process images.
            Example for VGG16: tf.keras.applications.vgg16.preprocess_input.
        label_preprocessor: Optional function to pre-process labels.

    Returns:
        Preprocessed dataset.
    """

    def _preprocess(x: tf.Tensor, y: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        if image_preprocessor is not None:
            x = image_preprocessor(x)
        if label_preprocessor is not None:
            y = label_preprocessor(y)
        return x, y

    return dataset.map(_preprocess)


def main() -> None:
    """Play around with code here."""
    dataset = tfds.load(name="plant_village", split="train", as_supervised=True)
    labelled_datasets = get_random_datasets(dataset, num_batches=7)
    for ds, labels in labelled_datasets:
        for image, label in ds:
            _ = 0  # Debug here


if __name__ == "__main__":
    main()
