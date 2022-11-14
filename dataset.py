import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

# Available datasets: https://www.tensorflow.org/datasets/catalog/overview
CIFAR10 = "cifar10"
CIFAR100 = "cifar100"

DEFAULT_NUM_DATASETS = 8
DEFAULT_NUM_LABELS = 10
DEFAULT_NUM_EXAMPLES = -1


def get_random_datasets(
    dataset: str = CIFAR100,
    num_ds: int = DEFAULT_NUM_DATASETS,
    num_labels: int = DEFAULT_NUM_LABELS,
    num_ex: int = DEFAULT_NUM_EXAMPLES,
    subset: str = "train",
) -> list[tuple[tf.data.Dataset, np.ndarray]]:
    """
    Get list of dataset, label pairings.

    Args:
        dataset: Name of the TensorFlow dataset to load.
        num_ds: Number of datasets to fetch, default is 8.
        num_labels: Number of labels per dataset, default is 10.
        num_ex: Number of examples within a dataset, default fetches all.
        subset: Subset to sample from, either train (default) or test.

    Returns:
        List of tuples of dataset, labels (int).
    """
    full_ds = tfds.load(name=dataset, split=subset, as_supervised=True)
    full_labels = np.fromiter(set(int(label) for _, label in full_ds), dtype=int)
    labels = np.random.choice(full_labels, size=(num_ds, num_labels), replace=False)
    return [
        (
            full_ds.filter(lambda x, y, row=i: tf.reduce_any(y == labels[row])).take(
                num_ex
            ),
            labels[i],
        )
        for i in range(num_ds)
    ]


def main() -> None:
    """Play around with code here."""
    labelled_datasets = get_random_datasets(num_ex=200)
    for ds, labels in labelled_datasets:
        for image, label in ds.prefetch(10):
            _ = 0  # Debug here


if __name__ == "__main__":
    main()
