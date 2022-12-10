import glob
import math
import os

import numpy as np
import tensorflow as tf
from sklearn.decomposition import PCA

from data import DATA_DIR
from models.core import TransferModel

EMBEDDED_MODEL_DIMS = (
    TransferModel.LAYER_3_NUM_FILTERS,
    TransferModel.LAYER_2_NUM_FILTERS * math.prod(TransferModel.KERNEL_SIZE),
)


def embed_model(model: TransferModel) -> np.ndarray:
    """Extract and flatten the weights from the 2nd Conv2D layer."""
    try:
        # Discard biases
        weights: np.ndarray = model.layer3.conv.get_weights()[0]
    except IndexError:
        # Build the model so weights are populated
        model.build(input_shape=model.input_layer.input_shape[0])
        weights = model.layer3.conv.get_weights()[0]
    # Flatten to (# filters, -1)
    return weights.reshape((-1, weights.shape[-1])).transpose()


DEFAULT_PCA_NUM_COMPONENTS = 256


def embed_dataset_pca(
    dataset: np.ndarray | tf.data.Dataset,
    pca_num_components: int = DEFAULT_PCA_NUM_COMPONENTS,
) -> np.ndarray:
    """Reduce the dimensionality of the dataset using PCA and averaging."""
    if isinstance(dataset, tf.data.Dataset):
        if len(dataset.element_spec) > 2 or len(dataset.element_spec) <= 0:
            raise NotImplementedError("Unexpected dataset shape.")
        elif len(dataset.element_spec) == 1:
            dataset: np.ndarray = np.vstack(list(dataset))
        else:
            dataset = np.vstack([image for image, _ in dataset])
    # Flatten images
    flattened_dataset = dataset.reshape((dataset.shape[0], -1))
    # Use PCA to reduce the dimensionality
    pca = PCA(n_components=pca_num_components, svd_solver="full")
    reduced_dataset = pca.fit(flattened_dataset).transform(flattened_dataset)
    # Average along examples
    return reduced_dataset.mean(axis=0)


def get_npy_paths(path: str) -> list[str]:
    """Get a list of paths to files with the npy extension."""
    return glob.glob(os.path.join(path, "*.npy"))


def pca_dataset(path: str) -> None:
    all_data_list = []
    for data_path in get_npy_paths(path):
        all_data_list.append(embed_dataset_pca(dataset=np.load(data_path)))
    np.save(os.path.join(path, "data.npy"), np.array(all_data_list))


def pca_single_dataset(path: str, num_repeats: int = 1) -> np.ndarray:
    x_average = embed_dataset_pca(dataset=np.load(path))
    return np.repeat(x_average[np.newaxis, :], repeats=num_repeats, axis=0)


if __name__ == "__main__":
    all_data = pca_single_dataset(
        path=os.path.join(DATA_DIR, "X_test.npy"), num_repeats=2
    )
    np.save(os.path.join(DATA_DIR, "x_test_new.npy"), all_data)
