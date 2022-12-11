import collections
import glob
import math
import os

import numpy as np
import tensorflow as tf
from sklearn.decomposition import PCA

from data import DATA_DIR
from data.dataset import DEFAULT_NUM_CLASSES
from models.core import ChoiceNetv2, TransferModel

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


def embed_dataset_with_model(
    dataset: tf.data.Dataset, model: TransferModel
) -> np.ndarray:
    """Embed the dataset by using the trained TransferModel."""
    # Extract all activations from just before the head of the TransferModel
    class_to_activations: dict[int, list[tf.Tensor]] = collections.defaultdict(list)
    for batch_images, batch_labels in dataset:
        for activation, label in zip(
            model(batch_images, include_top=False), batch_labels
        ):
            class_to_activations[tf.argmax(label).numpy()].append(activation)
    # Rows are classes (in sorted order), cols are mean activation
    class_to_mean_activation = np.stack(
        [
            np.stack(activations).mean(axis=0)
            for _, activations in sorted(class_to_activations.items())
        ],
        axis=0,
    )
    # Flatten to be shape (num_classes, -1)
    return class_to_mean_activation.reshape(class_to_mean_activation.shape[0], -1)


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
    # Flatten images to shape (n, -1), where n = number of examples
    flattened_dataset = dataset.reshape((dataset.shape[0], -1))
    # Use PCA to reduce the dimensionality to shape (n, pca_num_components)
    pca = PCA(n_components=pca_num_components, svd_solver="full")
    reduced_dataset = pca.fit(flattened_dataset).transform(flattened_dataset)
    # Average along examples to return shape (pca_num_components,)
    return reduced_dataset.mean(axis=0)


def embed_dataset_resnet50v2(
    dataset: tf.data.Dataset, num_classes: int = DEFAULT_NUM_CLASSES
) -> np.ndarray:
    """
    Embed the input dataset using ResNet50V2 and keeping the highest activations.

    More specifically:
    1. Pass the dataset through ResNet50V2
    2. Calculate the average activation for each class
    3. Keep num_classes classes's average activations
    """
    images_spec, _ = dataset.element_spec
    embedding_model = tf.keras.Sequential(
        layers=[
            tf.keras.Input(shape=images_spec.shape[1:]),
            tf.keras.layers.Lambda(
                tf.keras.applications.resnet_v2.preprocess_input,
                name="preprocess_images",
            ),
            tf.keras.applications.ResNet50V2(include_top=False),
        ]
    )
    class_to_activations: dict[int, list[tf.Tensor]] = collections.defaultdict(list)
    for batch_images, batch_labels in dataset:
        for activation, label in zip(embedding_model(batch_images), batch_labels):
            class_to_activations[tf.argmax(label).numpy()].append(
                tf.squeeze(activation)
            )
    # Rows are classes (in sorted order), cols are mean activation
    class_to_mean_activation = np.stack(
        [
            np.stack(activations).mean(axis=0)
            for _, activations in sorted(class_to_activations.items())
        ],
        axis=0,
    )
    # Keep the #num_classes classes with the highest activations
    class_to_mean_abs_activation = np.abs(class_to_mean_activation).mean(axis=1)
    classes_to_keep = (
        class_to_mean_abs_activation
        > sorted(class_to_mean_abs_activation, reverse=True)[num_classes]
    )
    highest_class_to_mean_activation = class_to_mean_activation[classes_to_keep]
    assert highest_class_to_mean_activation.shape == (
        num_classes,
        ChoiceNetv2.RESNET50V2_BEFORE_TOP_DIM,
    )
    return highest_class_to_mean_activation


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
