import argparse
import collections
import csv
import json
import math
import os
import random
from collections.abc import Iterable, Mapping
from typing import TypeAlias

import matplotlib.collections
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from data.dataset import (
    DATASET_CONFIGS,
    DEFAULT_BATCH_SIZE,
    DEFAULT_NUM_CLASSES,
    DEFAULT_SEED,
)
from embedding.embed import embed_dataset_pca, embed_model
from models.core import ChoiceNetv1, TransferModel
from training import LOG_DIR, TLDS_DIR
from training.create_tlds import (
    DEFAULT_CSV_SUMMARY,
    FINE_TUNE_DS_SAVE_NAME,
    PLANT_LEAVES_TRAIN_SAVE_DIR,
)

TLDatasetKey: TypeAlias = tuple[str, str, str]
TLDataset: TypeAlias = dict[
    TLDatasetKey, tuple[str, tuple[np.ndarray, np.ndarray], float]
]


def parse_summary(
    summary_path: str = DEFAULT_CSV_SUMMARY, num_classes: int = DEFAULT_NUM_CLASSES
) -> Iterable[tf.keras.Model, str, tuple[str, str], float]:
    """Yield TL model, dataset path, (ds name, ds category), and accuracy tuples."""
    with open(summary_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            dataset_name: str = row["dataset"]
            if i == 0:
                dataset_config = DATASET_CONFIGS[dataset_name]
                model = TransferModel(
                    input_shape=dataset_config.image_shape,
                    num_classes=num_classes,
                )
            try:
                tl_model_folder = ",".join(map(str, json.loads(row["labels"])))
                if dataset_name == "cifar100" or dataset_name.startswith(
                    "imagenet_resized"
                ):
                    key: TLDatasetKey = (dataset_name, "random", row["labels"])
                elif dataset_name == "bird-species":
                    key = (dataset_name, "dissimilar", row["labels"])
                elif dataset_name == "plant_village":
                    key = (dataset_name, "similar", row["labels"])
                else:
                    raise NotImplementedError(f"Unspecified dataset {dataset_name}.")
            except json.decoder.JSONDecodeError:
                tl_model_folder = row["labels"]
                key = (dataset_name, "rand init", row["labels"])
            saved_path = os.path.join(
                TLDS_DIR, row["seed"], dataset_name, tl_model_folder
            )
            tl_weights_path = os.path.join(saved_path, "tl_model")
            tl_dataset_path = os.path.join(saved_path, "tl_dataset")
            model.load_weights(tl_weights_path).expect_partial()
            yield model, tl_dataset_path, key, float(row["accuracy"])


CATEGORY_TO_COLOR: dict[str, str] = {
    "dissimilar": "red",
    "random": "orange",
    "similar": "green",
    "rand init": "grey",
}


def shuffle_mapping(d: Mapping) -> dict:
    """Shuffle the input mapping, not in place."""
    as_tuple = list(d.items())
    random.shuffle(as_tuple)
    return dict(as_tuple)


def build_raw_tlds(
    summary_path: str = DEFAULT_CSV_SUMMARY, num_classes: int = DEFAULT_NUM_CLASSES
) -> TLDataset:
    """Build a raw version of the transfer-learning dataset, with shuffling."""
    plants_ft_ds = tf.data.Dataset.load(
        os.path.join(PLANT_LEAVES_TRAIN_SAVE_DIR, FINE_TUNE_DS_SAVE_NAME)
    )
    embedded_plants_ft_ds = embed_dataset_pca(plants_ft_ds)

    tlds: TLDataset = {}
    for tl_model, _, key, accuracy in parse_summary(summary_path, num_classes):
        # Use of the key ensures that if we have a duplicated TL dataset, that
        # we use the most recently-saved one
        tlds[key] = (key[1], (embed_model(tl_model), embedded_plants_ft_ds), accuracy)
    return shuffle_mapping(tlds)


class TLDSSequence(tf.keras.utils.Sequence):
    """Convert raw transfer learning dataset (TLDS) into trainable form."""

    @classmethod
    def collate_arrays(
        cls, dataset: TLDataset
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Vertically stack embeddings and accuracies into stacked ndarrays."""
        flattened = [
            (x[0][np.newaxis, :], x[1][np.newaxis, :], y)
            for (_, x, y) in dataset.values()
        ]
        x0, x1, y = zip(*flattened)
        return np.vstack(x0), np.vstack(x1), np.array(y)

    def __init__(self, dataset: TLDataset, batch_size: int = 32):
        self.arrays = self.collate_arrays(dataset)
        self.batch_size = batch_size

    def __len__(self) -> int:
        return math.ceil(len(self.arrays[2]) / self.batch_size)

    def __getitem__(self, idx: int) -> tuple[tuple[np.ndarray, np.ndarray], np.ndarray]:
        bslice = slice(idx * self.batch_size, (idx + 1) * self.batch_size)
        return (self.arrays[0][bslice], self.arrays[1][bslice]), self.arrays[2][bslice]

    def get_accuracies(self, idx: int) -> np.ndarray:
        """Get only the dataset's accuracies for the input batch index."""
        bslice = slice(idx * self.batch_size, (idx + 1) * self.batch_size)
        return self.arrays[2][bslice]


MAX_NUM_EPOCHS = 150
EARLY_STOPPING_PATIENCE = 20
LEARNING_RATE = 1e-4


def train_test(args: argparse.Namespace) -> None:
    tf.random.set_seed(args.seed)

    tlds = build_raw_tlds(summary_path=args.tlds_csv_summary)
    num_training_ds = int(len(tlds) * (1 - args.validation_split))
    if num_training_ds >= len(tlds):  # Reuse training ds as test ds
        training_dataseq = TLDSSequence(tlds, batch_size=args.batch_size)
        test_dataseq = training_dataseq
    else:
        training_dataseq = TLDSSequence(
            tlds[:num_training_ds], batch_size=args.batch_size
        )
        test_dataseq = TLDSSequence(tlds[num_training_ds:], batch_size=args.batch_size)

    model = ChoiceNetv1()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=args.learning_rate),
        loss="mse",
        metrics=["mse"],
    )

    model.fit(
        training_dataseq,
        epochs=args.num_epochs,
        callbacks=[
            tf.keras.callbacks.TensorBoard(
                log_dir=os.path.join(LOG_DIR, "choicenet_v1", str(args.seed)),
                histogram_freq=1,
            ),
            tf.keras.callbacks.EarlyStopping(
                monitor="loss",
                patience=EARLY_STOPPING_PATIENCE,
                restore_best_weights=True,
                verbose=1,
            ),
        ],
    )
    preds: np.ndarray = model.predict(test_dataseq)

    all_results: dict[str, list[tuple[float, float]]] = collections.defaultdict(list)
    for i in range(len(test_dataseq)):
        batch_preds = preds[i : i + args.batch_size].squeeze()
        batch_accuracies = test_dataseq.get_accuracies(i)
        for j, (pred, accuracy) in enumerate(zip(batch_preds, batch_accuracies)):
            dataset_category = list(tlds)[i * test_dataseq.batch_size + j][1]
            all_results[dataset_category].append((accuracy, pred))
            print(
                f"Example {i}.{j} with category {dataset_category}: "
                f"predicted accuracy {pred * 100:.3f}%, "
                f"actual accuracy {accuracy * 100:.3f}%."
            )

    fig, ax = plt.subplots()
    scatter_plots: dict[str, matplotlib.collections.Collection] = {
        category: ax.scatter(
            *list(zip(*data)),
            label=f"{category} (x{len(data)})",
            color=CATEGORY_TO_COLOR[category],
        )
        for category, data in all_results.items()
    }
    x_lim, y_lim = ax.get_xlim(), ax.get_ylim()
    ax.plot([0, 1], [0, 1], color="grey", label="unit line")
    ax.set_xlim(x_lim)
    ax.set_ylim(y_lim)
    ax.set_xlabel("Actual Accuracy")
    ax.set_ylabel("Predicted Accuracy")
    ax.grid()
    ax.legend()
    fig.tight_layout()
    fig.savefig("choicenet_v1_performance.png")
    _ = 0  # Debug here


def main() -> None:
    parser = argparse.ArgumentParser(description="Train ChoiceNet v1")
    parser.add_argument(
        "-s", "--seed", type=int, default=DEFAULT_SEED, help="random seed"
    )
    parser.add_argument(
        "--run_nickname",
        type=str,
        default="foo",
        help="nickname for saving logs/models",
    )
    parser.add_argument(
        "--tlds_csv_summary",
        type=str,
        default=DEFAULT_CSV_SUMMARY,
        help="transfer learning dataset summary CSV file location",
    )
    parser.add_argument(
        "--learning_rate", type=float, default=LEARNING_RATE, help="learning rate"
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=MAX_NUM_EPOCHS,
        help="number of epochs for training ChoiceNet",
    )
    parser.add_argument(
        "--validation_split",
        type=float,
        default=0.0,
        help="fraction of transfer learning datasets to use for validation",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="batch size for training and prediction",
    )
    parser.add_argument(
        "--log_dir", type=str, default=LOG_DIR, help="log base directory"
    )
    train_test(parser.parse_args())


if __name__ == "__main__":
    main()
