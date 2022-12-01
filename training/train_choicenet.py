import argparse
import csv
import json
import math
import os

import numpy as np
import tensorflow as tf

from data.dataset import DATASET_CONFIGS, DEFAULT_BATCH_SIZE, DEFAULT_SEED
from embedding.embed import embed_dataset, embed_model
from models.core import ChoiceNetSimple, TransferModel
from training import LOG_DIR, TLDS_DIR
from training.create_tlds import (
    BIRD_SPECIES_TRAIN_SAVE_DIR,
    DEFAULT_CSV_SUMMARY,
    PLANT_DISEASES_TRAIN_SAVE_DIR,
)


class TLDSSequence(tf.keras.utils.Sequence):
    """Convert raw transfer learning dataset (TLDS) into trainable form."""

    @classmethod
    def collate_arrays(
        cls, dataset: list[tuple[tuple[np.ndarray, np.ndarray], float]]
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Vertically stack embeddings and accuracies into stacked ndarrays."""
        flattened = [
            (x[0][np.newaxis, :], x[1][np.newaxis, :], y) for (x, y) in dataset
        ]
        x0, x1, y = zip(*flattened)
        return np.vstack(x0), np.vstack(x1), np.array(y)

    def __init__(
        self,
        dataset: list[tuple[tuple[np.ndarray, np.ndarray], float]],
        batch_size: int = 32,
    ):
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


def build_raw_tlds(
    summary_path: str,
) -> list[tuple[tuple[np.ndarray, np.ndarray], float]]:
    """Build a raw version of the transfer-learning dataset."""
    plants_ft_ds = tf.data.Dataset.load(PLANT_DISEASES_TRAIN_SAVE_DIR)
    birds_ft_ds = tf.data.Dataset.load(BIRD_SPECIES_TRAIN_SAVE_DIR)

    tlds: list[tuple[tuple[np.ndarray, np.ndarray], float]] = []
    with open(summary_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            dataset_config = DATASET_CONFIGS[row["dataset"]]
            model = TransferModel(
                input_shape=dataset_config.image_shape,
                num_classes=dataset_config.num_classes,
            )
            weights_path = os.path.join(
                TLDS_DIR,
                row["seed"],
                ",".join(map(str, json.loads(row["labels"]))),
                "tl_model",
            )
            model.load_weights(weights_path).expect_partial()
            embedded_model = embed_model(model)
            for ds, accuracy in [
                (plants_ft_ds, float(row["plants_accuracy"])),
                (birds_ft_ds, float(row["birds_accuracy"])),
            ]:
                tlds.append(((embedded_model, embed_dataset(ds)), accuracy))
    return tlds


def train(args: argparse.Namespace) -> None:
    tf.random.set_seed(args.seed)

    tlds = build_raw_tlds(summary_path=args.tlds_csv_summary)

    model = ChoiceNetSimple()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=args.learning_rate),
        loss="mse",
        metrics=["mse"],
    )

    num_training_ds = int(len(tlds) * (1 - args.validation_split))
    if num_training_ds >= len(tlds):
        raise ValueError(
            f"Split {args.validation_split} results in an empty test dataset."
        )
    training_dataseq = TLDSSequence(tlds[:num_training_ds], batch_size=args.batch_size)
    test_dataseq = TLDSSequence(tlds[num_training_ds:], batch_size=args.batch_size)

    model.fit(training_dataseq)
    preds: np.ndarray = model.predict(test_dataseq)
    accuracies: list[np.ndarray] = [
        test_dataseq.get_accuracies(i) for i in range(len(test_dataseq))
    ]
    _ = 0  # Debug here


def main() -> None:
    parser = argparse.ArgumentParser(description="Train ChoiceNet")
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
        "--learning_rate", type=float, default=1e-3, help="learning rate"
    )
    parser.add_argument(
        "--validation_split",
        type=float,
        default=0.3,
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
    train(parser.parse_args())


if __name__ == "__main__":
    main()
