import argparse
import csv
import json
import math
import os

import numpy as np
import tensorflow as tf

from data.dataset import (
    DATASET_CONFIGS,
    DEFAULT_SEED,
    get_dataset_subset,
    get_plant_diseases_datasets,
)
from embedding.embed import embed_dataset, embed_model
from models.core import ChoiceNetSimple, TransferModel
from training import LOG_DIR
from training.create_tlds import DEFAULT_CSV_SUMMARY, TL_MODELS_SAVE_DIR


class TLDSSequence(tf.keras.utils.Sequence):
    """Handy sequence to auto-compute batch size."""

    def __init__(
        self,
        dataset: list[tuple[tuple[np.ndarray, np.ndarray], float]],
        batch_size: int = 32,
    ):
        self.dataset = dataset
        self.batch_size = batch_size

    def __len__(self) -> int:
        return math.ceil(len(self.dataset) / self.batch_size)

    def __getitem__(self, idx: int) -> tuple[tuple[np.ndarray, np.ndarray], np.ndarray]:
        x, y = self.dataset[idx]
        return x, np.array([y])


def train(args: argparse.Namespace) -> None:
    tf.random.set_seed(args.seed)

    tlds: list[tuple[tuple[np.ndarray, np.ndarray], float]] = []
    with open(args.tlds_csv_summary, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            dataset_config = DATASET_CONFIGS[row["dataset"]]
            labels_name = json.loads(row["labels"])
            dataset = get_dataset_subset(
                labels=labels_name,
                dataset=row["dataset"],
                batch_size=int(row["batch_size"]),
                num_batches=int(row["num_batches"]),
            )
            accuracy = float(row["accuracy"])
            model = TransferModel(
                input_shape=dataset_config.image_shape,
                num_classes=dataset_config.num_classes,
            )
            weights_path = os.path.join(
                TL_MODELS_SAVE_DIR,
                row["seed"],
                row["nickname"],
                ",".join(map(str, labels_name)),
            )
            model.load_weights(weights_path).expect_partial()
            tlds.append(
                (
                    (
                        embed_model(model)[np.newaxis, :],
                        embed_dataset(dataset)[np.newaxis, :],
                    ),
                    accuracy,
                )
            )

    model = ChoiceNetSimple()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=args.learning_rate),
        loss="mse",
        metrics=["mse"],
    )
    sequence = TLDSSequence(tlds)  # TODO: make batch_size an arg
    model.fit(sequence)
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
        "--log_dir", type=str, default=LOG_DIR, help="log base directory"
    )
    train(parser.parse_args())


if __name__ == "__main__":
    main()
