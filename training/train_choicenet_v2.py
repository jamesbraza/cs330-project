import argparse
import collections
import os

import matplotlib.collections
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from data.dataset import DEFAULT_BATCH_SIZE, DEFAULT_NUM_CLASSES, DEFAULT_SEED
from embedding.embed import embed_dataset_resnet50v2, embed_dataset_with_model
from models.core import ChoiceNetv2
from training import LOG_DIR
from training.create_tlds import DEFAULT_CSV_SUMMARY, PLANT_LEAVES_TRAIN_SAVE_DIR
from training.train_choicenet_v1 import (
    LABEL_TO_COLOR,
    TLDataset,
    TLDSSequence,
    parse_summary,
)


def build_raw_tlds(
    summary_path: str = DEFAULT_CSV_SUMMARY, num_classes: int = DEFAULT_NUM_CLASSES
) -> TLDataset:
    """Build a raw version of the transfer-learning dataset."""
    plants_ft_ds = tf.data.Dataset.load(PLANT_LEAVES_TRAIN_SAVE_DIR)
    embedded_plants_ft_ds = embed_dataset_resnet50v2(plants_ft_ds)

    tlds: TLDataset = []
    for tl_model, tl_dataset_path, ds_nickname, accuracy in parse_summary(
        summary_path, num_classes
    ):
        if ds_nickname == "rand init":
            continue  # Skip over rand init, as it has no associated TL dataset
        embedded_model = embed_dataset_with_model(
            dataset=tf.data.Dataset.load(tl_dataset_path), model=tl_model
        )
        tlds.append((ds_nickname, (embedded_model, embedded_plants_ft_ds), accuracy))
    return tlds


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

    model = ChoiceNetv2()
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
                log_dir=os.path.join(LOG_DIR, "choicenet", str(args.seed)),
                histogram_freq=1,
            )
        ],
    )
    preds: np.ndarray = model.predict(test_dataseq)

    all_results: dict[str, list[tuple[float, float]]] = collections.defaultdict(list)
    for i in range(len(test_dataseq)):
        batch_preds = preds[i : i + args.batch_size].squeeze()
        batch_accuracies = test_dataseq.get_accuracies(i)
        for j, (pred, accuracy) in enumerate(zip(batch_preds, batch_accuracies)):
            dataset_nickname = tlds[i * test_dataseq.batch_size + j][0]
            all_results[dataset_nickname].append((accuracy, pred))
            print(
                f"Example {i}.{j} with nickname {dataset_nickname}: "
                f"predicted accuracy {pred * 100:.3f}%, "
                f"actual accuracy {accuracy * 100:.3f}%."
            )

    fig, ax = plt.subplots()
    scatter_plots: dict[str, matplotlib.collections.Collection] = {
        label: ax.scatter(
            *list(zip(*data)),
            label=f"{label} (x{len(data)})",
            color=LABEL_TO_COLOR[label],
        )
        for label, data in all_results.items()
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
    fig.savefig("choicenet_v2_performance.png")
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
        "--learning_rate", type=float, default=1e-3, help="learning rate"
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=15,
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
