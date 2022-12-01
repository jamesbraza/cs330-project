import argparse
import csv
import math
import os

import tensorflow as tf

from data import DATA_DIR
from data.dataset import (
    DATASET_CONFIGS,
    DEFAULT_BATCH_SIZE,
    DEFAULT_NUM_CLASSES,
    DEFAULT_NUM_DATASETS,
    DEFAULT_SEED,
    PLANT_DISEASES_REL_PATH,
    get_plant_diseases_datasets,
    get_random_datasets,
    preprocess,
    split,
)
from models import MODEL_SAVE_DIR
from models.core import TransferModel
from training import LOG_DIR, TRAINING_DIR

DEFAULT_CSV_SUMMARY = os.path.join(TRAINING_DIR, "tlds_summary.csv")
TL_MODELS_SAVE_DIR = os.path.join(MODEL_SAVE_DIR, "tl_models")
FINE_TUNE_DS_SAVE_DIR = os.path.join(
    DATA_DIR, PLANT_DISEASES_REL_PATH.split("/")[0], "ds_export"
)


def preprocess_standardize(
    ds: tf.data.Dataset, num_classes: int, prefetch_size: int = tf.data.AUTOTUNE
) -> tf.data.Dataset:
    """Standardize images, convert labels to one-hot vector, and prefetch."""
    return preprocess(
        ds,
        num_classes=num_classes,
        image_preprocessor=tf.image.per_image_standardization,
    ).prefetch(prefetch_size)


def train(args: argparse.Namespace) -> None:
    tf.random.set_seed(args.seed)
    dataset_config = DATASET_CONFIGS[args.dataset]

    seed_nickname = [str(args.seed), args.run_nickname]
    summary_fields = args.dataset, args.batch_size, args.tl_num_batches, *seed_nickname
    with open(args.tlds_csv_summary, mode="w", encoding="utf-8") as f:
        csv.writer(f).writerow(
            [
                "dataset",
                "batch_size",
                "num_batches",
                "seed",
                "nickname",
                "labels",
                "accuracy",
            ]
        )

    # NOTE: these are already batched
    ft_ds, test_ds, plants_labels = get_plant_diseases_datasets(
        num_train_batch=args.ft_num_batches,
        num_val_batch=args.test_num_batches,
        seed=args.seed,
        batch_size=args.batch_size,
        image_size=dataset_config.image_shape[:-1],
    )
    ft_ds, test_ds = (
        preprocess_standardize(ds, num_classes=len(plants_labels))
        for ds in (ft_ds, test_ds)
    )
    if os.path.exists(FINE_TUNE_DS_SAVE_DIR):
        os.removedirs(FINE_TUNE_DS_SAVE_DIR)  # Only persist one dataset
    # Save this for ChoiceNet training downstream
    ft_ds.save(FINE_TUNE_DS_SAVE_DIR)

    model = TransferModel(
        input_shape=dataset_config.image_shape, num_classes=dataset_config.num_classes
    )
    # Build to populate weights for Checkpoint
    model.build(input_shape=model.input_layer.input_shape[0])

    # 1. Save randomly initialized weights to begin training with the same state
    random_init_path = os.path.join(MODEL_SAVE_DIR, "base_models", *seed_nickname)
    random_init_checkpoint = tf.train.Checkpoint(model)
    if not os.path.exists(f"{random_init_path}-1.index"):
        saved_path = random_init_checkpoint.save(random_init_path)
        assert saved_path == f"{random_init_path}-1"

    for i, (dataset, labels) in enumerate(
        get_random_datasets(
            dataset=dataset_config.name,
            num_ds=args.tl_num_datasets,
            num_classes=args.tl_num_classes,
            batch_size=args.batch_size,
            num_batches=args.tl_num_batches,
            seed=args.seed,
        )
    ):
        labels_name = str(list(labels)).replace(" ", "")
        # Keras can't handle [] per https://github.com/keras-team/keras/issues/17265
        labels_path = labels_name[1:-1]

        # 2. Load randomly initialized weights
        random_init_checkpoint.restore(f"{random_init_path}-1")
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=args.tl_learning_rate),
            loss="categorical_crossentropy",
            metrics=["accuracy"],
        )

        # 3. Perform the transfer learning
        train_ds, val_ds = (
            preprocess_standardize(ds, num_classes=dataset_config.num_classes)
            for ds in split(dataset)
        )
        model.fit(
            train_ds,
            epochs=args.num_epochs,
            validation_data=val_ds,
            callbacks=[
                tf.keras.callbacks.TensorBoard(
                    log_dir=os.path.join(
                        LOG_DIR, "tl_logs", *seed_nickname, labels_path
                    ),
                    histogram_freq=1,
                )
            ],
        )
        # Save the TL model as part of the transfer learning dataset
        model.save_weights(
            os.path.join(TL_MODELS_SAVE_DIR, *seed_nickname, labels_path)
        )

        # 4. Prepare for fine-tuning
        # TODO: utilize copying of weights here
        new_model = model.clone(new_num_classes=len(plants_labels))
        new_model.layer1.trainable = False
        new_model.layer2.trainable = False

        # NOTE: reset the optimizer before fine-tuning
        new_model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=args.ft_learning_rate),
            loss="categorical_crossentropy",
            metrics=["accuracy"],
        )

        # 5. Perform the fine-tuning
        new_model.fit(
            ft_ds,
            epochs=args.num_epochs,
            callbacks=[
                tf.keras.callbacks.TensorBoard(
                    log_dir=os.path.join(
                        LOG_DIR, "ft_logs", *seed_nickname, labels_path
                    ),
                    histogram_freq=1,
                )
            ],
        )

        # 6. Perform predictions on the test dataset
        loss, accuracy = new_model.evaluate(test_ds)
        with open(args.tlds_csv_summary, mode="a", encoding="utf-8") as f:
            csv.writer(f).writerow([*summary_fields, labels_name, accuracy])

    _ = 0  # Debug here


def main() -> None:
    parser = argparse.ArgumentParser(description="Create the transfer-learning dataset")
    parser.add_argument(
        "-s", "--seed", type=int, default=DEFAULT_SEED, help="random seed"
    )
    parser.add_argument(
        "-b",
        "--batch_size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="batch size during transfer learning, fine tuning, and prediction",
    )
    parser.add_argument(
        "-d", "--dataset", type=str, default="cifar100", help="source dataset name"
    )
    parser.add_argument(
        "--tl_num_datasets",
        type=int,
        default=DEFAULT_NUM_DATASETS,
        help="number of transfer learning datasets to sample",
    )
    parser.add_argument(
        "--tl_num_classes",
        type=int,
        default=DEFAULT_NUM_CLASSES,
        help="number of classes to have in each transfer learning dataset",
    )
    parser.add_argument(
        "--tl_num_batches",
        type=int,
        default=math.ceil(3000 / DEFAULT_BATCH_SIZE),
        help="number of batches to have in each transfer learning dataset",
    )
    parser.add_argument(
        "--tl_learning_rate",
        type=float,
        default=1e-3,
        help="learning rate for transfer learning",
    )
    parser.add_argument(
        "--ft_num_batches",
        type=int,
        default=math.ceil(1000 / DEFAULT_BATCH_SIZE),
        help="number of batches to have in the fine tuning dataset",
    )
    parser.add_argument(
        "--ft_learning_rate",
        type=float,
        default=1e-3,
        help="learning rate for fine-tuning",
    )
    parser.add_argument(
        "--test_num_batches",
        type=int,
        default=math.ceil(200 / DEFAULT_BATCH_SIZE),
        help="number of batches to have in the test dataset",
    )
    parser.add_argument("--num_epochs", type=int, default=15, help="number of epochs")
    parser.add_argument(
        "--run_nickname",
        type=str,
        default="foo",
        help="nickname for saving logs/models",
    )
    parser.add_argument(
        "--log_dir", type=str, default=LOG_DIR, help="log base directory"
    )
    parser.add_argument(
        "--tlds_csv_summary",
        type=str,
        default=DEFAULT_CSV_SUMMARY,
        help="transfer learning dataset summary CSV file location",
    )
    train(parser.parse_args())


if __name__ == "__main__":
    main()
