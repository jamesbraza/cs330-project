import argparse
import csv
import math
import os
import shutil

import tensorflow as tf

from data import DATA_DIR
from data.dataset import (
    BIRD_SPECIES_REL_PATH,
    DATASET_CONFIGS,
    DEFAULT_BATCH_SIZE,
    DEFAULT_NUM_CLASSES,
    DEFAULT_NUM_DATASETS,
    DEFAULT_SEED,
    PLANT_DISEASES_REL_PATH,
    get_bird_species_datasets,
    get_plant_diseases_datasets,
    get_random_datasets,
    preprocess,
    split,
)
from models import MODEL_SAVE_DIR
from models.core import TransferModel
from training import LOG_DIR, TLDS_DIR, TRAINING_DIR

DEFAULT_CSV_SUMMARY = os.path.join(TRAINING_DIR, "tlds_summary.csv")
PLANT_DISEASES_TRAIN_SAVE_DIR = os.path.join(
    DATA_DIR, *PLANT_DISEASES_REL_PATH.split("/"), "train_ds_export"
)
BIRD_SPECIES_TRAIN_SAVE_DIR = os.path.join(
    DATA_DIR, *BIRD_SPECIES_REL_PATH.split("/"), "train_ds_export"
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


def preprocess_ds_save(
    ft_ds: tf.data.Dataset,
    test_ds: tf.data.Dataset,
    num_classes: int,
    ft_ds_save_dir: str,
) -> tuple[tf.data.Dataset, tf.data.Dataset]:
    """Preprocess both fine-tuning and test datasets, saving the fine-tuning dataset."""
    ft_ds, test_ds = (
        preprocess_standardize(ds, num_classes=num_classes) for ds in (ft_ds, test_ds)
    )
    if os.path.exists(ft_ds_save_dir):
        shutil.rmtree(ft_ds_save_dir)  # Only persist one dataset
    ft_ds.save(ft_ds_save_dir)
    return ft_ds, test_ds


def train(args: argparse.Namespace) -> None:
    tf.random.set_seed(args.seed)

    with open(args.tlds_csv_summary, mode="w", encoding="utf-8") as f:
        csv.writer(f).writerow(
            ["dataset", "seed", "labels", "plants_accuracy", "birds_accuracy"]
        )

    dataset_config = DATASET_CONFIGS[args.dataset]
    plants_ft_ds, plants_test_ds, plants_labels = get_plant_diseases_datasets(
        num_train_batch=args.ft_num_batches,
        num_val_batch=args.test_num_batches,
        seed=args.seed,
        batch_size=args.batch_size,
        image_size=dataset_config.image_shape[:-1],
    )
    plants_ft_ds, plants_test_ds = preprocess_ds_save(
        plants_ft_ds,
        plants_test_ds,
        num_classes=len(plants_labels),
        ft_ds_save_dir=PLANT_DISEASES_TRAIN_SAVE_DIR,
    )
    birds_ft_ds, birds_test_ds, birds_labels = get_bird_species_datasets(
        num_train_batch=args.ft_num_batches,
        num_val_batch=args.test_num_batches,
        seed=args.seed,
        batch_size=args.batch_size,
        image_size=dataset_config.image_shape[:-1],
    )
    birds_ft_ds, birds_test_ds = preprocess_ds_save(
        birds_ft_ds,
        birds_test_ds,
        num_classes=len(birds_labels),
        ft_ds_save_dir=BIRD_SPECIES_TRAIN_SAVE_DIR,
    )

    def compute_accuracy(tl_model: tf.keras.Model) -> list[float]:
        accuracies: list[float] = []
        for (ft_ds, test_ds, num_classes) in [
            (plants_ft_ds, plants_test_ds, len(plants_labels)),
            (birds_ft_ds, birds_test_ds, len(birds_labels)),
        ]:
            new_model = tl_model.clone(new_num_classes=num_classes)
            new_model.layer1.trainable = False
            new_model.layer2.trainable = False

            # NOTE: reset the optimizer before fine-tuning
            new_model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=args.ft_learning_rate),
                loss="categorical_crossentropy",
                metrics=["accuracy"],
            )

            new_model.fit(ft_ds, epochs=args.num_epochs)
            _, accuracy = new_model.evaluate(test_ds)
            accuracies.append(accuracy)
        return accuracies

    model = TransferModel(
        input_shape=dataset_config.image_shape, num_classes=dataset_config.num_classes
    )
    # Build to populate weights for Checkpoint
    model.build(input_shape=model.input_layer.input_shape[0])

    # Save results for randomly initialized model
    model.save_weights(os.path.join(TLDS_DIR, str(args.seed), "randinit", "tl_model"))
    with open(args.tlds_csv_summary, mode="a", encoding="utf-8") as f:
        csv.writer(f).writerow(
            [args.dataset, args.seed, "randinit"] + compute_accuracy(model)
        )

    # 1. Save randomly initialized weights to begin training with the same state
    random_init_path = os.path.join(MODEL_SAVE_DIR, "base_models", str(args.seed))
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
        base_tl_path = os.path.join(TLDS_DIR, str(args.seed), labels_path)

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
        train_ds.save(os.path.join(base_tl_path, "tl_dataset"))
        model.fit(
            train_ds,
            epochs=args.num_epochs,
            validation_data=val_ds,
            callbacks=[
                tf.keras.callbacks.TensorBoard(
                    log_dir=os.path.join(
                        LOG_DIR, "tl_logs", str(args.seed), labels_path
                    ),
                    histogram_freq=1,
                )
            ],
        )
        model.save_weights(os.path.join(base_tl_path, "tl_model"))

        # 4. Perform fine-tuning, testing, and save the accuracy
        with open(args.tlds_csv_summary, mode="a", encoding="utf-8") as f:
            csv.writer(f).writerow(
                [args.dataset, args.seed, labels_name] + compute_accuracy(model)
            )

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
