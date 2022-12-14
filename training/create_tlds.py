import argparse
import csv
import math
import os
import shutil
from collections.abc import Callable, Sequence
from functools import partial

import tensorflow as tf
import tensorflow_datasets as tfds

from data import DATA_DIR
from data.dataset import (
    BIRD_SPECIES_REL_PATH,
    DATASET_CONFIGS,
    DEFAULT_BATCH_SIZE,
    DEFAULT_NUM_CLASSES,
    DEFAULT_SEED,
    PLANT_DISEASES_REL_PATH,
    PLANT_LEAVES_REL_PATH,
    Shape,
    get_bird_species_datasets,
    get_plant_leaves_datasets,
    get_random_datasets,
)
from models import MODEL_SAVE_DIR
from models.core import TransferModel
from training import LOG_DIR, TLDS_DIR, TRAINING_DIR

DEFAULT_CSV_SUMMARY = os.path.join(TRAINING_DIR, "tlds_summary.csv")
PLANT_DISEASES_TRAIN_SAVE_DIR = os.path.join(
    DATA_DIR, *PLANT_DISEASES_REL_PATH.split("/"), "train_ds_export"
)
PLANT_LEAVES_TRAIN_SAVE_DIR = os.path.join(
    DATA_DIR, *PLANT_LEAVES_REL_PATH.split("/"), "train_ds_export"
)
BIRD_SPECIES_TRAIN_SAVE_DIR = os.path.join(
    DATA_DIR, *BIRD_SPECIES_REL_PATH.split("/"), "train_ds_export"
)
FINE_TUNE_DS_SAVE_NAME = "ft"
TEST_DS_SAVE_NAME = "test"


def preprocess_standardize(
    ds: tf.data.Dataset,
    num_classes: int,
    labels: Sequence[int] | None = None,
    image_size: Shape | None = None,
    prefetch_size: int = tf.data.AUTOTUNE,
) -> tf.data.Dataset:
    """Standardize images, convert labels to one-hot vector, and prefetch."""

    def image_preprocessor(image: tf.Tensor) -> tf.Tensor:
        if image_size is not None and image.shape != image_size:
            if len(image_size) == 4:
                _, *hw, _ = image_size
            elif len(image_size) == 3:
                *hw, _ = image_size
            else:
                raise NotImplementedError(f"Unimplemented shape {image_size}.")
            image = tf.image.resize(image, size=hw)
        return tf.image.per_image_standardization(image)

    label_preprocessor = partial(tf.one_hot, depth=num_classes)

    if labels is None:  # Don't remap labels
        labels_preprocessor = label_preprocessor
    else:  # Remap labels
        assert len(labels) == num_classes
        labels_tensor = tf.constant(labels, dtype=tf.int32)
        mapped_labels_tensor = tf.constant(list(range(len(labels))), dtype=tf.int32)
        labels_map = tf.lookup.StaticHashTable(
            tf.lookup.KeyValueTensorInitializer(labels_tensor, mapped_labels_tensor),
            default_value=-1,
        )

        def label_remapping_preprocessor(label: tf.Tensor) -> tf.Tensor:
            """Remap labels to be in the range of the number of labels."""
            return label_preprocessor(labels_map.lookup(tf.cast(label, tf.int32)))

        labels_preprocessor = label_remapping_preprocessor

    return ds.map(
        lambda x, y: (image_preprocessor(x), labels_preprocessor(y))
    ).prefetch(prefetch_size)


def preprocess_ds_save(
    ft_ds: tf.data.Dataset,
    test_ds: tf.data.Dataset,
    preprocessor: Callable[[tf.data.Dataset], tf.data.Dataset],
    save_dir: str,
) -> tuple[tf.data.Dataset, tf.data.Dataset]:
    """Preprocess and save both fine-tuning and test datasets."""
    ft_ds, test_ds = (preprocessor(ds) for ds in (ft_ds, test_ds))
    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)  # Only persist one copy
    ft_ds.save(os.path.join(save_dir, FINE_TUNE_DS_SAVE_NAME))
    test_ds.save(os.path.join(save_dir, TEST_DS_SAVE_NAME))
    return ft_ds, test_ds


def train(args: argparse.Namespace) -> None:
    tf.random.set_seed(args.seed)

    with open(args.tlds_csv_summary, mode="w", encoding="utf-8") as f:
        csv.writer(f).writerow(["dataset", "seed", "labels", "accuracy"])

    dataset_config = DATASET_CONFIGS["imagenet_resized/32x32"]

    if not os.path.exists(PLANT_LEAVES_TRAIN_SAVE_DIR):
        plants_ft_ds, plants_test_ds, plant_labels = get_plant_leaves_datasets(
            num_train_batch=args.ft_num_batches,
            num_val_batch=args.test_num_batches,
            seed=args.seed,
            batch_size=args.batch_size,
            image_size=dataset_config.image_shape[:-1],
        )
        plants_ft_ds, plants_test_ds = preprocess_ds_save(
            plants_ft_ds,
            plants_test_ds,
            preprocessor=partial(preprocess_standardize, num_classes=len(plant_labels)),
            save_dir=PLANT_LEAVES_TRAIN_SAVE_DIR,
        )
    # Reload in fine-tuning and test datasets as a speed optimization
    plants_ft_ds = tf.data.Dataset.load(
        os.path.join(PLANT_LEAVES_TRAIN_SAVE_DIR, FINE_TUNE_DS_SAVE_NAME)
    )
    plants_test_ds = tf.data.Dataset.load(
        os.path.join(PLANT_LEAVES_TRAIN_SAVE_DIR, TEST_DS_SAVE_NAME)
    )
    num_ft_classes = DATASET_CONFIGS["plant-leaves"].num_classes

    def compute_accuracy(tl_model: TransferModel) -> float:
        new_model = tl_model.clone(new_num_classes=num_ft_classes)
        new_model.layer1.trainable = False
        new_model.layer2.trainable = False

        # NOTE: reset the optimizer before fine-tuning
        new_model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=args.ft_learning_rate),
            loss="categorical_crossentropy",
            metrics=["accuracy"],
        )

        new_model.fit(plants_ft_ds, epochs=args.num_epochs)
        _, accuracy = new_model.evaluate(plants_test_ds)
        return accuracy

    model = TransferModel(
        input_shape=dataset_config.image_shape,
        num_classes=args.tl_num_classes,
    )
    # Build to populate weights for Checkpoint
    model.build(input_shape=model.input_layer.input_shape[0])

    # Save results for randomly initialized model
    model.save_weights(
        os.path.join(TLDS_DIR, str(args.seed), "cifar100", "randinit", "tl_model")
    )
    with open(args.tlds_csv_summary, mode="a", encoding="utf-8") as f:
        csv.writer(f).writerow(
            ["cifar100", args.seed, "randinit", compute_accuracy(model)]
        )

    # 1. Save randomly initialized weights to begin training with the same state
    random_init_path = os.path.join(MODEL_SAVE_DIR, "base_models", str(args.seed))
    random_init_checkpoint = tf.train.Checkpoint(model)
    if not os.path.exists(f"{random_init_path}-1.index"):
        saved_path = random_init_checkpoint.save(random_init_path)
        assert saved_path == f"{random_init_path}-1"

    kwargs = {
        "num_classes": args.tl_num_classes,
        "batch_size": args.batch_size,
        "num_batches": args.tl_num_batches,
        "seed": args.seed,
    }
    cifar100_random_datasets = get_random_datasets(
        dataset=tfds.load(name="cifar100", split="train", as_supervised=True),
        num_ds=args.tl_num_random_cifar100_datasets,
        **kwargs,
    )
    imagenet_random_datasets = get_random_datasets(
        dataset=tfds.load(
            name="imagenet_resized/32x32", split="train", as_supervised=True
        ),
        num_ds=args.tl_num_random_imagenet_datasets,
        **kwargs,
    )
    plants_village_datasets = get_random_datasets(
        dataset=tfds.load(name="plant_village", split="train", as_supervised=True),
        num_ds=args.tl_num_similar_datasets,
        **kwargs,
    )
    birds_random_datasets = get_random_datasets(
        dataset=get_bird_species_datasets(
            seed=args.seed, batch_size=None, image_size=dataset_config.image_shape[:-1]
        )[0],
        num_ds=args.tl_num_dissimilar_datasets,
        **kwargs,
    )
    for i, (dataset_name, dataset, labels) in enumerate(
        [("cifar100", *v) for v in cifar100_random_datasets]
        + [("plant_village", *v) for v in plants_village_datasets]
        + [("bird-species", *v) for v in birds_random_datasets]
        + [("imagenet_resized/32x32", *v) for v in imagenet_random_datasets]
    ):
        labels_name = str(list(labels)).replace(" ", "")
        # Keras can't handle [] per https://github.com/keras-team/keras/issues/17265
        labels_path = labels_name[1:-1]
        base_tl_path = os.path.join(TLDS_DIR, str(args.seed), dataset_name, labels_path)

        # 2. Load randomly initialized weights
        random_init_checkpoint.restore(f"{random_init_path}-1")
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=args.tl_learning_rate),
            loss="categorical_crossentropy",
            metrics=["accuracy"],
        )

        # 3. Perform the transfer learning
        dataset_preprocessor = partial(
            preprocess_standardize,
            num_classes=model.dense.units,
            labels=labels,
            image_size=(None, *dataset_config.image_shape),
        )
        train_ds = dataset_preprocessor(dataset)
        train_ds.save(os.path.join(base_tl_path, "tl_dataset"))
        model.fit(
            train_ds,
            epochs=args.num_epochs,
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
                [dataset_name, args.seed, labels_name, compute_accuracy(model)]
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
        "--tl_num_random_cifar100_datasets",
        type=int,
        default=10,
        help="number of random transfer learning datasets to sample from CIFAR100",
    )
    parser.add_argument(
        "--tl_num_random_imagenet_datasets",
        type=int,
        default=30,
        help="number of random transfer learning datasets to sample from ImageNet",
    )
    parser.add_argument(
        "--tl_num_similar_datasets",
        type=int,
        default=3,
        help="number of similar transfer learning datasets to sample",
    )
    parser.add_argument(
        "--tl_num_dissimilar_datasets",
        type=int,
        default=3,
        help="number of dissimilar transfer learning datasets to sample",
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
        default=math.ceil(15e3 / DEFAULT_BATCH_SIZE),
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
        default=math.ceil(1.5e3 / DEFAULT_BATCH_SIZE),  # Keep small to emphasize TL
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
        default=math.ceil(2e3 / DEFAULT_BATCH_SIZE),
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
