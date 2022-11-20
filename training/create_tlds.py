import argparse
import math
import os

import numpy as np
import tensorflow as tf

from data.dataset import (
    DATASET_CONFIGS,
    DEFAULT_BATCH_SIZE,
    DEFAULT_NUM_CLASSES,
    DEFAULT_NUM_DATASETS,
    DEFAULT_SEED,
    get_plant_diseases_datasets,
    get_random_datasets,
    preprocess,
    split,
)
from models import MODEL_SAVE_DIR
from models.core import TransferModel
from training import LOG_DIR


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

    model = TransferModel(
        input_shape=dataset_config.image_shape, num_classes=dataset_config.num_classes
    )

    # 1. Save randomly initialized weights to begin training with the same state
    base_weights_path = os.path.join(MODEL_SAVE_DIR, "base_models", *seed_nickname)
    model.save_weights(base_weights_path)

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

        # 2. Load randomly initialized weights
        model.load_weights(base_weights_path)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=args.learning_rate),
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
                        LOG_DIR, "tl_logs", *seed_nickname, labels_name
                    ),
                    histogram_freq=1,
                )
            ],
        )
        model.save_weights(
            os.path.join(MODEL_SAVE_DIR, "tl_models", *seed_nickname, labels_name)
        )

        # 4. Prepare for fine-tuning
        # TODO: utilize copying of weights here
        new_model = model.clone(new_num_classes=len(plants_labels))
        new_model.layer1.trainable = False
        new_model.layer2.trainable = False

        # NOTE: reset the optimizer before fine-tuning
        new_model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=args.learning_rate),
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
                        LOG_DIR, "ft_logs", *seed_nickname, labels_name
                    ),
                    histogram_freq=1,
                )
            ],
        )
        new_model.save_weights(
            os.path.join(MODEL_SAVE_DIR, "ft_models", *seed_nickname, labels_name)
        )

        # 6. Perform predictions on the test dataset
        num_all_correct, count = 0, 0
        for batch_images, batch_labels in test_ds:
            num_all_correct += np.sum(
                new_model.predict(batch_images).argmax(axis=1)
                == tf.argmax(batch_labels, axis=1)
            )
            count += batch_labels.shape[0]
        accuracy = num_all_correct / count
        _ = 0

    _ = 0


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
        default=math.ceil(2000 / DEFAULT_BATCH_SIZE),
        help="number of batches to have in each transfer learning dataset",
    )
    parser.add_argument(
        "--ft_num_batches",
        type=int,
        default=math.ceil(200 / DEFAULT_BATCH_SIZE),
        help="number of batches to have in the fine tuning dataset",
    )
    parser.add_argument(
        "--test_num_batches",
        type=int,
        default=math.ceil(200 / DEFAULT_BATCH_SIZE),
        help="number of batches to have in the test dataset",
    )
    parser.add_argument(
        "--learning_rate", type=float, default=1e-3, help="learning rate"
    )
    parser.add_argument("--num_epochs", type=int, default=15, help="number of epochs")
    parser.add_argument(
        "--run_nickname",
        type=str,
        default="",
        help="nickname for saving logs/models",
    )
    parser.add_argument(
        "--log_dir",
        type=str,
        default=LOG_DIR,
        help="log base directory",
    )
    train(parser.parse_args())


if __name__ == "__main__":
    main()
