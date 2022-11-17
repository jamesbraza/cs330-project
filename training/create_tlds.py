import os
from argparse import ArgumentParser

import tensorflow as tf

from data.dataset import (
    DATASET_CONFIGS,
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


def train(args) -> None:
    tf.random.set_seed(args.seed)
    dataset_config = DATASET_CONFIGS[args.dataset]

    seed_nickname = [str(args.seed), args.run_nickname]
    base_models_dir = os.path.join(MODEL_SAVE_DIR, "base_models", *seed_nickname)
    tl_models_dir = os.path.join(MODEL_SAVE_DIR, "tl_models", *seed_nickname)
    ft_models_dir = os.path.join(MODEL_SAVE_DIR, "ft_models", *seed_nickname)
    tl_log_dir = os.path.join(LOG_DIR, "tl_logs", *seed_nickname)
    ft_log_dir = os.path.join(LOG_DIR, "ft_logs", *seed_nickname)

    ft_ds, test_ds, labels = get_plant_diseases_datasets(
        num_train_ex=args.ft_num_examples,
        num_val_ex=args.test_num_examples,
        seed=args.seed,
    )

    model = TransferModel()

    # 1. Save randomly initialized weights to begin training with the same state
    base_weights_path = os.path.join(base_models_dir)
    model.save_weights(base_weights_path)

    for i, (dataset, labels) in enumerate(
        get_random_datasets(
            dataset=dataset_config.name,
            num_ds=args.tl_num_datasets,
            num_classes=args.tl_num_classes,
            num_ex=args.tl_num_examples,
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

        # 3. Prepare datasets for training
        train_ds, val_ds = split(dataset)
        train_ds = (
            preprocess(train_ds, num_classes=dataset_config.num_classes)
            .batch(args.batch_size)
            .prefetch(tf.data.AUTOTUNE)
        )
        val_ds = (
            preprocess(val_ds, num_classes=dataset_config.num_classes)
            .batch(args.batch_size)
            .prefetch(tf.data.AUTOTUNE)
        )

        # 4. Perform the transfer learning
        model.fit(
            train_ds,
            epochs=args.num_epochs,
            validation_data=val_ds,
            callbacks=[
                tf.keras.callbacks.TensorBoard(
                    log_dir=os.path.join(tl_log_dir, labels_name), histogram_freq=1
                )
            ],
        )
        model.save_weights(os.path.join(tl_models_dir, labels_name))

        # 5. Perform the fine tuning
        # Reset the optimizer
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=args.learning_rate),
            loss="categorical_crossentropy",
            metrics=["accuracy"],
        )
        model.fit(
            ft_ds,
            epochs=args.num_epochs,
            callbacks=[
                tf.keras.callbacks.TensorBoard(
                    log_dir=os.path.join(ft_log_dir, labels_name), histogram_freq=1
                )
            ],
        )
        model.save_weights(os.path.join(ft_models_dir, labels_name))

        _ = 0
        model.predict()

    _ = 0


def main() -> None:
    parser = ArgumentParser(description="Create the transfer-learning dataset")
    parser.add_argument(
        "-s", "--seed", type=int, default=DEFAULT_SEED, help="random seed"
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
        "--tl_num_examples",
        type=int,
        default=200,
        help="number of examples to have in each transfer learning dataset",
    )
    parser.add_argument(
        "--ft_num_examples",
        type=int,
        default=200,
        help="number of examples to have in the fine tuning dataset",
    )
    parser.add_argument(
        "--test_num_examples",
        type=int,
        default=200,
        help="number of examples to have in the test dataset",
    )
    parser.add_argument(
        "-b", "--batch_size", type=int, default=32, help="learning rate"
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
