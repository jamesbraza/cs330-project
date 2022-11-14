import os

import tensorflow as tf

from dataset import CIFAR100, get_random_datasets, preprocess_dataset, split
from models import MODEL_SAVE_DIR, VGG_TOP_FC_UNITS, get_model

tf.config.run_functions_eagerly(True)

# Num epochs if not early stopped
MAX_NUM_EPOCHS = 64
# Patience of EarlyStopping callback
ES_PATIENCE_EPOCHS = 8
# Number of images per batch
BATCH_SIZE = 20
# Number of validation set batches to check after each epoch, set None to check
# all validation batches
VALIDATION_STEPS: int | None = None
# Set to a nickname for the save file to help facilitate reuse
SAVE_NICKNAME: str = "bananas"
# Directory to place logs for TensorBoard
LOG_DIR = os.path.dirname(__file__)
# Dataset being used for experiment
DATASET = CIFAR100

for i, (dataset, labels) in enumerate(
    get_random_datasets(dataset=DATASET.name, num_ex=200)
):
    model = get_model(
        top_fc_units=(*VGG_TOP_FC_UNITS[:-1], 100),
        image_shape=DATASET.image_shape,
    )
    model.compile(
        optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
    )
    train_ds, val_ds = split(dataset)
    train_ds = (
        preprocess_dataset(train_ds, num_classes=DATASET.num_classes)
        .batch(BATCH_SIZE)
        .prefetch(tf.data.AUTOTUNE)
    )
    val_ds = (
        preprocess_dataset(val_ds, num_classes=DATASET.num_classes)
        .batch(BATCH_SIZE)
        .prefetch(tf.data.AUTOTUNE)
    )

    # TODO: incorporate ModelCheckpoint callback
    callbacks: list[tf.keras.callbacks.Callback] = [
        tf.keras.callbacks.TensorBoard(log_dir=LOG_DIR, histogram_freq=1),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=ES_PATIENCE_EPOCHS,
            restore_best_weights=True,
            verbose=1,
        ),
    ]

    history: tf.keras.callbacks.History = model.fit(
        train_ds, epochs=MAX_NUM_EPOCHS, validation_data=val_ds, callbacks=callbacks
    )
    model.save(os.path.join(MODEL_SAVE_DIR, f"{SAVE_NICKNAME}{i}"))
    _ = 0  # Debug here
