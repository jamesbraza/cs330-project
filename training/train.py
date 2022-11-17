import os

import tensorflow as tf

from data.dataset import DATASET_CONFIGS, get_random_datasets, preprocess, split
from models import MODEL_SAVE_DIR
from models.vgg16 import VGG_TOP_FC_UNITS, get_model
from training import LOG_DIR

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
# Dataset being used for experiment
DATASET_CONFIG = DATASET_CONFIGS["cifar100"]

for i, (dataset, labels) in enumerate(
    get_random_datasets(dataset=DATASET_CONFIG.name, num_ex=200)
):
    model = get_model(
        top_fc_units=(*VGG_TOP_FC_UNITS[:-1], 100),
        image_shape=DATASET_CONFIG.image_shape,
    )
    model.compile(
        optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
    )
    train_ds, val_ds = split(dataset)
    train_ds = (
        preprocess(train_ds, num_classes=DATASET_CONFIG.num_classes)
        .batch(BATCH_SIZE)
        .prefetch(tf.data.AUTOTUNE)
    )
    val_ds = (
        preprocess(val_ds, num_classes=DATASET_CONFIG.num_classes)
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
