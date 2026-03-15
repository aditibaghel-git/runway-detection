import os
import json
from tensorflow import keras
from config.config import *
from src.data_generator import AnchorDataGenerator
from src.model import build_model

def train():
    with open(COORDS_TRAIN_FILE) as f:
        coords_train = json.load(f)
    with open(COORDS_TEST_FILE) as f:
        coords_test = json.load(f)

    images_train = sorted(os.listdir(IMAGES_TRAIN_DIR))[:TRAIN_COUNT]
    masks_train = sorted(os.listdir(MASKS_TRAIN_DIR))[:TRAIN_COUNT]
    coords_train_keys = sorted(coords_train.keys())[:TRAIN_COUNT]

    images_val = sorted(os.listdir(IMAGES_TEST_DIR))[:TRAIN_COUNT]
    masks_val = sorted(os.listdir(MASKS_TEST_DIR))[:TRAIN_COUNT]
    coords_val_keys = sorted(coords_test.keys())[:TRAIN_COUNT]

    train_gen = AnchorDataGenerator(
        images_train, masks_train, coords_train_keys,
        IMAGES_TRAIN_DIR, MASKS_TRAIN_DIR, coords_train,
        BATCH_SIZE, (TARGET_H, TARGET_W), COORD_VECTOR_SIZE
    )

    val_gen = AnchorDataGenerator(
        images_val, masks_val, coords_val_keys,
        IMAGES_TEST_DIR, MASKS_TEST_DIR, coords_test,
        BATCH_SIZE, (TARGET_H, TARGET_W), COORD_VECTOR_SIZE,
        shuffle=False, is_train=False
    )

    model = build_model(
        (TARGET_H, TARGET_W, 3),
        COORD_VECTOR_SIZE
    )

    model.compile(
        optimizer=keras.optimizers.Adam(LEARNING_RATE),
        loss={
            "mask_output": "binary_crossentropy",
            "anchor_output": "mse"
        },
        loss_weights={
            "mask_output": 1.0,
            "anchor_output": 10.0
        }
    )

    model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=EPOCHS,
        callbacks=[
            keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)
        ]
    )

    model.save("outputs/models/runway_model.h5")

if __name__ == "__main__":
    train()