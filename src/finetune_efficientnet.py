
from pathlib import Path

import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    ReduceLROnPlateau
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
IMAGE_ROOT = PROJECT_ROOT / "dataset" / "CUB_200_2011" / "images"
PROCESSED_ROOT = PROJECT_ROOT / "dataset" / "processed"
MODEL_ROOT = PROJECT_ROOT / "models"

IMG_SIZE = 224
BATCH_SIZE = 16
AUTOTUNE = tf.data.AUTOTUNE

augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.05),
    tf.keras.layers.RandomZoom(0.10),
    tf.keras.layers.RandomContrast(0.10),
])


def load_image(image_path, label, training=False):
    """Load images in the 0-255 range expected by Keras EfficientNetB0."""
    image = tf.io.read_file(image_path)
    image = tf.io.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [IMG_SIZE, IMG_SIZE])
    if training:
        image = augmentation(image, training=True)
    return tf.clip_by_value(image, 0.0, 255.0), label


def create_dataset(dataframe, class_to_index, training=False):
    """Build datasets with the class map used by the original training run."""
    paths = [str(IMAGE_ROOT / path) for path in dataframe["image_path"]]
    labels = [class_to_index[class_id] for class_id in dataframe["class_id"]]
    dataset = tf.data.Dataset.from_tensor_slices((paths, labels))

    if training:
        dataset = dataset.shuffle(
            len(paths), seed=42, reshuffle_each_iteration=True
        )

    return (
        dataset.map(
            lambda path, label: load_image(path, label, training),
            num_parallel_calls=AUTOTUNE,
        )
        .batch(BATCH_SIZE)
        .prefetch(AUTOTUNE)
    )


train_df = pd.read_csv(PROCESSED_ROOT / "train.csv")
val_df = pd.read_csv(PROCESSED_ROOT / "val.csv")

# Reproduce train_effecientnet.py's zero-based label mapping.
class_ids = sorted(train_df["class_id"].unique())
class_to_index = {class_id: index for index, class_id in enumerate(class_ids)}
train_ds = create_dataset(train_df, class_to_index, training=True)
val_ds = create_dataset(val_df, class_to_index)

# Continue from the validation-best checkpoint.
model = load_model(MODEL_ROOT / "efficientnet_best.keras")

# Find EfficientNet backbone
base_model = None

for layer in model.layers:
    if isinstance(layer, tf.keras.Model):
        base_model = layer
        break

if base_model is None:
    raise ValueError("EfficientNet backbone not found")

# Unfreeze the backbone
base_model.trainable = True

# Freeze all but the last 30 layers
for layer in base_model.layers[:-30]:
    layer.trainable = False

# Keep BatchNormalization frozen
for layer in base_model.layers:
    if isinstance(layer, tf.keras.layers.BatchNormalization):
        layer.trainable = False

print("Trainable layers:",
      sum(layer.trainable for layer in base_model.layers))

# Recompile with a lower learning rate
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

callbacks = [
    ModelCheckpoint(
        MODEL_ROOT / "efficientnet_finetuned_best.keras",
        monitor="val_accuracy",
        save_best_only=True,
        mode="max",
        verbose=1
    ),

    EarlyStopping(
        monitor="val_loss",
        patience=4,
        restore_best_weights=True,
        verbose=1
    ),

    ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.2,
        patience=2,
        min_lr=1e-7,
        verbose=1
    )
]

# Train
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=15,
    callbacks=callbacks
)

# Save final model
model.save(MODEL_ROOT / "efficientnet_finetuned_final.keras")

print("Fine-tuning completed!")
