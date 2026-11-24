
import tensorflow as tf
import pandas as pd
import numpy as np
import json

from pathlib import Path

# ==========================================
# 1. CONFIGURATION
# ==========================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

IMAGE_ROOT = PROJECT_ROOT / "dataset" / "CUB_200_2011" / "images"
PROCESSED_ROOT = PROJECT_ROOT / "dataset" / "processed"
MODEL_ROOT = PROJECT_ROOT / "models"

MODEL_ROOT.mkdir(exist_ok=True)

IMG_SIZE = 224
BATCH_SIZE = 16
EPOCHS = 20
NUM_CLASSES = 20

AUTOTUNE = tf.data.AUTOTUNE

# ==========================================
# 2. CHECK GPU
# ==========================================

print("TensorFlow version:", tf.__version__)
print("Available GPUs:", tf.config.list_physical_devices("GPU"))

# ==========================================
# 3. LOAD DATA
# ==========================================

train_df = pd.read_csv(PROCESSED_ROOT / "train.csv")
val_df = pd.read_csv(PROCESSED_ROOT / "val.csv")

class_ids = sorted(train_df["class_id"].unique())

class_to_index = {
    class_id: index
    for index, class_id in enumerate(class_ids)
}

class_names = (
    train_df[["class_id", "class_name"]]
    .drop_duplicates()
    .sort_values("class_id")
)

index_to_class = {
    index: class_names.loc[
        class_names["class_id"] == class_id,
        "class_name"
    ].iloc[0]
    for class_id, index in class_to_index.items()
}

assert len(class_to_index) == NUM_CLASSES

with open(MODEL_ROOT / "class_names.json", "w") as f:
    json.dump(index_to_class, f, indent=4)

# ==========================================
# 4. DATA AUGMENTATION
# ==========================================

augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.05),
    tf.keras.layers.RandomZoom(0.10),
    tf.keras.layers.RandomContrast(0.10)
])

# ==========================================
# 5. IMAGE LOADING
# ==========================================

def load_image(image_path, label, training=False):

    image = tf.io.read_file(image_path)

    image = tf.io.decode_jpeg(
        image,
        channels=3
    )

    image = tf.image.resize(
        image,
        [IMG_SIZE, IMG_SIZE]
    )

    if training:
        image = augmentation(image, training=True)

    image = tf.clip_by_value(
        image, 0.0, 255.0
    )

    return image, label


# ==========================================
# 6. CREATE TF DATASETS
# ==========================================

def create_dataset(df, training=False):

    paths = [
        str(IMAGE_ROOT / p)
        for p in df["image_path"]
    ]

    labels = [
        class_to_index[c]
        for c in df["class_id"]
    ]

    ds = tf.data.Dataset.from_tensor_slices(
        (paths, labels)
    )

    if training:
        ds = ds.shuffle(
            len(paths),
            seed=42,
            reshuffle_each_iteration=True
        )

    ds = ds.map(
        lambda p, l: load_image(p, l, training),
        num_parallel_calls=AUTOTUNE
    )

    ds = ds.batch(BATCH_SIZE)
    ds = ds.prefetch(AUTOTUNE)

    return ds


train_ds = create_dataset(
    train_df,
    training=True
)

val_ds = create_dataset(
    val_df,
    training=False
)

# ==========================================
# 7. LOAD PRETRAINED EFFICIENTNETB0
# ==========================================

base_model = tf.keras.applications.EfficientNetB0(
    weights="imagenet",
    include_top=False,
    input_shape=(IMG_SIZE, IMG_SIZE, 3)
)

# Freeze all pretrained layers initially
base_model.trainable = False

# ==========================================
# 8. BUILD CLASSIFICATION HEAD
# ==========================================

inputs = tf.keras.Input(
    shape=(IMG_SIZE, IMG_SIZE, 3)
)

# Extract pretrained features
x = base_model(
    inputs,
    training=False
)

# Convert feature maps into a vector
x = tf.keras.layers.GlobalAveragePooling2D()(x)

# Reduce overfitting
x = tf.keras.layers.Dropout(0.3)(x)

# Predict 20 bird species
outputs = tf.keras.layers.Dense(
    NUM_CLASSES,
    activation="softmax"
)(x)

model = tf.keras.Model(
    inputs,
    outputs
)

# ==========================================
# 9. COMPILE MODEL
# ==========================================

model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.001
    ),

    loss="sparse_categorical_crossentropy",

    metrics=["accuracy"]
)

print("\nMODEL SUMMARY")
model.summary()

# ==========================================
# 10. CALLBACKS
# ==========================================

callbacks = [

    tf.keras.callbacks.ModelCheckpoint(
        filepath=str(MODEL_ROOT / "efficientnet_best.keras"),
        monitor="val_accuracy",
        save_best_only=True,
        mode="max",
        verbose=1
    ),

    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True,
        verbose=1
    ),

    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.2,
        patience=2,
        min_lr=1e-7,
        verbose=1
    ),

    tf.keras.callbacks.CSVLogger(
        str(MODEL_ROOT / "efficientnet_history.csv")
    )
]

# ==========================================
# 11. TRAIN MODEL
# ==========================================

print("\nStarting EfficientNetB0 training...")

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=callbacks
)

# ==========================================
# 12. SAVE FINAL MODEL
# ==========================================

model.save(
    MODEL_ROOT / "efficientnet_final.keras"
)

print("\nTraining completed!")
print("Best model: models/efficientnet_best.keras")
print("Final model: models/efficientnet_final.keras")
print("Training history: models/efficientnet_history.csv")
