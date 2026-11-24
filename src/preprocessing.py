
import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json

# ---------------------------------
# 1. CONFIGURATION
# ---------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATASET_ROOT = PROJECT_ROOT / "dataset" / "CUB_200_2011"
IMAGE_ROOT = DATASET_ROOT / "images"

PROCESSED_ROOT = PROJECT_ROOT / "dataset" / "processed"

IMG_SIZE = 224
BATCH_SIZE = 16
AUTOTUNE = tf.data.AUTOTUNE

# ---------------------------------
# 2. LOAD CSV FILES
# ---------------------------------

train_df = pd.read_csv(PROCESSED_ROOT / "train.csv")
val_df = pd.read_csv(PROCESSED_ROOT / "val.csv")
test_df = pd.read_csv(PROCESSED_ROOT / "test.csv")

print("Training images:", len(train_df))
print("Validation images:", len(val_df))
print("Testing images:", len(test_df))

# ---------------------------------
# 3. CREATE CLASS LABEL MAPPING
# ---------------------------------

# Original CUB class IDs start from 1.
# Convert selected classes into labels 0 to 19.

class_ids = sorted(train_df["class_id"].unique())

class_names = (
    train_df[["class_id", "class_name"]]
    .drop_duplicates()
    .sort_values("class_id")
)

class_to_index = {
    class_id: index
    for index, class_id in enumerate(class_ids)
}

index_to_class = {
    index: class_names.loc[
        class_names["class_id"] == class_id,
        "class_name"
    ].iloc[0]
    for class_id, index in class_to_index.items()
}

NUM_CLASSES = len(class_to_index)

# Save mapping for future prediction
with open(PROJECT_ROOT / "class_names.json", "w") as f:
    json.dump(index_to_class, f, indent=4)

print("\nNumber of classes:", NUM_CLASSES)
print("Class mapping:", class_to_index)

# ---------------------------------
# 4. IMAGE AUGMENTATION
# ---------------------------------

augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.05),
    tf.keras.layers.RandomZoom(0.10),
    tf.keras.layers.RandomContrast(0.10)
])

# ---------------------------------
# 5. IMAGE LOADING AND PREPROCESSING
# ---------------------------------

def load_image(image_path, label, training=False):

    # Read image from disk
    image = tf.io.read_file(image_path)

    # Decode JPEG and convert to RGB
    image = tf.io.decode_jpeg(image, channels=3)

    # Resize to 224 x 224
    image = tf.image.resize(
        image,
        [IMG_SIZE, IMG_SIZE]
    )

    # Training images only: apply augmentation
    if training:
        image = augmentation(image, training=True)

    # Keep pixel range compatible with
    # Keras EfficientNetB0 (0 to 255).
    image = tf.clip_by_value(image, 0.0, 255.0)

    return image, label


# ---------------------------------
# 6. CREATE TENSORFLOW DATASET
# ---------------------------------

def create_dataset(df, training=False):

    image_paths = [
        str(IMAGE_ROOT / path)
        for path in df["image_path"]
    ]

    labels = [
        class_to_index[class_id]
        for class_id in df["class_id"]
    ]

    dataset = tf.data.Dataset.from_tensor_slices(
        (image_paths, labels)
    )

    if training:
        dataset = dataset.shuffle(
            buffer_size=len(image_paths),
            seed=42,
            reshuffle_each_iteration=True
        )

    dataset = dataset.map(
        lambda path, label: load_image(
            path, label, training
        ),
        num_parallel_calls=AUTOTUNE
    )

    dataset = dataset.batch(BATCH_SIZE)

    dataset = dataset.prefetch(AUTOTUNE)

    return dataset


# ---------------------------------
# 7. CREATE TRAIN, VALIDATION, TEST
# ---------------------------------

train_ds = create_dataset(
    train_df,
    training=True
)

val_ds = create_dataset(
    val_df,
    training=False
)

test_ds = create_dataset(
    test_df,
    training=False
)

# ---------------------------------
# 8. CHECK OUTPUT SHAPES
# ---------------------------------

images, labels = next(iter(train_ds))

print("\nPREPROCESSING RESULTS")
print("Image batch shape:", images.shape)
print("Label batch shape:", labels.shape)
print("Pixel minimum:", tf.reduce_min(images).numpy())
print("Pixel maximum:", tf.reduce_max(images).numpy())

# ---------------------------------
# 9. VISUALIZE PREPROCESSED IMAGES
# ---------------------------------

plt.figure(figsize=(12, 8))

for i in range(min(9, len(images))):

    plt.subplot(3, 3, i + 1)

    # Convert pixel values to 0-1 for display
    plt.imshow(images[i].numpy() / 255.0)

    label = labels[i].numpy()
    plt.title(index_to_class[int(label)])

    plt.axis("off")

plt.tight_layout()
plt.show()

print("\nImage preprocessing completed!")
