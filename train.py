import tensorflow as tf
import pandas as pd
import numpy as np
import os

from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

# ==========================================
# 1. Paths
# ==========================================

BASE_DIR = "test_set/test_set"
CSV_PATH = os.path.join(BASE_DIR, "cat_dog.csv")

IMAGE_SIZE = (160, 160)
BATCH_SIZE = 32
EPOCHS = 15

# ==========================================
# 2. Read CSV
# ==========================================

df = pd.read_csv(CSV_PATH)

print("Dataset shape:", df.shape)
print("Columns:", df.columns.tolist())

# ==========================================
# 3. Create full image paths
# ==========================================

def get_image_path(filename):

    if filename.startswith("cat"):
        return os.path.join(BASE_DIR, "cats", filename)

    else:
        return os.path.join(BASE_DIR, "dogs", filename)


df["filepath"] = df["image"].apply(get_image_path)

# Remove missing images
df = df[df["filepath"].apply(os.path.exists)]

print("Images found:", len(df))

# ==========================================
# 4. Labels
# ==========================================

X = df["filepath"].values
y = df["labels"].values

print("Cats:", np.sum(y == 0))
print("Dogs:", np.sum(y == 1))

# ==========================================
# 5. Proper stratified split
# ==========================================

X_train, X_val, y_train, y_val = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Training images:", len(X_train))
print("Validation images:", len(X_val))

# ==========================================
# 6. Image loading function
# ==========================================

def load_image(filepath, label):

    image = tf.io.read_file(filepath)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, IMAGE_SIZE)

    image = tf.cast(image, tf.float32)

    return image, label


# ==========================================
# 7. Create TensorFlow datasets
# ==========================================

train_ds = tf.data.Dataset.from_tensor_slices(
    (X_train, y_train)
)

val_ds = tf.data.Dataset.from_tensor_slices(
    (X_val, y_val)
)

train_ds = train_ds.map(
    load_image,
    num_parallel_calls=tf.data.AUTOTUNE
)

val_ds = val_ds.map(
    load_image,
    num_parallel_calls=tf.data.AUTOTUNE
)

train_ds = train_ds.shuffle(1000).batch(BATCH_SIZE).prefetch(
    tf.data.AUTOTUNE
)

val_ds = val_ds.batch(BATCH_SIZE).prefetch(
    tf.data.AUTOTUNE
)

# ==========================================
# 8. Data augmentation
# ==========================================

data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1)
])

# ==========================================
# 9. MobileNetV2 base model
# ==========================================

base_model = MobileNetV2(
    input_shape=(160, 160, 3),
    include_top=False,
    weights="imagenet"
)

# Freeze pretrained layers
base_model.trainable = False

# ==========================================
# 10. Build model
# ==========================================

inputs = layers.Input(shape=(160, 160, 3))

x = data_augmentation(inputs)

x = tf.keras.applications.mobilenet_v2.preprocess_input(x)

x = base_model(x, training=False)

x = layers.GlobalAveragePooling2D()(x)

x = layers.Dense(128, activation="relu")(x)

x = layers.Dropout(0.4)(x)

outputs = layers.Dense(1, activation="sigmoid")(x)

model = models.Model(inputs, outputs)

# ==========================================
# 11. Compile
# ==========================================

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# ==========================================
# 12. Callbacks
# ==========================================

early_stopping = EarlyStopping(
    monitor="val_accuracy",
    patience=3,
    restore_best_weights=True
)

reduce_lr = ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.5,
    patience=2,
    min_lr=0.000001
)

checkpoint = ModelCheckpoint(
    "cat_dog_best.keras",
    monitor="val_accuracy",
    save_best_only=True
)

# ==========================================
# 13. Train
# ==========================================

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=[
        early_stopping,
        reduce_lr,
        checkpoint
    ]
)

# ==========================================
# 14. Evaluate
# ==========================================

loss, accuracy = model.evaluate(val_ds)

print("\n===================================")
print("TRAINING COMPLETED")
print("===================================")
print("Validation Accuracy:", round(accuracy * 100, 2), "%")
print("Validation Loss:", round(loss, 4))

# ==========================================
# 15. Save model
# ==========================================

model.save("cat_dog_model.keras")

print("\nModel saved:")
print("cat_dog_model.keras")
print("cat_dog_best.keras")
