import os
import random
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.utils.class_weight import compute_class_weight


SEED = 42
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 12

DATA_DIR = "data/images"   
MODEL_OUT = "model/skin_cancer_model_v2.keras"

TRAIN_RATIO = 0.80  
VAL_RATIO   = 0.10   
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)


classes = sorted([d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d))])
print("Classes:", classes)

all_paths = []
all_labels = []

for idx, c in enumerate(classes):
    class_dir = os.path.join(DATA_DIR, c)
    for f in os.listdir(class_dir):
        if f.lower().endswith((".jpg", ".jpeg", ".png")):
            all_paths.append(os.path.join(class_dir, f))
            all_labels.append(idx)

all_paths = np.array(all_paths)
all_labels = np.array(all_labels)


perm = np.random.permutation(len(all_paths))
all_paths = all_paths[perm]
all_labels = all_labels[perm]

n_total = len(all_paths)
n_train = int(n_total * TRAIN_RATIO)

train_paths, test_paths = all_paths[:n_train], all_paths[n_train:]
train_labels, test_labels = all_labels[:n_train], all_labels[n_train:]


n_val = int(len(train_paths) * (VAL_RATIO / TRAIN_RATIO))  
val_paths = train_paths[:n_val]
val_labels = train_labels[:n_val]
train_paths2 = train_paths[n_val:]
train_labels2 = train_labels[n_val:]

print(f"Total: {n_total}")
print(f"Train: {len(train_paths2)} | Val: {len(val_paths)} | Test: {len(test_paths)}")


def load_img(path, label):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, IMG_SIZE)
    img = tf.cast(img, tf.float32) / 255.0
    return img, label

train_ds = tf.data.Dataset.from_tensor_slices((train_paths2, train_labels2)).map(load_img, num_parallel_calls=tf.data.AUTOTUNE)
val_ds   = tf.data.Dataset.from_tensor_slices((val_paths, val_labels)).map(load_img, num_parallel_calls=tf.data.AUTOTUNE)
test_ds  = tf.data.Dataset.from_tensor_slices((test_paths, test_labels)).map(load_img, num_parallel_calls=tf.data.AUTOTUNE)


augment = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
    layers.RandomContrast(0.1),
])

train_ds = train_ds.shuffle(2048, seed=SEED).batch(BATCH_SIZE).map(lambda x, y: (augment(x, training=True), y),
                                                                   num_parallel_calls=tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE)
val_ds   = val_ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
test_ds  = test_ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)


class_weights = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(train_labels2),
    y=train_labels2
)
class_weights = {i: float(w) for i, w in enumerate(class_weights)}
print("Class weights:", class_weights)


base = MobileNetV2(input_shape=(*IMG_SIZE, 3), include_top=False, weights="imagenet")
base.trainable = False

inputs = layers.Input(shape=(*IMG_SIZE, 3))
x = inputs
x = base(x, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.25)(x)
x = layers.Dense(128, activation="relu")(x)
outputs = layers.Dense(len(classes), activation="softmax")(x)

model = models.Model(inputs, outputs)
model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

print(model.summary())


os.makedirs(os.path.dirname(MODEL_OUT), exist_ok=True)

cbs = [
    ModelCheckpoint(MODEL_OUT, monitor="val_accuracy", save_best_only=True, mode="max", verbose=1),
    EarlyStopping(monitor="val_accuracy", patience=4, restore_best_weights=True, mode="max"),
    ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-6, verbose=1)
]


history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    class_weight=class_weights,
    callbacks=cbs
)

print(f"Model saved to: {MODEL_OUT}")


base.trainable = True
for layer in base.layers[:-40]:
    layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-5),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

history_ft = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=8,
    class_weight=class_weights,
    callbacks=cbs
)

print("Fine-tuning done.")
print(f"Total: {n_total}")
print(f"Train: {len(train_paths2)}")
print(f"Val: {len(val_paths)}")
print(f"Test: {len(test_paths)}")

import pickle
with open("model/eval/training_history.pkl", "wb") as f:
    pickle.dump({"phase1": history.history, "phase2": history_ft.history}, f)