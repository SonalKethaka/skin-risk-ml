import os, math
import tensorflow as tf
import pandas as pd
from collections import Counter

ROOT = os.path.dirname(os.path.dirname(__file__))
OUT = os.path.join(ROOT, "outputs")
EXPS = os.path.join(ROOT, "exports")
os.makedirs(OUT, exist_ok=True); os.makedirs(EXPS, exist_ok=True)

IMG_SIZE = (224, 224)
BATCH = 32
EPOCHS = 8  # quick first run

def make_ds(csv_path, training=False):
    df = pd.read_csv(csv_path)
    paths = df["image_path"].tolist()
    labels = df["target"].astype("int32").tolist()  # 0/1

    ds = tf.data.Dataset.from_tensor_slices((paths, labels))

    def _load(path, y):
        img = tf.io.read_file(path)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.convert_image_dtype(img, tf.float32)  # [0,1]
        img = tf.image.resize(img, IMG_SIZE)
        if training:
            img = tf.image.random_flip_left_right(img)
            img = tf.image.random_brightness(img, 0.1)
            img = tf.image.random_contrast(img, 0.9, 1.1)
        return img, y

    ds = ds.map(_load, num_parallel_calls=tf.data.AUTOTUNE)
    if training:
        ds = ds.shuffle(2048, reshuffle_each_iteration=True)
    ds = ds.batch(BATCH).prefetch(tf.data.AUTOTUNE)
    return ds, len(df), Counter(labels)

def build_model():
    base = tf.keras.applications.MobileNetV2(
        input_shape=(*IMG_SIZE,3), include_top=False, weights="imagenet"
    )
    base.trainable = False  # freeze for quick baseline
    inputs = tf.keras.Input(shape=(*IMG_SIZE,3))
    x = tf.keras.applications.mobilenet_v2.preprocess_input(inputs*255.0)  # expects [-1,1]
    x = base(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    outputs = tf.keras.layers.Dense(1, activation="sigmoid")(x)
    model = tf.keras.Model(inputs, outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="binary_crossentropy",
        metrics=[tf.keras.metrics.AUC(name="auc"), "accuracy"]
    )
    return model

def class_weights(counter):
    total = sum(counter.values())
    return {0: total/(2.0*counter.get(0,1)), 1: total/(2.0*counter.get(1,1))}

if __name__ == "__main__":
    train_csv = os.path.join(OUT, "train_split.csv")
    val_csv   = os.path.join(OUT, "val_split.csv")

    train_ds, n_train, train_counts = make_ds(train_csv, training=True)
    val_ds,   n_val,   _            = make_ds(val_csv, training=False)

    print("Train class counts:", train_counts)
    cw = class_weights(train_counts)
    print("Using class weights:", cw)

    model = build_model()

    ckpt = tf.keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(EXPS, "checkpoint.keras"),
        monitor="val_auc", mode="max", save_best_only=True
    )
    es = tf.keras.callbacks.EarlyStopping(monitor="val_auc", mode="max", patience=3, restore_best_weights=True)

    steps_per_epoch = math.ceil(n_train / BATCH)
    val_steps = math.ceil(n_val / BATCH)

    model.fit(
        train_ds, validation_data=val_ds,
        epochs=EPOCHS,
        steps_per_epoch=steps_per_epoch, validation_steps=val_steps,
        class_weight=cw,
        callbacks=[ckpt, es]
    )

    saved_dir = os.path.join(EXPS, "saved_model")
    tf.saved_model.save(model, saved_dir)
    print("SavedModel exported to:", saved_dir)