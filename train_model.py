import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

# ── Config ─────────────────────────────────────
DATASET_PATH = "datasets"
MODEL_PATH   = "cervical_model.keras"
IMG_SIZE     = (224, 224)
BATCH_SIZE   = 16
EPOCHS       = 20
NUM_CLASSES  = 5

# ── Reproducibility ────────────────────────────
tf.random.set_seed(42)

# ── Data Augmentation ──────────────────────────
train_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    rotation_range=20,
    zoom_range=0.2,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    fill_mode="nearest"
)

# Validation — only rescale, no augmentation
val_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

train_data = train_datagen.flow_from_directory(
    DATASET_PATH,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    subset="training",
    class_mode="categorical",
    shuffle=True
)

val_data = val_datagen.flow_from_directory(
    DATASET_PATH,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    subset="validation",
    class_mode="categorical",
    shuffle=False
)

# Print class mapping
print("\n📋 Class indices:", train_data.class_indices)

# ── Base Model ─────────────────────────────────
base_model = MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights="imagenet"
)

# Freeze all base layers initially
base_model.trainable = False

# ── Classification Head ────────────────────────
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.3)(x)                          # prevent overfitting
x = Dense(128, activation="relu")(x)         # extra dense layer
x = Dropout(0.2)(x)
predictions = Dense(NUM_CLASSES, activation="softmax")(x)

model = Model(inputs=base_model.input, outputs=predictions)

# ── Compile ────────────────────────────────────
model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# ── Callbacks ──────────────────────────────────
callbacks = [

    # Save best model automatically
    ModelCheckpoint(
        MODEL_PATH,
        monitor="val_accuracy",
        save_best_only=True,
        verbose=1
    ),

    # Stop early if no improvement for 5 epochs
    EarlyStopping(
        monitor="val_accuracy",
        patience=5,
        restore_best_weights=True,
        verbose=1
    ),

    # Reduce LR when stuck
    ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=3,
        min_lr=1e-6,
        verbose=1
    ),
]

# ── Phase 1: Train head only ───────────────────
print("\n🔵 Phase 1: Training classification head...")

history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=EPOCHS,
    callbacks=callbacks
)

# ── Phase 2: Fine-tune top layers ─────────────
print("\n🟢 Phase 2: Fine-tuning top 30 layers of MobileNetV2...")

base_model.trainable = True

# Freeze all except last 30 layers
for layer in base_model.layers[:-30]:
    layer.trainable = False

# Recompile with lower LR for fine-tuning
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

history_finetune = model.fit(
    train_data,
    validation_data=val_data,
    epochs=10,
    callbacks=callbacks
)

# ── Results ────────────────────────────────────
val_loss, val_acc = model.evaluate(val_data)
print(f"\n✅ Final Validation Accuracy : {val_acc * 100:.2f}%")
print(f"✅ Final Validation Loss     : {val_loss:.4f}")
print(f"✅ Model saved → {MODEL_PATH}")