import os
import json
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Paths
DATASET_DIR = "dataset"
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "model.h5")
CLASSES_PATH = os.path.join(MODEL_DIR, "classes.json")

# Create models folder if not exists
os.makedirs(MODEL_DIR, exist_ok=True)

# Image settings
IMG_SIZE = 150
BATCH_SIZE = 16

# Data Generator
datagen = ImageDataGenerator(rescale=1.0/255, validation_split=0.2)

train_flow = datagen.flow_from_directory(
    DATASET_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="training"
)

val_flow = datagen.flow_from_directory(
    DATASET_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="validation"
)

# CNN Model
model = Sequential([
    Conv2D(32, (3, 3), activation="relu", input_shape=(IMG_SIZE, IMG_SIZE, 3)),
    MaxPooling2D(),
    Conv2D(64, (3, 3), activation="relu"),
    MaxPooling2D(),
    Conv2D(128, (3, 3), activation="relu"),
    MaxPooling2D(),
    Flatten(),
    Dense(128, activation="relu"),
    Dropout(0.5),
    Dense(train_flow.num_classes, activation="softmax")
])

model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

print("🔄 Training model...")
history = model.fit(
    train_flow,
    validation_data=val_flow,
    epochs=10
)

# Save Model
model.save(MODEL_PATH)
print(f"✅ Model saved at: {MODEL_PATH}")

# Save class names
with open(CLASSES_PATH, "w") as f:
    json.dump(train_flow.class_indices, f)

print(f"✅ Class labels saved at: {CLASSES_PATH}")

