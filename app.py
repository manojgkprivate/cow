import os
import json
from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# Paths
MODEL_PATH = "models/model.h5"
CLASSES_PATH = "models/classes.json"

# Load model
model = load_model(MODEL_PATH)

# Load classes
with open(CLASSES_PATH, "r") as f:
    class_indices = json.load(f)

# Reverse mapping
classes = {v: k for k, v in class_indices.items()}

# Image size must match train.py
IMG_SIZE = 150

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return "No file uploaded"

    file = request.files["image"]
    filepath = "static/upload.jpg"
    file.save(filepath)

    # Load image in training size
    img = load_img(filepath, target_size=(IMG_SIZE, IMG_SIZE))
    img = img_to_array(img) / 255.0
    img = np.expand_dims(img, axis=0)

    # Predict
    preds = model.predict(img)[0]
    class_id = np.argmax(preds)
    confidence = preds[class_id] * 100

    label = classes[class_id]

    return render_template("result.html",
                           label=label,
                           confidence=round(confidence, 2),
                           image_path=filepath)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)

