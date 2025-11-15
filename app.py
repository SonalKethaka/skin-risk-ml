# skin-risk-ml/app.py
import os
from io import BytesIO
from typing import Tuple

import numpy as np
from PIL import Image
from flask import Flask, request, jsonify
import tensorflow as tf

# Paths
ROOT = os.path.dirname(__file__)
EXPS = os.path.join(ROOT, "exports")
MODEL_PATH = os.path.join(EXPS, "checkpoint.keras")  # you already save this in train.py

# Model / data config (must match train.py)
IMG_SIZE: Tuple[int, int] = (224, 224)
CLASS_NAMES = ["benign (Not Harmfull)", "malignant (Harmfull)"]  # 0 -> benign, 1 -> malignant

# Load model once at startup
print("Loading model from:", MODEL_PATH)
model = tf.keras.models.load_model(MODEL_PATH)
print("Model loaded.")

app = Flask(__name__)


def preprocess_image(file_bytes: bytes) -> np.ndarray:
    """
    Mirror the preprocessing in src/train.py:

    - PIL open
    - resize to (224, 224)
    - convert to float32 in [0, 1]

    The saved model itself already contains the MobileNetV2
    preprocess_input step, so we don't repeat it here.
    """
    img = Image.open(BytesIO(file_bytes)).convert("RGB")
    img = img.resize(IMG_SIZE)
    arr = np.array(img, dtype=np.float32) / 255.0  # [0,1] like train.py
    arr = np.expand_dims(arr, axis=0)  # (1, 224, 224, 3)
    return arr


@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files["file"]

    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    try:
        # Preprocess
        img_arr = preprocess_image(file.read())

        # Predict
        preds = model.predict(img_arr)
        # Your model has Dense(1, activation="sigmoid"), so shape == (1, 1)
        prob = float(preds[0][0])
        # map to class index and confidence
        label_idx = int(prob >= 0.5)
        confidence = prob if prob >= 0.5 else 1.0 - prob

        label = CLASS_NAMES[label_idx]

        return jsonify(
            {
                "label": label,
                "confidence": confidence,
                "details": (
                    "This is an AI-based preliminary assessment based on your "
                    "uploaded photo. Please consult a dermatologist for any "
                    "medical concerns or diagnosis."
                ),
            }
        )
    except Exception as e:
        print("Prediction error:", e)
        return jsonify({"error": "Prediction failed"}), 500


if __name__ == "__main__":
    # Run the Flask app on port 8000
    app.run(host="0.0.0.0", port=8000, debug=True)