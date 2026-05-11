import os
import io
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from flask import Flask, request, jsonify, render_template
from google.cloud import storage
import base64
from PIL import Image
from google.cloud import bigquery
from datetime import datetime
import hashlib

app = Flask(__name__)

# ── Config ──────────────────────────────────────────────────────────────────
BUCKET_NAME = "deepfake-dataset-26"
MODEL_BLOB = "deepshield_final.keras"

import platform
if platform.system() == "Windows":
    MODEL_PATH = "deepshield_final.keras"
else:
    MODEL_PATH = "/tmp/deepshield_final.keras"
IMG_SIZE = 224

model = None   # loaded once on first request

# ── GCP credentials from environment variable ────────────────────────────────
def setup_credentials():
    creds_json = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS_JSON")
    if creds_json:
        creds_path = "/tmp/gcp_creds.json"
        with open(creds_path, "w") as f:
            f.write(creds_json)
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = creds_path
    else:
         # Local development fallback
        local_creds = r"C:\Users\HP\Downloads\deepshield-493817-ca210739a648.json"
        if os.path.exists(local_creds):
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = local_creds
        else:
            print("Using default environment credentials")
# ── Load model from GCS ──────────────────────────────────────────────────────
bq_client = None

def get_bq_client():
    global bq_client

    if bq_client is None:
        setup_credentials()
        bq_client = bigquery.Client()

    return bq_client

def load_model():
    global model
    if model is not None:
        return model

    setup_credentials()

    if not os.path.exists(MODEL_PATH):
        print("Downloading model from GCS...")
        client = storage.Client()
        bucket = client.bucket(BUCKET_NAME)
        blob = bucket.blob(MODEL_BLOB)
        blob.download_to_filename(MODEL_PATH)
        print("Model downloaded!")

    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    print("Model loaded successfully!")
    return model
# ── Preprocess image ─────────────────────────────────────────────────────────
def preprocess_image(image_bytes):
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize((IMG_SIZE, IMG_SIZE))

    img_array = np.array(img)
    img_array = preprocess_input(img_array)

    img_array = np.expand_dims(img_array, axis=0)

    return img_array.astype(np.float32)

# ── Grad-CAM ─────────────────────────────────────────────────────────────────
def get_gradcam_heatmap(model, img_array, last_conv_layer_name="Conv_1"):
    grad_model = Model(
        inputs  = model.input,
        outputs = [
            model.get_layer(last_conv_layer_name).output,
            model.output
        ]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        loss = predictions[:, 0]

    grads       = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap      = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap      = tf.squeeze(heatmap)
    heatmap      = np.maximum(heatmap, 0)

    if np.max(heatmap) != 0:
        heatmap = heatmap / np.max(heatmap)

    return np.array(heatmap)


def overlay_gradcam(image_bytes, heatmap, alpha=0.4):
    # Original image
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize((IMG_SIZE, IMG_SIZE))
    img_np = np.array(img)

    # Resize heatmap
    heatmap_resized = cv2.resize(heatmap, (IMG_SIZE, IMG_SIZE))

    # Apply jet colormap
    heatmap_colored = cv2.applyColorMap(
        np.uint8(255 * heatmap_resized),
        cv2.COLORMAP_JET
    )
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

    # Overlay
    overlaid = cv2.addWeighted(img_np, 1 - alpha, heatmap_colored, alpha, 0)

    return img_np, overlaid


def image_to_base64(img_array):
    img = Image.fromarray(img_array.astype(np.uint8))
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG")
    buffer.seek(0)
    return base64.b64encode(buffer.read()).decode("utf-8")

def log_prediction(prediction, confidence, image_bytes):
    try:
        client = get_bq_client()

        table_id = "deepshield-493817.deepshield_data.prediction_logs"

        image_hash = hashlib.md5(image_bytes).hexdigest()

        rows_to_insert = [
            {
                "timestamp": datetime.utcnow().isoformat(),
                "prediction": prediction,
                "confidence_score": float(confidence),
                "image_hash": image_hash,
                "gradcam_generated": True
            }
        ]

        errors = client.insert_rows_json(table_id, rows_to_insert)

        if errors:
            print("BigQuery insert errors:", errors)
        else:
            print("Prediction logged successfully!")

    except Exception as e:
        print("BigQuery logging failed:", e)

# ── Routes ───────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    image_file  = request.files["image"]
    image_bytes = image_file.read()

    # Load model
    mdl = load_model()

    # Preprocess
    img_array = preprocess_image(image_bytes)

    # Predict
    prediction = mdl.predict(img_array, verbose=0)[0][0]

    label = "REAL" if prediction > 0.5 else "FAKE"
    confidence = float(prediction if prediction > 0.5 else 1 - prediction)

    # Log prediction to BigQuery
    log_prediction(label, confidence * 100, image_bytes)

    # Grad-CAM
    heatmap = get_gradcam_heatmap(mdl, img_array)
    original_img, overlaid_img = overlay_gradcam(image_bytes, heatmap)

    # Convert images to base64 to send to frontend
    original_b64 = image_to_base64(original_img)
    overlaid_b64 = image_to_base64(overlaid_img)

    return jsonify({
        "label": label,
        "confidence": round(confidence * 100, 1),
        "original": original_b64,
        "heatmap": overlaid_b64
    })

# ── Run ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)