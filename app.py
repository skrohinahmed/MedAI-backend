import io
import os
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import onnxruntime as ort
from PIL import Image
import numpy as np
import torch
from torchvision import transforms

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes and origins

# Load ONNX model
session = ort.InferenceSession("model/resnet18_brain_tumor.onnx", providers=["CPUExecutionProvider"])

# Transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

# Your classes
class_names = ["glioma", "meningioma", "notumor", "pituitary"]

def predict_image(file):
    img_bytes = file.read()
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")

    tensor = transform(img).unsqueeze(0).numpy()

    inputs = {session.get_inputs()[0].name: tensor}
    outputs = session.run(None, inputs)

    logits = outputs[0]
    pred_idx = np.argmax(logits, axis=1)[0]
    confidence = np.max(torch.softmax(torch.tensor(logits), dim=1).numpy())

    return class_names[pred_idx], float(confidence)

# -----------------------------
# ROUTES
# -----------------------------

@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files["image"]

    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    label, conf = predict_image(file)
    return jsonify({
        "prediction": label,
        "confidence": round(conf, 4)
    })

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
