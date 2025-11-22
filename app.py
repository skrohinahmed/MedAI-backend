from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import resnet18, ResNet18_Weights
from PIL import Image
import io
import os

# ----------------------------------------------------
# Device
# ----------------------------------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using:", DEVICE)

# ----------------------------------------------------
# Paths & Classes
# ----------------------------------------------------
MODEL_PATH = "model/model_weights.pth"
CLASS_NAMES = ["Glioma", "Meningioma", "Notumor", "Pituitary"]


# ----------------------------------------------------
# ConvAttention
# ----------------------------------------------------
class ConvAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avg_out, max_out], dim=1)
        out = self.conv(out)
        attention = self.sigmoid(out)
        return x * attention


# ----------------------------------------------------
# ResNet18 with attention
# ----------------------------------------------------
class ResNet18WithConvAttention(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        base = resnet18(weights=ResNet18_Weights.DEFAULT)

        self.features = nn.Sequential(
            base.conv1,
            base.bn1,
            base.relu,
            base.maxpool,
            base.layer1,
            base.layer2,
            base.layer3,
        )

        self.attn = ConvAttention(kernel_size=5)
        self.layer4 = base.layer4
        self.avgpool = base.avgpool

        self.fc = nn.Sequential(
            nn.Linear(base.fc.in_features, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.attn(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)


# ----------------------------------------------------
# Load model weights
# ----------------------------------------------------
model = ResNet18WithConvAttention(num_classes=len(CLASS_NAMES))
state = torch.load(
    MODEL_PATH,
    map_location=DEVICE,
    weights_only=True
)

model.load_state_dict(state)
model.to(DEVICE)
model.eval()

# ----------------------------------------------------
# Transform
# ----------------------------------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

# ----------------------------------------------------
# Flask App
# ----------------------------------------------------
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})



@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    img_bytes = request.files["image"].read()
    image = Image.open(io.BytesIO(img_bytes)).convert("RGB")

    image = transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs = model(image)
        probs = torch.softmax(outputs, dim=1)
        conf, pred = torch.max(probs, 1)

    return jsonify({
        "predicted_class": CLASS_NAMES[pred.item()],
        "confidence": float(conf.item())
    })


# ----------------------------------------------------
# Run App
# ----------------------------------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
