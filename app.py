from flask import Flask, request, jsonify,render_template
from flask_cors import CORS
import torch
from torchvision import transforms
from PIL import Image
import io

# ======================================================
# ⚙️ Configuration
# ======================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "model/best_resnet18_attention_full.pth"
class_names = ['Glioma', 'Meningioma', 'No_Tumor', 'Pituitary']

# ======================================================
# 🧩 Model Definition (must match training)
# ======================================================
from torchvision.models import resnet18, ResNet18_Weights
import torch.nn as nn

class ConvAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(ConvAttention, self).__init__()
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
        self.attn = ConvAttention(kernel_size=7)
        self.layer4 = base.layer4
        self.avgpool = base.avgpool
        self.fc = nn.Sequential(
            nn.Linear(base.fc.in_features, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, num_classes)
        )
    def forward(self, x):
        x = self.features(x)
        x = self.attn(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

# ======================================================
# 🔍 Load Model
# ======================================================
import torch.serialization
torch.serialization.add_safe_globals([ResNet18WithConvAttention, ConvAttention])

model = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
model.to(DEVICE)
model.eval()
print("✅ Model loaded successfully!")

# ======================================================
# 🖼️ Image Transform
# ======================================================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

# ======================================================
# 🔗 Flask App
# ======================================================
app = Flask(__name__)

# Enable CORS for all routes and origins
CORS(app)

# OR if you want to be more specific:
# CORS(app, resources={r"/*": {"origins": "*"}})

# 🏠 Home Route
@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")

# 🔮 Prediction Route
@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files["image"]
    img_bytes = file.read()
    image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    image = transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs = model(image)
        probs = torch.softmax(outputs, dim=1)
        conf, pred = torch.max(probs, 1)

    response = {
        "predicted_class": class_names[pred.item()],
        "confidence": float(conf.item())
    }
    return jsonify(response)

# ======================================================
# 🏃 Run Server
# ======================================================
if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)