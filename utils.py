import torch
from torchvision import transforms
from PIL import Image
import torch.nn.functional as F

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Validation transforms (same as training)
val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

def load_model(model_path):
    model = torch.load(model_path, map_location=DEVICE)
    model.eval()
    return model
import torch
from torchvision import transforms
from PIL import Image
import torch.nn.functional as F

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Validation transforms
val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

def load_model(model_path):
    # Load the full saved model
    model = torch.load(model_path, map_location=DEVICE)
    model.eval()
    return model

def predict_image(model, image: Image.Image, classes: list):
    img = val_transform(image).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        outputs = model(img)
        probs = F.softmax(outputs, dim=1).cpu().numpy()[0]
        pred_index = probs.argmax()
    return classes[pred_index]

def predict_image(model, image: Image.Image, classes: list):
    img = val_transform(image).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        outputs = model(img)
        probs = F.softmax(outputs, dim=1).cpu().numpy()[0]
        pred_index = probs.argmax()
    return {
        "predicted_class": classes[pred_index],
        "probabilities": {cls: float(probs[i]) for i, cls in enumerate(classes)}
    }
