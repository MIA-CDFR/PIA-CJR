"""
Solar Panel Drone Image Classifier
DenseNet121 Flask backend
Aligned with predict.py
"""

# ============================================================
# IMPORTS
# ============================================================

import io
import sys
import time

from pathlib import Path

# ============================================================
# PROJECT ROOT
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

sys.path.append(str(PROJECT_ROOT))

# ============================================================
# IMPORTS
# ============================================================

import torch
import torch.nn.functional as F

from PIL import Image

from flask import Flask, request, jsonify, send_from_directory

from flask_cors import CORS

from torchvision import transforms

from src.models.densenet_model import build_densenet121

from src.utils.config_loader import load_config

# ============================================================
# CONFIG
# ============================================================

config = load_config()

CLASS_NAMES = config["CLASS_NAMES"]

IMAGE_SIZE = config["IMAGE_SIZE"]

DEFECT_THRESHOLD = config["DEFECT_THRESHOLD"]

MODELS_DIR = Path(config["MODELS_DIR"])

# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================================================
# UPTIME
# ============================================================

START_TIME = time.time()

# ============================================================
# STARTUP LOG
# ============================================================

print()
print("====================================")
print("PIA-CJR SOLAR CLASSIFIER")
print("====================================")
print(f"Models dir : {MODELS_DIR}")
print(f"Device     : {DEVICE}")
print(f"Threshold  : {DEFECT_THRESHOLD}")
print(f"Classes    : {CLASS_NAMES}")
print("====================================")
print()

# ============================================================
# FLASK APP
# ============================================================

app = Flask(__name__)

CORS(app)

# ============================================================
# TRANSFORMS
# ============================================================

transform = transforms.Compose(
    [
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

# ============================================================
# LIST MODELS
# ============================================================


def list_models():

    model_files = sorted(MODELS_DIR.glob("*.pth"))

    return [model.name for model in model_files]


# ============================================================
# BUILD MODEL
# ============================================================


def build_model():

    model = build_densenet121(num_classes=2, pretrained=False, dropout=0.30)

    return model


# ============================================================
# LOAD MODEL
# ============================================================


def load_model(model_filename):

    model = build_model()

    model_path = MODELS_DIR / model_filename

    if not model_path.exists():

        raise FileNotFoundError(f"Model not found:\n" f"{model_path}")

    print()
    print(f"Loading model: {model_path}")

    checkpoint = torch.load(model_path, map_location=DEVICE)

    model.load_state_dict(checkpoint)

    model.to(DEVICE)

    model.eval()

    print("Model loaded successfully")

    return model


# ============================================================
# PREDICT IMAGE
# ============================================================


def predict_image(image, model):

    tensor = transform(image).unsqueeze(0)

    tensor = tensor.to(DEVICE)

    with torch.no_grad():

        outputs = model(tensor)

        probs = F.softmax(outputs, dim=1)

        non_defect_prob = probs[0, 0].item()

        defect_prob = probs[0, 1].item()

        # ----------------------------------------------------
        # THRESHOLD DECISION
        # ----------------------------------------------------

        if defect_prob >= DEFECT_THRESHOLD:

            predicted_idx = 1

            confidence = defect_prob

        else:

            predicted_idx = 0

            confidence = 1 - defect_prob

    predicted_class = CLASS_NAMES[predicted_idx]

    all_probabilities = {
        "non_defect": round(non_defect_prob * 100, 2),
        "defect": round(defect_prob * 100, 2),
    }

    return {
        "prediction": predicted_class,
        "confidence": round(confidence * 100, 2),
        "defect_probability": round(defect_prob * 100, 2),
        "all_probabilities": all_probabilities,
    }


# ============================================================
# ROUTES
# ============================================================


@app.route("/")
@app.route("/solar_classifier.html")
def index():

    return send_from_directory("templates", "solar_classifier.html")


# ============================================================
# HEALTH — enriquecido
# ============================================================


@app.route("/health")
def health():
    """
    Devolve informação completa sobre o estado do servidor.
    O frontend usa estes dados para preencher os pills de info
    no header (device, threshold, modelos disponíveis) sem
    precisar de chamadas adicionais.
    """

    models = list_models()

    return jsonify(
        {
            "status": "online",
            "device": str(DEVICE),
            "models_available": len(models),
            "models": models,
            "classes": CLASS_NAMES,
            "num_classes": len(CLASS_NAMES),
            "threshold": DEFECT_THRESHOLD,
            "image_size": IMAGE_SIZE,
            "uptime_seconds": int(time.time() - START_TIME),
        }
    )


# ============================================================
# MODELS
# ============================================================


@app.route("/models")
def models_api():

    return jsonify(list_models())


# ============================================================
# PREDICT
# ============================================================


@app.route("/predict", methods=["POST"])
def predict():

    try:

        # ----------------------------------------------------
        # VALIDATE FILE
        # ----------------------------------------------------

        if "file" not in request.files:

            return jsonify({"error": "No file uploaded"}), 400

        file = request.files["file"]

        if file.filename == "":

            return jsonify({"error": "Empty filename"}), 400

        # ----------------------------------------------------
        # MODEL
        # ----------------------------------------------------

        model_filename = request.form.get("model_name")

        if model_filename is None:

            return jsonify({"error": "No model selected"}), 400

        print()
        print("====================================")
        print(f"Model : {model_filename}")
        print(f"Image : {file.filename}")
        print("====================================")

        # ----------------------------------------------------
        # LOAD MODEL
        # ----------------------------------------------------

        model = load_model(model_filename)

        # ----------------------------------------------------
        # LOAD IMAGE
        # ----------------------------------------------------

        image = Image.open(io.BytesIO(file.read())).convert("RGB")

        # ----------------------------------------------------
        # PREDICT
        # ----------------------------------------------------

        prediction = predict_image(image=image, model=model)

        prediction["model"] = model_filename

        return jsonify(prediction)

    except Exception as e:

        print()
        print(f"ERROR: {e}")
        print()

        return jsonify({"error": str(e)}), 500


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    app.run(host="0.0.0.0", port=5000, debug=True)
