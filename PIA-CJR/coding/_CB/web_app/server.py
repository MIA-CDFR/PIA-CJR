# ============================================================
# FILE: web_app/server.py
# ============================================================

import sys

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

sys.path.append(str(PROJECT_ROOT))

# ============================================================
# IMPORTS
# ============================================================

import io

import torch

import torch.nn.functional as F

from PIL import Image

from flask import Flask, render_template, request, jsonify

from torchvision import transforms

from src.utils.config_loader import load_config

from src.models.densenet_model import build_densenet121

from src.models.efficientnet_model import build_efficientnet_b0

# ============================================================
# CONFIG
# ============================================================

config = load_config()

CLASS_NAMES = config["CLASS_NAMES"]

IMAGE_SIZE = config["IMAGE_SIZE"]

MODELS_DIR = Path(config["MODELS_DIR"])

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================================================
# FLASK APP
# ============================================================

app = Flask(__name__)

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
# HELPERS
# ============================================================


def list_models():

    model_files = sorted(MODELS_DIR.glob("*.pth"))

    return [model.name for model in model_files]


# ============================================================
# MODEL BUILDERS REGISTRY
# ============================================================

MODEL_BUILDERS = {
    "densenet121": build_densenet121,
    "efficientnet_b0": build_efficientnet_b0,
}


# ============================================================
# DETECT MODEL TYPE
# ============================================================


def detect_model_type(model_filename):

    filename = model_filename.lower()

    for model_name in MODEL_BUILDERS:

        if model_name in filename:

            return model_name

    raise ValueError(f"Unsupported model filename: " f"{model_filename}")


# ============================================================
# BUILD MODEL
# ============================================================


def build_model(model_type):

    if model_type not in MODEL_BUILDERS:

        raise ValueError(f"Unsupported model type: " f"{model_type}")

    model_config = config["MODELS"][model_type]

    num_classes = model_config["NUM_CLASSES"]

    dropout = model_config["DROPOUT"]

    builder = MODEL_BUILDERS[model_type]

    model = builder(num_classes=num_classes, pretrained=False, dropout=dropout)

    return model


# ============================================================
# LOAD MODEL
# ============================================================


def load_model(model_filename):

    model_type = detect_model_type(model_filename)

    model = build_model(model_type)

    model_path = MODELS_DIR / model_filename

    if not model_path.exists():

        raise FileNotFoundError(f"Model not found:\n" f"{model_path}")

    checkpoint = torch.load(model_path, map_location=DEVICE)

    model.load_state_dict(checkpoint)

    model.to(DEVICE)

    model.eval()

    return model


# ============================================================
# PREDICT IMAGE
# ============================================================


def predict_image(image, model):

    image_tensor = transform(image).unsqueeze(0)

    image_tensor = image_tensor.to(DEVICE)

    with torch.no_grad():

        outputs = model(image_tensor)

        probs = F.softmax(outputs, dim=1)

        # ----------------------------------------------------
        # PROBABILITIES
        # ----------------------------------------------------

        non_defect_prob = probs[0, 0].item()

        defect_prob = probs[0, 1].item()

        # ----------------------------------------------------
        # THRESHOLD
        # ----------------------------------------------------

        threshold = config["DEFECT_THRESHOLD"]

        if defect_prob >= threshold:

            predicted_idx = 1

            confidence = defect_prob

        else:

            predicted_idx = 0

            confidence = 1 - defect_prob

    # --------------------------------------------------------
    # CLASS NAME
    # --------------------------------------------------------

    predicted_class = CLASS_NAMES[predicted_idx]

    # --------------------------------------------------------
    # ALL PROBABILITIES
    # --------------------------------------------------------

    all_probabilities = {
        "non_defect": round(non_defect_prob * 100, 2),
        "defect": round(defect_prob * 100, 2),
    }

    # --------------------------------------------------------
    # RETURN
    # --------------------------------------------------------

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
def index():

    models = list_models()

    return render_template("solar_classifier.html", models=models)


# ============================================================
# API - LIST MODELS
# ============================================================


@app.route("/models")
def get_models():

    return jsonify(list_models())


# ============================================================
# API - PREDICT
# ============================================================


@app.route("/predict", methods=["POST"])
def predict():

    try:

        # ----------------------------------------------------
        # FILE
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

        return jsonify({"error": str(e)}), 500


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print()

    print("====================================")

    print("PIA-CJR SOLAR CLASSIFIER")

    print("====================================")

    print(f"Models dir : {MODELS_DIR}")

    print(f"Device     : {DEVICE}")

    print("====================================")

    print()

    app.run(host="0.0.0.0", port=5000, debug=True)
