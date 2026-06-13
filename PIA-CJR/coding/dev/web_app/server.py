"""
Solar Panel Drone Image Classifier
DenseNet121 Flask backend
Aligned with predict.py

em de receber as imagens dos paineis solares já recortadas.


Devemos chamar o ficheiro /templates/solar_classifier.html para servir o frontend,
    e o endpoint /predict para receber as imagens e devolver as predições.
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
# LOAD MODEL (com cache, robusto como server_drone.py)
# ============================================================

def _load_densenet(model_filename: str):
    """Carrega um modelo DenseNet do disco."""
    model = build_model()
    model_path = MODELS_DIR / model_filename
    
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    
    print(f"Loading model from: {model_path}")
    checkpoint = torch.load(model_path, map_location=DEVICE)
    model.load_state_dict(checkpoint)
    model.to(DEVICE)
    model.eval()
    print(f"Model loaded successfully")
    return model

# Cache de modelos (lazy loading, persistente em memória)
_model_cache: dict = {}

def get_model(model_filename: str):
    """Devolve modelo do cache ou carrega-o (lazy, persistente)."""
    if model_filename not in _model_cache:
        print(f"[CACHE] Loading model: {model_filename}")
        _model_cache[model_filename] = _load_densenet(model_filename)
        print(f"[CACHE] Model cached: {model_filename}")
    return _model_cache[model_filename]


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
        # VALIDATE FILES  (aceita múltiplos ficheiros)
        # ----------------------------------------------------

        files = request.files.getlist("files")

        # compatibilidade retroativa: aceita também "file" (singular)
        if not files:
            single = request.files.get("file")
            if single:
                files = [single]

        if not files:
            return jsonify({"error": "No file uploaded"}), 400

        files = [f for f in files if f.filename != ""]
        if not files:
            return jsonify({"error": "Empty filename"}), 400

        valid_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.tif', '.webp'}

        # ----------------------------------------------------
        # MODEL
        # ----------------------------------------------------

        model_filename = request.form.get("model_name")
        if model_filename is None:
            return jsonify({"error": "No model selected"}), 400

        model = get_model(model_filename)

        print()
        print("====================================")
        print(f"Model  : {model_filename}")
        print(f"Images : {len(files)}")
        print("====================================")

        # ----------------------------------------------------
        # PROCESSAR CADA IMAGEM
        # ----------------------------------------------------

        results = []

        for file in files:

            file_ext = Path(file.filename).suffix.lower()
            if file_ext not in valid_extensions:
                results.append({
                    "filename": file.filename,
                    "error": f"Formato inválido: {file_ext}",
                })
                continue

            try:
                image = Image.open(io.BytesIO(file.read())).convert("RGB")
            except Exception as e:
                results.append({
                    "filename": file.filename,
                    "error": f"Não foi possível abrir a imagem: {str(e)}",
                })
                continue

            prediction = predict_image(image=image, model=model)
            prediction["filename"] = file.filename
            prediction["model"]    = model_filename

            print(f"  [{file.filename}] {prediction['prediction']} ({prediction['confidence']:.1f}%)")

            results.append(prediction)

        # ----------------------------------------------------
        # RESPOSTA
        # ----------------------------------------------------

        # Se foi enviado apenas 1 ficheiro, devolve o dict diretamente
        # (compatibilidade com frontend antigo que espera objeto, não lista)
        if len(results) == 1:
            return jsonify(results[0])

        return jsonify({"results": results, "model": model_filename})

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