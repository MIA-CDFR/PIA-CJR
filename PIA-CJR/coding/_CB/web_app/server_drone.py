"""
Solar Panel Drone Image Classifier — Drone Edition
FILE: scripts/server_drone.py

Recebe imagens de drone (não recortadas).
Pipeline:
    1. YOLO segmentation  → deteta e recorta painéis
    2. DenseNet121        → classifica cada painel (defect / non_defect)
    3. Guarda painéis em disco  (data/rectified_panels/<timestamp>/)
    4. Devolve resultado agregado ao frontend
"""

# ============================================================
# IMPORTS
# ============================================================

import io
import sys
import time
import os

from pathlib import Path
from datetime import datetime

# ============================================================
# PROJECT ROOT
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

# ============================================================
# IMPORTS
# ============================================================

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from PIL import Image

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

from torchvision import transforms
from ultralytics import YOLO

from src.models.densenet_model import build_densenet121
from src.utils.config_loader import load_config

# ============================================================
# CONFIG
# ============================================================

config = load_config()

CLASS_NAMES     = config["CLASS_NAMES"]
IMAGE_SIZE      = config["IMAGE_SIZE"]
DEFECT_THRESHOLD = config["DEFECT_THRESHOLD"]
MODELS_DIR      = Path(config["MODELS_DIR"])
YOLO_BEST_PT    = config["YOLO_BEST_PT"]
RAW_RECTIFIED_PANELS_DIR = Path(config["RAW_RECTIFIED_PANELS_DIR"])

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
print("PIA-CJR SOLAR CLASSIFIER — DRONE")
print("====================================")
print(f"Models dir : {MODELS_DIR}")
print(f"Device     : {DEVICE}")
print(f"Threshold  : {DEFECT_THRESHOLD}")
print(f"Classes    : {CLASS_NAMES}")
print("====================================")
print()

# ============================================================
# LOAD MODELS AT STARTUP  (persistentes em memória)
# ============================================================

# ---- YOLO ----
_yolo_model_path = MODELS_DIR / YOLO_BEST_PT
print(f"Loading YOLO model: {_yolo_model_path}")
YOLO_MODEL = YOLO(str(_yolo_model_path))
print("YOLO model loaded.")

# ---- DenseNet ----

def _list_classifier_models():
    return sorted(MODELS_DIR.glob("*.pth"))

def _load_densenet(model_filename: str):
    model = build_densenet121(num_classes=2, pretrained=False, dropout=0.30)
    model_path = MODELS_DIR / model_filename
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    checkpoint = torch.load(model_path, map_location=DEVICE)
    model.load_state_dict(checkpoint)
    model.to(DEVICE)
    model.eval()
    return model

# Cache: { filename -> model }
_densenet_cache: dict = {}

def get_densenet(model_filename: str):
    """Devolve modelo do cache ou carrega-o (lazy, persistente)."""
    if model_filename not in _densenet_cache:
        print(f"Loading DenseNet: {model_filename}")
        _densenet_cache[model_filename] = _load_densenet(model_filename)
        print(f"DenseNet loaded: {model_filename}")
    return _densenet_cache[model_filename]

# ============================================================
# TRANSFORMS
# ============================================================

TRANSFORM = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# ============================================================
# FLASK APP
# ============================================================

app = Flask(__name__)
CORS(app)

# ============================================================
# HELPERS — PANEL EXTRACTION  (baseado em generate_rectified_panels.py)
# ============================================================

def extract_panels_from_image(
    original_img: np.ndarray,
    source_stem: str,
    session_dir: Path,
) -> list[dict]:
    """
    Corre YOLO na imagem, recorta cada painel detetado,
    guarda em session_dir e devolve lista de dicts com:
        panel_id, panel_name, filename, pil_image
    """
    panels = []

    results = YOLO_MODEL(original_img)

    panel_index = 0

    for r in results:
        if r.masks is None:
            continue

        masks = r.masks.data.cpu().numpy()

        for mask in masks:

            # ---- máscara binária ----
            mask_bin = (mask * 255).astype(np.uint8)
            mask_bin = cv2.resize(
                mask_bin,
                (original_img.shape[1], original_img.shape[0])
            )

            # ---- contornos ----
            contours, _ = cv2.findContours(
                mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            if not contours:
                continue

            contour = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(contour)
            if area < 1000:
                continue

            # ---- bounding rotated rect ----
            rect = cv2.minAreaRect(contour)
            box  = cv2.boxPoints(rect)
            box  = np.int32(box)

            width  = int(rect[1][0])
            height = int(rect[1][1])

            if width < 20 or height < 20:
                continue

            # ---- orientação vertical ----
            if width > height:
                width, height = height, width

            dst_pts = np.array(
                [[0, height - 1], [0, 0],
                 [width - 1, 0], [width - 1, height - 1]],
                dtype="float32",
            )
            src_pts = box.astype("float32")

            # ---- perspective transform ----
            M = cv2.getPerspectiveTransform(src_pts, dst_pts)

            working_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)

            warped = cv2.warpPerspective(working_img, M, (width, height))

            if warped.shape[1] > warped.shape[0]:
                warped = cv2.rotate(warped, cv2.ROTATE_90_CLOCKWISE)

            warped = cv2.resize(warped, (256, 512))

            # ---- guardar em disco ----
            panel_name     = f"{source_stem}_panel_{panel_index:05d}"
            output_filename = f"{panel_name}.png"
            output_path     = session_dir / output_filename

            cv2.imwrite(str(output_path), warped)

            # ---- converter para PIL (RGB) para o classificador ----
            warped_rgb = cv2.cvtColor(warped, cv2.COLOR_GRAY2RGB)
            pil_image  = Image.fromarray(warped_rgb)

            panels.append({
                "panel_index":  panel_index,
                "panel_name":   panel_name,
                "filename":     output_filename,
                "pil_image":    pil_image,
            })

            panel_index += 1

    return panels


# ============================================================
# HELPERS — CLASSIFY PANEL
# ============================================================

def classify_panel(pil_image: Image.Image, densenet) -> dict:
    tensor = TRANSFORM(pil_image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs = densenet(tensor)
        probs   = F.softmax(outputs, dim=1)

        non_defect_prob = probs[0, 0].item()
        defect_prob     = probs[0, 1].item()

        if defect_prob >= DEFECT_THRESHOLD:
            predicted_idx = 1
            confidence    = defect_prob
        else:
            predicted_idx = 0
            confidence    = 1 - defect_prob

    predicted_class = CLASS_NAMES[predicted_idx]

    return {
        "prediction":       predicted_class,
        "confidence":       round(confidence * 100, 2),
        "defect_probability": round(defect_prob * 100, 2),
        "all_probabilities": {
            "non_defect": round(non_defect_prob * 100, 2),
            "defect":     round(defect_prob * 100, 2),
        },
    }


# ============================================================
# ROUTES
# ============================================================

@app.route("/")
@app.route("/solar_classifier_drone.html")
def index():
    return send_from_directory("templates", "solar_classifier_drone.html")


# ============================================================
# HEALTH
# ============================================================

@app.route("/health")
def health():
    classifier_models = [m.name for m in _list_classifier_models()]

    return jsonify({
        "status":            "online",
        "device":            str(DEVICE),
        "yolo_model":        str(YOLO_BEST_PT),
        "models_available":  len(classifier_models),
        "models":            classifier_models,
        "classes":           CLASS_NAMES,
        "num_classes":       len(CLASS_NAMES),
        "threshold":         DEFECT_THRESHOLD,
        "image_size":        IMAGE_SIZE,
        "uptime_seconds":    int(time.time() - START_TIME),
    })


# ============================================================
# MODELS
# ============================================================

@app.route("/models")
def models_api():
    return jsonify([m.name for m in _list_classifier_models()])


# ============================================================
# SERVE PANEL IMAGES
# ============================================================

@app.route("/panels/<session_name>/<filename>")
def serve_panel(session_name, filename):
    """Serve um painel recortado guardado em disco."""
    panel_dir = RAW_RECTIFIED_PANELS_DIR / session_name
    return send_from_directory(str(panel_dir), filename)


# ============================================================
# PREDICT DRONE
# ============================================================

@app.route("/predict_drone", methods=["POST"])
def predict_drone():
    """
    Recebe uma imagem de drone e devolve:
    {
        "session_dir": "...",
        "panels_detected": 5,
        "panels_with_defect": 2,
        "panels": [
            {
                "panel_name": "img001_panel_00000",
                "filename": "img001_panel_00000.png",
                "prediction": "defect",
                "confidence": 91.3,
                "defect_probability": 91.3,
                "all_probabilities": { "non_defect": 8.7, "defect": 91.3 }
            },
            ...
        ],
        "model": "densenet121.pth"
    }
    """
    try:

        # ---- validar ficheiro ----
        if "file" not in request.files:
            return jsonify({"error": "No file uploaded"}), 400

        file = request.files["file"]
        if file.filename == "":
            return jsonify({"error": "Empty filename"}), 400

        # ---- modelo classifier ----
        model_filename = request.form.get("model_name")
        if not model_filename:
            return jsonify({"error": "No model selected"}), 400

        print()
        print("====================================")
        print(f"[DRONE] Model : {model_filename}")
        print(f"[DRONE] Image : {file.filename}")
        print("====================================")

        # ---- criar pasta de sessão ----
        timestamp    = datetime.now().strftime("%Y%m%d_%H%M%S")
        source_stem  = Path(file.filename).stem
        session_name = f"{timestamp}_{source_stem}"
        session_dir  = RAW_RECTIFIED_PANELS_DIR / session_name
        session_dir.mkdir(parents=True, exist_ok=True)

        print(f"[DRONE] Session dir: {session_dir}")

        # ---- ler imagem ----
        file_bytes   = np.frombuffer(file.read(), np.uint8)
        original_img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        if original_img is None:
            return jsonify({"error": "Could not decode image"}), 400

        # ---- extrair painéis com YOLO ----
        panels_raw = extract_panels_from_image(
            original_img=original_img,
            source_stem=source_stem,
            session_dir=session_dir,
        )

        print(f"[DRONE] Panels detected: {len(panels_raw)}")

        if not panels_raw:
            return jsonify({
                "session_dir":        str(session_dir),
                "session_name":       session_name,
                "panels_detected":    0,
                "panels_with_defect": 0,
                "panels":             [],
                "model":              model_filename,
            })

        # ---- classificar cada painel ----
        densenet = get_densenet(model_filename)

        results = []
        defect_count = 0

        for panel in panels_raw:
            classification = classify_panel(panel["pil_image"], densenet)

            if classification["prediction"] == "defect":
                defect_count += 1

            results.append({
                "panel_name":         panel["panel_name"],
                "filename":           panel["filename"],
                **classification,
            })

            print(
                f"  [{panel['panel_name']}] "
                f"{classification['prediction']} "
                f"({classification['confidence']:.1f}%)"
            )

        return jsonify({
            "session_dir":        str(session_dir),
            "session_name":       session_name,
            "panels_detected":    len(results),
            "panels_with_defect": defect_count,
            "panels":             results,
            "model":              model_filename,
        })

    except Exception as e:
        print(f"\nERROR: {e}\n")
        return jsonify({"error": str(e)}), 500


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True)
