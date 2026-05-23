import streamlit as st
import cv2
import numpy as np
import os
from PIL import Image
from pathlib import Path

# ============================================================
# CONFIG
# ============================================================

OUTPUT_DIR = "outputs"
UPLOAD_DIR = "input_images"

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(UPLOAD_DIR, exist_ok=True)

# ============================================================
# STREAMLIT UI
# ============================================================

st.set_page_config(
    page_title="Solar Panel Extractor",
    layout="wide"
)

st.title("☀️ Solar Panel Extractor")
st.markdown(
    "Deteção automática de painéis solares em imagens térmicas"
)

# ============================================================
# UPLOAD
# ============================================================

uploaded_file = st.file_uploader(
    "Upload Thermal Image",
    type=["jpg", "jpeg", "png"]
)

# ============================================================
# FUNÇÕES
# ============================================================

def order_points(pts):

    rect = np.zeros((4, 2), dtype="float32")

    s = pts.sum(axis=1)

    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    diff = np.diff(pts, axis=1)

    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    return rect


def four_point_transform(image, pts):

    rect = order_points(pts)

    (tl, tr, br, bl) = rect

    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)

    maxWidth = max(int(widthA), int(widthB))

    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)

    maxHeight = max(int(heightA), int(heightB))

    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]
    ], dtype="float32")

    M = cv2.getPerspectiveTransform(rect, dst)

    warped = cv2.warpPerspective(
        image,
        M,
        (maxWidth, maxHeight)
    )

    return warped


def detect_panels(image):

    original = image.copy()

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # ========================================================
    # EDGE DETECTION
    # ========================================================

    edges = cv2.Canny(gray, 80, 200)

    # ========================================================
    # MORPHOLOGY
    # ========================================================

    kernel = np.ones((3, 3), np.uint8)

    edges = cv2.dilate(edges, kernel, iterations=1)

    edges = cv2.morphologyEx(
        edges,
        cv2.MORPH_CLOSE,
        kernel,
        iterations=2
    )

    # ========================================================
    # CONTORNOS
    # ========================================================

    contours, _ = cv2.findContours(
        edges,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    detected = image.copy()

    extracted_panels = []

    panel_id = 0

    for contour in contours:

        area = cv2.contourArea(contour)

        if area < 3000:
            continue

        rect = cv2.minAreaRect(contour)

        box = cv2.boxPoints(rect)

        box = np.int32(box)

        width = rect[1][0]
        height = rect[1][1]

        if width == 0 or height == 0:
            continue

        ratio = max(width, height) / min(width, height)

        # ====================================================
        # FILTRO GEOMÉTRICO
        # ====================================================

        if ratio < 1.2 or ratio > 3.5:
            continue

        try:

            warped = four_point_transform(
                original,
                box.astype(np.float32)
            )

        except:
            continue

        h, w = warped.shape[:2]

        if h < 40 or w < 40:
            continue

        extracted_panels.append(warped)

        cv2.drawContours(
            detected,
            [box],
            0,
            (0, 255, 0),
            2
        )

        panel_id += 1

    return detected, extracted_panels, edges

# ============================================================
# PROCESSAMENTO
# ============================================================

if uploaded_file is not None:

    file_path = os.path.join(
        UPLOAD_DIR,
        uploaded_file.name
    )

    with open(file_path, "wb") as f:
        f.write(uploaded_file.read())

    image = cv2.imread(file_path)

    st.subheader("Imagem Original")

    st.image(
        cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
        use_container_width=True
    )

    # ========================================================
    # DETEÇÃO
    # ========================================================

    detected, panels, edges = detect_panels(image)

    # ========================================================
    # VISUALIZAÇÃO
    # ========================================================

    col1, col2 = st.columns(2)

    with col1:

        st.subheader("Edges")

        st.image(edges, use_container_width=True)

    with col2:

        st.subheader("Deteções")

        st.image(
            cv2.cvtColor(detected, cv2.COLOR_BGR2RGB),
            use_container_width=True
        )

    # ========================================================
    # PAINÉIS EXTRAÍDOS
    # ========================================================

    st.subheader(f"Painéis Detectados: {len(panels)}")

    cols = st.columns(4)

    for idx, panel in enumerate(panels):

        save_path = os.path.join(
            OUTPUT_DIR,
            f"panel_{idx}.png"
        )

        cv2.imwrite(save_path, panel)

        with cols[idx % 4]:

            st.image(
                cv2.cvtColor(panel, cv2.COLOR_BGR2RGB),
                caption=f"Panel {idx}",
                use_container_width=True
            )

            with open(save_path, "rb") as file:

                st.download_button(
                    label="Download",
                    data=file,
                    file_name=f"panel_{idx}.png",
                    mime="image/png",
                    key=f"download_{idx}"
                )

    st.success("Processamento concluído.")
