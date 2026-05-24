"""
solar_panel_extractor.py
========================
Extrai painéis solares individuais de fotografias aéreas de drone,
preparando-os para submissão ao modelo elpv (300x300 grayscale).

Funciona com:
  - Qualquer orientação/rotação do array
  - Número variável de painéis por fila
  - Painéis de tamanho irregular
  - Fundo heterogéneo (terra, rochas, vegetação)

Requisitos:
  pip install opencv-python-headless numpy scipy matplotlib

Uso:
  python solar_panel_extractor.py --input foto.jpg --output ./paineis/
  python solar_panel_extractor.py --input ./fotos/ --output ./paineis/ --debug
"""

import cv2
import numpy as np
import argparse
import os
from pathlib import Path
from scipy.ndimage import uniform_filter1d
from scipy.signal import find_peaks

# ─────────────────────────────────────────────
#  PARÂMETROS (ajusta conforme as tuas imagens)
# ─────────────────────────────────────────────
PARAMS = {
    # Segmentação de filas
    "blue_threshold": 20,  # threshold para detetar pixels "azuis" de painéis
    "morph_close_px": 60,  # tamanho do kernel de closing (fechar gaps internos)
    "morph_open_px": 30,  # tamanho do kernel de opening (remover ruído)
    "min_row_area": 100_000,  # área mínima (px²) para uma fila ser considerada
    # Segmentação de painéis dentro de cada fila
    "proj_smooth_px": 20,  # suavização da projeção
    "valley_height": 0.20,  # altura mínima do vale (0-1)
    "valley_prominence": 0.15,  # prominência mínima do vale
    "min_panel_width_frac": 0.06,  # largura mínima do painel (fração da fila)
    "min_blue_density": 0.08,  # densidade mínima de pixels azuis no painel
    # Output
    "output_size": (300, 300),  # tamanho final (compatível com elpv)
    "clahe_clip": 2.0,  # contraste CLAHE
    "clahe_grid": (8, 8),  # grid CLAHE
}


def build_blue_mask(img: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Cria máscara de pixels 'azuis' (painéis solares).
    Usa b - 0.5*(r+g) para separar painéis do fundo verde/castanho.
    Devolve (blue_raw, blue_cleaned).
    """
    b, g, r = cv2.split(img)
    blue_raw = np.clip(
        b.astype(np.float32) - 0.5 * (r.astype(np.float32) + g.astype(np.float32)),
        0,
        255,
    ).astype(np.uint8)

    _, thresh = cv2.threshold(
        blue_raw, PARAMS["blue_threshold"], 255, cv2.THRESH_BINARY
    )

    k_close = cv2.getStructuringElement(
        cv2.MORPH_RECT, (PARAMS["morph_close_px"], PARAMS["morph_close_px"])
    )
    k_open = cv2.getStructuringElement(
        cv2.MORPH_RECT, (PARAMS["morph_open_px"], PARAMS["morph_open_px"])
    )
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, k_close)
    cleaned = cv2.morphologyEx(closed, cv2.MORPH_OPEN, k_open)

    return blue_raw, cleaned


def detect_rows(cleaned: np.ndarray) -> list:
    """
    Deteta as filas de painéis (blobs grandes na máscara processada).
    Devolve lista de contornos ordenados da esquerda para a direita.
    """
    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    rows = [c for c in contours if cv2.contourArea(c) >= PARAMS["min_row_area"]]
    # Ordena da esquerda para a direita pelo centro X
    rows.sort(key=lambda c: cv2.minAreaRect(c)[0][0])
    return rows


def rotate_and_crop(
    src: np.ndarray, cx: float, cy: float, w: float, h: float, angle: float
) -> np.ndarray:
    """Aplica rotação e crop centrado na fila."""
    interp = cv2.INTER_CUBIC if src.dtype == np.uint8 else cv2.INTER_NEAREST
    M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
    rotated = cv2.warpAffine(src, M, (src.shape[1], src.shape[0]), flags=interp)
    x1 = max(0, int(cx - w / 2))
    y1 = max(0, int(cy - h / 2))
    x2 = min(src.shape[1], int(cx + w / 2))
    y2 = min(src.shape[0], int(cy + h / 2))
    return rotated[y1:y2, x1:x2]


def find_panel_cuts(row_blue_raw: np.ndarray, rw: int) -> np.ndarray:
    """
    Encontra as posições X de separação entre painéis numa fila.
    Usa projeção horizontal + deteção de vales.
    """
    proj = row_blue_raw.sum(axis=0).astype(np.float32)
    proj_norm = proj / (proj.max() + 1e-6)
    proj_smooth = uniform_filter1d(proj_norm, size=PARAMS["proj_smooth_px"])

    min_dist = max(int(rw * PARAMS["min_panel_width_frac"]), 50)
    valleys, _ = find_peaks(
        1.0 - proj_smooth,
        height=PARAMS["valley_height"],
        distance=min_dist,
        prominence=PARAMS["valley_prominence"],
    )
    return valleys


def postprocess_panel(gray_crop: np.ndarray) -> np.ndarray:
    """
    Normaliza para o formato elpv: resize 300x300 + CLAHE.
    """
    resized = cv2.resize(gray_crop, PARAMS["output_size"])
    clahe = cv2.createCLAHE(
        clipLimit=PARAMS["clahe_clip"], tileGridSize=PARAMS["clahe_grid"]
    )
    return clahe.apply(resized)


def extract_panels(img_path: str, output_dir: str, debug: bool = False) -> list[dict]:
    """
    Pipeline completo: imagem drone → painéis individuais prontos para elpv.

    Devolve lista de dicionários com:
        id, row, panel_idx, path, height_px, width_px
    """
    img = cv2.imread(str(img_path))
    if img is None:
        raise FileNotFoundError(f"Não foi possível ler: {img_path}")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blue_raw, cleaned = build_blue_mask(img)
    row_contours = detect_rows(cleaned)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    base_name = Path(img_path).stem

    if debug:
        debug_dir = output_dir / "debug"
        debug_dir.mkdir(exist_ok=True)

    all_panels = []
    global_id = 0

    print(f"\n[{base_name}] {len(row_contours)} filas detetadas")

    for row_i, cnt in enumerate(row_contours):
        rect = cv2.minAreaRect(cnt)
        (cx, cy), (w, h), angle = rect

        # Garante w > h (dimensão longa = comprimento da fila)
        if w < h:
            w, h = h, w
            angle += 90

        # Extrai a fila corrigida de rotação
        row_gray = rotate_and_crop(gray, cx, cy, w, h, angle)
        row_blue = rotate_and_crop(blue_raw, cx, cy, w, h, angle)
        rh, rw = row_gray.shape

        if rw == 0 or rh == 0:
            continue

        # Encontra cortes entre painéis
        cuts = find_panel_cuts(row_blue, rw)

        if debug:
            _save_debug_row(row_gray, row_blue, cuts, row_i, debug_dir)

        boundaries = [0] + list(cuts) + [rw]
        panel_count = 0

        for pi in range(len(boundaries) - 1):
            xs, xe = boundaries[pi], boundaries[pi + 1]
            pw = xe - xs

            # Filtra segmentos demasiado estreitos
            if pw < rw * PARAMS["min_panel_width_frac"]:
                continue

            # Filtra segmentos com pouco conteúdo azul (fundo/sombra)
            density = row_blue[:, xs:xe].astype(np.float32).mean() / 255.0
            if density < PARAMS["min_blue_density"]:
                continue

            panel_gray = row_gray[:, xs:xe]
            panel_processed = postprocess_panel(panel_gray)

            fname = output_dir / f"{base_name}_r{row_i:02d}_p{panel_count:02d}.png"
            cv2.imwrite(str(fname), panel_processed)

            all_panels.append(
                {
                    "id": global_id,
                    "row": row_i,
                    "panel_idx": panel_count,
                    "path": str(fname),
                    "height_px": rh,
                    "width_px": pw,
                    "blue_density": round(density, 3),
                }
            )
            global_id += 1
            panel_count += 1

        print(f"  Fila {row_i}: {panel_count} painéis extraídos")

    print(f"  → Total: {global_id} painéis guardados em {output_dir}/\n")

    if debug:
        _save_debug_overview(
            img, row_contours, output_dir / "debug" / f"{base_name}_overview.jpg"
        )

    return all_panels


# ─────────────────────────────────────────────
#  Funções de debug
# ─────────────────────────────────────────────


def _save_debug_row(row_gray, row_blue, cuts, row_i, debug_dir):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.ndimage import uniform_filter1d

    rh, rw = row_gray.shape
    proj = row_blue.sum(axis=0).astype(np.float32)
    proj_norm = proj / (proj.max() + 1e-6)
    proj_smooth = uniform_filter1d(proj_norm, size=PARAMS["proj_smooth_px"])

    fig, axes = plt.subplots(1, 3, figsize=(18, 3))
    axes[0].imshow(row_gray, cmap="gray", aspect="auto")
    for c in cuts:
        axes[0].axvline(c, color="red", lw=2)
    axes[0].set_title(f"Fila {row_i} — {len(cuts)+1} segmentos")
    axes[1].imshow(row_blue, cmap="gray", aspect="auto")
    axes[1].set_title("Canal azul raw")
    axes[2].plot(proj_smooth, "b")
    for c in cuts:
        axes[2].axvline(c, color="r", ls="--")
    axes[2].set_title("Projeção X + cortes")
    plt.tight_layout()
    plt.savefig(str(debug_dir / f"row_{row_i:02d}.png"), dpi=80)
    plt.close()


def _save_debug_overview(img, row_contours, out_path):
    debug_img = img.copy()
    colors = [(0, 255, 0), (0, 200, 255), (255, 100, 0), (200, 0, 255), (0, 255, 180)]
    for i, cnt in enumerate(row_contours):
        rect = cv2.minAreaRect(cnt)
        box = np.int32(cv2.boxPoints(rect))
        cv2.drawContours(debug_img, [box], 0, colors[i % len(colors)], 8)
        cx, cy = int(rect[0][0]), int(rect[0][1])
        cv2.putText(
            debug_img,
            f"Fila {i}",
            (cx - 60, cy),
            cv2.FONT_HERSHEY_SIMPLEX,
            2,
            colors[i % len(colors)],
            5,
        )
    small = cv2.resize(debug_img, (1280, 960))
    cv2.imwrite(str(out_path), small)


# ─────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Extrai painéis solares de imagens drone para o formato elpv"
    )
    parser.add_argument(
        "--input",
        "-i",
        required=True,
        help="Ficheiro de imagem ou directório com imagens",
    )
    parser.add_argument(
        "--output",
        "-o",
        default="./paineis_extraidos",
        help="Directório de output (default: ./paineis_extraidos)",
    )
    parser.add_argument(
        "--debug",
        "-d",
        action="store_true",
        help="Guarda imagens de debug (projeções, bounding boxes)",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    extensions = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}

    if input_path.is_file():
        images = [input_path]
    elif input_path.is_dir():
        images = [p for p in input_path.iterdir() if p.suffix.lower() in extensions]
        images.sort()
    else:
        print(f"Erro: {input_path} não existe")
        return

    print(f"A processar {len(images)} imagem(ns)...")
    total = 0
    for img_path in images:
        panels = extract_panels(str(img_path), args.output, debug=args.debug)
        total += len(panels)

    print(f"✓ Concluído. {total} painéis extraídos no total.")
    print(f"  Prontos para submissão ao modelo elpv (300x300 grayscale PNG)")


if __name__ == "__main__":
    main()
