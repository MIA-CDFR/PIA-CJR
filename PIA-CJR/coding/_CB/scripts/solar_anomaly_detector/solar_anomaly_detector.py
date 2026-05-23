"""
solar_anomaly_detector.py
=========================
Deteta anomalias em painéis solares extraídos de fotos de drone.
Não requer anotações — aprende o padrão "normal" a partir das tuas imagens.

Pipeline:
  1. Carrega todos os painéis extraídos pelo solar_panel_extractor.py
  2. Treina um AutoEncoder convolucional (aprende o aspeto "normal")
  3. Para cada painel: calcula o erro de reconstrução (anomaly score)
  4. Gera relatório visual ordenado por score + heatmaps de anomalia

Uso:
  # Treino + inferência (primeira vez)
  python solar_anomaly_detector.py --panels ./paineis_extraidos/ --output ./resultados/

  # Só inferência com modelo já treinado
  python solar_anomaly_detector.py --panels ./paineis_extraidos/ --output ./resultados/ \\
                                   --model ./resultados/ae_model.pth --no-train

  # Mais epochs para datasets maiores (>200 painéis)
  python solar_anomaly_detector.py --panels ./paineis_extraidos/ --output ./resultados/ \\
                                   --epochs 100

Requisitos:
  pip install torch torchvision opencv-python-headless numpy matplotlib
"""

import argparse
import os
import json
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# ──────────────────────────────────────────────
#  PARÂMETROS
# ──────────────────────────────────────────────
PARAMS = {
    "image_size": 128,       # resolução interna do modelo
    "batch_size": 8,         # reduz se OOM
    "learning_rate": 1e-3,
    "default_epochs": 50,    # aumenta com mais dados
    "num_workers": 0,        # 0 = main thread (mais compatível)
    "top_n_report": 15,      # nº de painéis no relatório detalhado
    "score_red": 60,         # score > X → vermelho (suspeito)
    "score_yellow": 30,      # score > X → amarelo (atenção)
}


# ──────────────────────────────────────────────
#  DATASET
# ──────────────────────────────────────────────
class PanelDataset(Dataset):
    def __init__(self, image_paths: list[str], size: int = 128):
        self.paths = image_paths
        self.size = size

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        img = cv2.imread(self.paths[i], cv2.IMREAD_GRAYSCALE)
        if img is None:
            img = np.zeros((self.size, self.size), dtype=np.uint8)
        img = cv2.resize(img, (self.size, self.size))
        tensor = torch.tensor(img, dtype=torch.float32).unsqueeze(0) / 255.0
        return tensor, self.paths[i]


# ──────────────────────────────────────────────
#  MODELO: AutoEncoder Convolucional
# ──────────────────────────────────────────────
class ConvAutoEncoder(nn.Module):
    """
    AutoEncoder convolucional para deteção de anomalias.

    Encoder: comprime o painel para uma representação latente.
    Decoder: reconstrói o painel a partir do latente.

    Painéis normais → reconstrução fiel (baixo erro).
    Painéis anómalos → reconstrução pobre (alto erro) → anomalia detetada.
    """

    def __init__(self):
        super().__init__()

        # Encoder: 128×128 → 8×8×64
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 3, stride=2, padding=1),   # 64×64
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),  # 32×32
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, stride=2, padding=1), # 16×16
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, 3, stride=2, padding=1), # 8×8
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        # Decoder: 8×8×64 → 128×128×1
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 128, 3, stride=2, padding=1, output_padding=1),  # 16×16
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1),  # 32×32
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1),   # 64×64
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 1, 3, stride=2, padding=1, output_padding=1),    # 128×128
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))


# ──────────────────────────────────────────────
#  TREINO
# ──────────────────────────────────────────────
def train(model: ConvAutoEncoder, loader: DataLoader,
          epochs: int, device: torch.device, output_dir: Path) -> list[float]:
    """Treina o AutoEncoder. Devolve histórico de loss."""
    optimizer = optim.Adam(model.parameters(), lr=PARAMS["learning_rate"])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    loss_fn = nn.MSELoss()
    history = []

    model.train()
    print(f"\nA treinar AutoEncoder ({epochs} epochs, {len(loader.dataset)} imagens)...")

    for epoch in range(epochs):
        total_loss = 0.0
        for imgs, _ in loader:
            imgs = imgs.to(device)
            rec = model(imgs)
            loss = loss_fn(rec, imgs)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        history.append(avg_loss)
        scheduler.step()

        if (epoch + 1) % max(1, epochs // 10) == 0:
            print(f"  Epoch {epoch+1:3d}/{epochs} │ loss = {avg_loss:.5f}")

    # Guarda modelo
    model_path = output_dir / "ae_model.pth"
    torch.save(model.state_dict(), model_path)
    print(f"\nModelo guardado em: {model_path}")

    # Gráfico de loss
    plt.figure(figsize=(8, 4), facecolor='#0f0f0f')
    plt.plot(history, color='#00aaff', linewidth=2)
    plt.title("Loss de treino", color='white', fontsize=13)
    plt.xlabel("Epoch", color='gray'); plt.ylabel("MSE Loss", color='gray')
    plt.tick_params(colors='gray')
    plt.gca().set_facecolor('#1a1a1a')
    for spine in plt.gca().spines.values():
        spine.set_edgecolor('#444')
    plt.tight_layout()
    plt.savefig(output_dir / "training_loss.png", dpi=100,
                facecolor='#0f0f0f', bbox_inches='tight')
    plt.close()

    return history


# ──────────────────────────────────────────────
#  INFERÊNCIA
# ──────────────────────────────────────────────
def predict(model: ConvAutoEncoder, image_paths: list[str],
            device: torch.device) -> list[dict]:
    """
    Calcula anomaly score e heatmap para cada painel.
    Devolve lista ordenada do mais anómalo para o mais normal.
    """
    model.eval()
    results = []

    with torch.no_grad():
        for path in image_paths:
            img_gray = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if img_gray is None:
                continue
            img_128 = cv2.resize(img_gray, (PARAMS["image_size"], PARAMS["image_size"]))
            t = (torch.tensor(img_128, dtype=torch.float32)
                 .unsqueeze(0).unsqueeze(0) / 255.0).to(device)

            rec = model(t)
            err_map = (rec - t) ** 2  # [1, 1, H, W]

            results.append({
                "path": path,
                "name": Path(path).stem,
                "score_raw": err_map.mean().item(),
                "heatmap": err_map.squeeze().cpu().numpy(),
                "original": img_128,
            })

    # Ordena: mais anómalo primeiro
    results.sort(key=lambda x: x["score_raw"], reverse=True)

    # Normaliza scores para 0-100
    raw_scores = np.array([r["score_raw"] for r in results])
    s_min, s_max = raw_scores.min(), raw_scores.max()
    for r in results:
        r["score"] = (r["score_raw"] - s_min) / (s_max - s_min + 1e-8) * 100

    return results


# ──────────────────────────────────────────────
#  RELATÓRIOS VISUAIS
# ──────────────────────────────────────────────
def _score_color(score: float) -> str:
    if score > PARAMS["score_red"]:    return "#ff4444"
    if score > PARAMS["score_yellow"]: return "#ffaa00"
    return "#44ff88"


def save_detailed_report(results: list[dict], output_dir: Path):
    """Top-N painéis com original + heatmap + overlay."""
    n = min(PARAMS["top_n_report"], len(results))
    fig = plt.figure(figsize=(18, n * 3.2), facecolor='#0f0f0f')
    plt.suptitle(
        "Relatório de Anomalias — Painéis Solares\n"
        "(AutoEncoder · ordenado por score de anomalia)",
        color='white', fontsize=15, fontweight='bold', y=0.995
    )

    for i, r in enumerate(results[:n]):
        ax_orig = fig.add_subplot(n, 3, i * 3 + 1)
        ax_heat = fig.add_subplot(n, 3, i * 3 + 2)
        ax_over = fig.add_subplot(n, 3, i * 3 + 3)
        color = _score_color(r["score"])

        # Original
        ax_orig.imshow(r["original"], cmap='gray', vmin=0, vmax=255)
        ax_orig.set_title(f"#{i+1}  {r['name']}", color='white', fontsize=8)
        ax_orig.axis('off')

        # Heatmap
        hm = r["heatmap"]
        hm_norm = (hm - hm.min()) / (hm.max() - hm.min() + 1e-8)
        ax_heat.imshow(hm_norm, cmap='hot', vmin=0, vmax=1)
        ax_heat.set_title("Mapa de erro", color='#ffaa00', fontsize=8)
        ax_heat.axis('off')

        # Overlay
        orig_f = np.stack([r["original"]] * 3, axis=-1).astype(np.float32) / 255.0
        heat_rgb = plt.cm.hot(hm_norm)[:, :, :3]
        overlay = orig_f * 0.55 + heat_rgb * 0.45
        ax_over.imshow(np.clip(overlay, 0, 1))
        ax_over.set_title(f"Score: {r['score']:.0f}/100",
                          color=color, fontsize=10, fontweight='bold')
        ax_over.axis('off')

        for ax in [ax_orig, ax_heat, ax_over]:
            ax.set_facecolor('#0f0f0f')

    plt.tight_layout(rect=[0, 0, 1, 0.993])
    out = output_dir / "report_top_anomalies.png"
    plt.savefig(out, dpi=100, facecolor='#0f0f0f', bbox_inches='tight')
    plt.close()
    print(f"Relatório detalhado: {out}")


def save_overview(results: list[dict], output_dir: Path):
    """Todos os painéis numa grelha ordenada por score."""
    n = len(results)
    cols = 7
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.6, rows * 3.0),
                             facecolor='#0f0f0f')
    axes_flat = axes.flatten() if rows > 1 else list(axes)

    for i, r in enumerate(results):
        ax = axes_flat[i]
        ax.imshow(r["original"], cmap='gray')
        color = _score_color(r["score"])
        ax.set_title(f"#{i+1}  {r['name']}\n{r['score']:.0f}/100",
                     color=color, fontsize=6.5)
        ax.axis('off')
        ax.set_facecolor('#0f0f0f')
        for spine in ax.spines.values():
            spine.set_edgecolor(color)
            spine.set_linewidth(2)
            spine.set_visible(True)

    for j in range(i + 1, len(axes_flat)):
        axes_flat[j].axis('off')

    legend = "verde<30  amarelo:30-60  vermelho>60"
    plt.suptitle(
        f"Todos os painéis ({n}) — ordenados por anomalia\n{legend}",
        color='white', fontsize=11, fontweight='bold'
    )
    plt.tight_layout()
    out = output_dir / "report_overview.png"
    plt.savefig(out, dpi=100, facecolor='#0f0f0f', bbox_inches='tight')
    plt.close()
    print(f"Overview: {out}")


def save_json_results(results: list[dict], output_dir: Path):
    """Exporta scores em JSON para integração com outros sistemas."""
    data = [
        {
            "rank": i + 1,
            "file": r["path"],
            "name": r["name"],
            "score": round(r["score"], 2),
            "score_raw": round(r["score_raw"], 6),
            "alert": (
                "RED"    if r["score"] > PARAMS["score_red"]    else
                "YELLOW" if r["score"] > PARAMS["score_yellow"] else
                "GREEN"
            ),
        }
        for i, r in enumerate(results)
    ]
    out = output_dir / "anomaly_scores.json"
    with open(out, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"Scores JSON: {out}")

    # Sumário no terminal
    red    = sum(1 for d in data if d["alert"] == "RED")
    yellow = sum(1 for d in data if d["alert"] == "YELLOW")
    green  = sum(1 for d in data if d["alert"] == "GREEN")
    print(f"\n{'='*45}")
    print(f"  SUMARIO DE ANOMALIAS ({len(data)} paineis)")
    print(f"{'='*45}")
    print(f"  VERMELHO (suspeito)  score > 60: {red:3d}")
    print(f"  AMARELO  (atencao)   score > 30: {yellow:3d}")
    print(f"  VERDE    (normal)    score < 30: {green:3d}")
    print(f"{'='*45}")
    if red > 0:
        print(f"\n  Top {min(5, red)} mais suspeitos:")
        for d in data[:min(5, red)]:
            print(f"    #{d['rank']:2d}  {d['name']}  score={d['score']:.0f}")


# ──────────────────────────────────────────────
#  MAIN
# ──────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Deteção de anomalias em painéis solares (sem anotações)"
    )
    parser.add_argument("--panels", "-p", required=True,
                        help="Directório com painéis extraídos (.png)")
    parser.add_argument("--output", "-o", default="./anomaly_results",
                        help="Directório de output")
    parser.add_argument("--epochs", "-e", type=int,
                        default=PARAMS["default_epochs"],
                        help=f"Epochs de treino (default: {PARAMS['default_epochs']})")
    parser.add_argument("--model", "-m", default=None,
                        help="Caminho para modelo .pth já treinado")
    parser.add_argument("--no-train", action="store_true",
                        help="Não treina — usa --model diretamente")
    args = parser.parse_args()

    # Setup
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Imagens
    exts = {".png", ".jpg", ".jpeg"}
    panels_dir = Path(args.panels)
    image_paths = sorted([str(p) for p in panels_dir.iterdir()
                          if p.suffix.lower() in exts])
    if not image_paths:
        print(f"Erro: nenhuma imagem encontrada em {panels_dir}")
        return
    print(f"Painéis encontrados: {len(image_paths)}")

    # Modelo
    model = ConvAutoEncoder().to(device)

    if args.no_train and args.model:
        model.load_state_dict(
            torch.load(args.model, map_location=device, weights_only=True)
        )
        print(f"Modelo carregado: {args.model}")
    else:
        # Treino
        dataset = PanelDataset(image_paths, size=PARAMS["image_size"])
        loader = DataLoader(
            dataset,
            batch_size=PARAMS["batch_size"],
            shuffle=True,
            num_workers=PARAMS["num_workers"],
        )
        train(model, loader, args.epochs, device, output_dir)

    # Inferência
    print("\nA calcular scores de anomalia...")
    results = predict(model, image_paths, device)

    # Relatórios
    print("\nA gerar relatórios...")
    save_detailed_report(results, output_dir)
    save_overview(results, output_dir)
    save_json_results(results, output_dir)

    print(f"\nConcluido. Resultados em: {output_dir}/")
    print("  report_top_anomalies.png  — top suspeitos com heatmap")
    print("  report_overview.png       — todos os paineis ordenados")
    print("  anomaly_scores.json       — scores para automatizacao")
    print("  ae_model.pth              — modelo treinado (reutilizavel)")
    print("  training_loss.png         — curva de treino")


if __name__ == "__main__":
    main()
