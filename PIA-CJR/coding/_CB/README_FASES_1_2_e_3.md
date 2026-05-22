# PIA-CJR — Deteção de Defeitos em Painéis Solares com Deep Learning

## Visão Geral

O projeto PIA-CJR consiste num pipeline profissional de Computer Vision / Deep Learning orientado para deteção de defeitos em painéis solares utilizando:

- imagens térmicas captadas por drone,
- imagens RGB / visible,
- modelos de Deep Learning.

O pipeline foi desenvolvido com foco em:

- arquitetura limpa,
- modularidade,
- escalabilidade,
- reproducibilidade,
- versionamento,
- rastreabilidade de experiências.

---

# Estrutura Atual do Projeto

```text
C:/_PIA_CJR/
│
├── dataset/
│   ├── raw/
│   ├── normalized/
│   ├── processed/
│   └── metadata/
│
└── outputs/
    └── _CB/
        ├── models/
        ├── figures/
        ├── logs/
        ├── experiments/
        ├── checkpoints/
        └── evaluation/


C:/_PIA/
└── coding/
    └── _CB/
        ├── configs/
        │   └── config.yaml
        │
        ├── scripts/
        │   ├── train.py
        │   └── evaluate.py
        │
        └── src/
            ├── datasets/
            ├── models/
            ├── training/
            ├── evaluation/
            └── utils/
```

---


# Objetivos do Projeto

O roadmap do projeto prevê:

- classificação binária,
- classificação multi-classe,
- fusão multimodal thermal + RGB,
- segmentação,
- deteção de objetos,
- Explainable AI (GradCAM),
- pipelines reais de inferência sobre imagens de drone.

---

# FASE 1 — TRAINING

Implementado:

- arquitetura modular,
- pipeline de treino,
- config.yaml centralizado,
- Dataset PyTorch customizado,
- integração ELPV,
- DenseNet121 pretrained,
- Transfer Learning,
- training loop,
- validation loop,
- DataLoaders,
- checkpoints durante o treino,
- geração automática de figures,
- métricas,
- reproducibilidade,
- versionamento de experiências,
- RUN_ID global.

---

# FASE 2 — EVALUATION

Implementado:

- evaluation pipeline independente,
- confusion matrix,
- precision,
- recall,
- F1-score,
- classification report,
- predictions CSV,
- exportação automática de métricas,
- benchmarking de modelos.
- confusion matrix,
- precision,
- recall,
- F1-score,
- classification reports,
- métricas avançadas.

---

# FASE 3 — INFERENCE

Implementado:

- pipeline independente de inferência,
- suporte para single image inference,
- suporte para folder inference recursivo,
- carregamento de modelos versionados,
- inferência sobre imagens reais de drone,
- exportação automática de predictions CSV,
- geração de imagens anotadas,
- suporte para estruturas de pastas complexas,
- reutilização da arquitectura do treino,
- integração com config.yaml.

---

# Próxima Fase (ROADMAP)


# FASE 4 — Computer Vision Avançado

Possíveis evoluções:

- GradCAM,
- multimodal learning,
- segmentação,
- YOLO,
- anomaly detection,
- thermal/RGB fusion.

---

# Dataset

O sistema utiliza atualmente o dataset:

## ELPV — Electroluminescence Photovoltaic Dataset

O dataset é carregado dinamicamente através de:

```python
from elpv_dataset.utils import load_dataset
```

Foi criada uma classe PyTorch customizada responsável por:

- carregar imagens,
- construir labels,
- aplicar transforms,
- devolver tensores PyTorch.

---

# PIPELINES

## Training

```text
Load config
↓
Load dataset
↓
Build model
↓
Train model
↓
Validate
↓
Save best model
↓
Generate figures
↓
Export metrics
```

## Evaluation

```text
Load config
↓
Load model
↓
Load weights
↓
Run evaluation
↓
Generate confusion matrix
↓
Generate classification report
↓
Export predictions CSV
```

## Predict / Inference


```text
Load config
↓
Load model
↓
Load model weights
↓
Load image(s)
↓
Apply transforms
↓
Run inference
↓
Softmax probabilities
↓
Prediction + confidence
↓
Save annotated images
↓
Export predictions CSV
```

---

# Outputs Gerados

## Training

```text
outputs/_CB/models/
```

Contém:
- modelos versionados,
- latest production model,
- snapshots YAML.

---

## Evaluation

```text
outputs/_CB/evaluation/
```

Exemplo:

```text
evaluation/
└── DenseNet121_CB_20260521_160502/
    ├── confusion_matrix.png
    ├── classification_report.txt
    ├── classification_report.csv
    ├── metrics_summary.txt
    └── predictions.csv
```

---

# Inference

```text
outputs/_CB/inference/
```

Exemplo:

```text
inference/
├── predictions.csv
└── annotated/
```

---

# Executar os vários pipelines

## Treino

```powershell
(DL_Project_venv) PS C:\_PIA\PIA-CJR\coding\_CB> python .\scripts\train.py
```

## Evaluation

### Modelo production

```powershell
(DL_Project_venv) PS C:\_PIA\PIA-CJR\coding\_CB> python .\scripts\evaluate.py
```

### Modelo específico

```powershell
(DL_Project_venv) PS C:\_PIA\PIA-CJR\coding\_CB> python .\scripts\evaluate.py --model DenseNet121_CB_20260521_160502.pth
```

## Predict / Inference

A inferência pode incidir sobre um ficheiro/imagem em particular, ou uma pasta (com subpastas recursivo) e utilizar:

- o modelo production actual ou
- modelos antigos/versionado.

No caso de especificar uma pasta, o sistema procura automaticamente:

- .jpg
- .jpeg
- .png
- .tiff
- .bmp


## Alguns exemplos de utilizção

### Exemplo 1: Trata imagens contidas na pasta (e subpastas, recursido) definida em config.yaml (RAW_DIR: "${DATASET_DIR}/raw") com o modelo activo por defeito em config.yaml
```powershell
python .\scripts\predict.py
```

### Exemplo 2: Trata imagem --input com  modelo activo por defeito em config.yaml
```powershell
python .\scripts\predict.py --input "C:\_PIA_CJR\dataset\raw\2026_05_08_VALPACOS\visible\dji_20260316131304_0220_v_b8033dbb-8eee-4b87-8f02-b32b13c75c8f.jpg"
```

### Exemplo 3: Trata imagens contidas na pasta(e subpastas, recursivo) --input com  modelo activo por defeito em config.yaml
```powershell
python .\scripts\predict.py --input "C:\_PIA_CJR\dataset\raw\2026_05_08_VALPACOS\"
```

### Exemplo 4: Trata imagens contidas na pasta (e subpastas, recursivo) --input com modelo --model
```powershell
python .\scripts\predict.py --input "C:\_PIA_CJR\dataset\raw\2026_05_08_VALPACOS\" --model "densenet121__CB_20260522_141732.pth"
```

### Exemplo 5: Trata imagem --input com modelo --model
```powershell
python .\scripts\predict.py --input "C:\_PIA_CJR\dataset\raw\2026_05_08_VALPACOS\visible\dji_20260316131304_0220_v_b8033dbb-8eee-4b87-8f02-b32b13c75c8f.jpg" --model "densenet121__CB_20260522_141732.pth"
```

---

# Tecnologias

- Python
- PyTorch
- torchvision
- OpenCV
- pandas
- matplotlib
- scikit-learn
- pathlib
- tqdm
- YAML

