# Deteção Automática de Anomalias Térmicas em Painéis Fotovoltaicos Utilizando Deep Learning (YOLO11 e PatchCore)

## Objetivo do Projeto

Criar uma pipeline automática capaz de:

1. Receber imagens térmicas captadas por drone
2. Detetar painéis solares automaticamente
3. Extrair e alinhar cada painel
4. Identificar anomalias térmicas
5. Gerar heatmaps e scores de defeito

---

# Arquitetura Final

```text
Drone thermal image
↓
YOLO11 segmentation
↓
Panel extraction
↓
Perspective rectification
↓
PatchCore anomaly detection
↓
Heatmaps
↓
CSV final
```

---

# Estrutura Final do Projeto

```text
ProjetoAI/
├── input_images/
├── rectified_panels/
├── anomaly_dataset/
│   └── train/
│       └── good/
├── outputs/
│   ├── anomalies/
│   └── results.csv
└── yolo/
    ├── train.py
    ├── generate_rectified_panels.py
    ├── train_patchcore.py
    ├── predict_patchcore.py
    ├── batch_anomaly_detection.py
    └── full_pipeline.py
```

---

# PASSO 1 — Preparação do Dataset YOLO

## Objetivo

Anotar painéis solares para treinar segmentação.

## Ferramenta usada

* Roboflow

## Tipo de tarefa

* Instance Segmentation

## Labels

```text
solar_panel
```

## Export format

```text
YOLOv8 Instance Segmentation
```

---

# PASSO 2 — Treino do YOLO11

## Modelo usado

```python
YOLO("yolo11s-seg.pt")
```

Posteriormente recomendado:

```python
YOLO("yolo11m-seg.pt")
```

---

## Ficheiro

```text
train.py
```

---

## Execução

```bash
python train.py
```

---

## Resultado

Modelo treinado:

```text
runs/segment/train-50/weights/best.pt
```

---

# PASSO 3 — Testar Segmentação

## Objetivo

Verificar se o YOLO segmenta corretamente os painéis.

---

## Resultado esperado

* máscaras dos painéis
* deteções corretas

---

# PASSO 4 — Extração e Rectificação de Painéis

## Objetivo

Transformar:

```text
imagem drone
```

em:

```text
painéis individuais alinhados
```

---

## Ficheiro

```text
generate_rectified_panels.py
```

---

## O que faz

* percorre todas as imagens drone
* segmenta painéis
* corrige perspetiva
* roda painéis verticalmente
* normaliza tamanho
* guarda painéis individuais

---

## Execução

```bash
python generate_rectified_panels.py
```

---

## Resultado

```text
rectified_panels/
├── panel_00001.png
├── panel_00002.png
├── panel_00003.png
```

---

# PASSO 5 — Limpeza Manual do Dataset

## Objetivo

Manter apenas painéis normais.

---

## Remover

* painéis cortados
* deformados
* desfocados
* falsos positivos
* painéis defeituosos
* perspetivas estranhas

---

## Manter

* painéis completos
* alinhados
* térmicamente normais
* geometria consistente

---

# PASSO 6 — Preparação do Dataset PatchCore

## Objetivo

Criar dataset de painéis saudáveis.

---

## Estrutura

```text
anomaly_dataset/
└── train/
    └── good/
        ├── panel_00001.png
        ├── panel_00002.png
```

---

## Importante

O PatchCore aprende:

```text
NORMALIDADE
```

Nunca colocar painéis defeituosos no treino.

---

# PASSO 7 — Treino do PatchCore

## Objetivo

Treinar modelo de anomaly detection.

---

## Biblioteca

```bash
pip install anomalib
```

---

## Dependências adicionais

```bash
pip install lightning
```

---

## Ficheiro

```text
train_patchcore.py
```

---

## Execução

```bash
python train_patchcore.py
```

---

## Resultado

Checkpoint gerado:

```text
results/Patchcore/solar/v1/weights/lightning/model.ckpt
```

---

# PASSO 8 — Inferência Individual PatchCore

## Objetivo

Analisar um único painel.

---

## Ficheiro

```text
predict_patchcore.py
```

---

## O que faz

* carrega modelo PatchCore
* analisa painel
* gera score de anomalia
* cria heatmap

---

## Execução

```bash
python predict_patchcore.py
```

---

## Resultado

```text
outputs/patchcore_result.png
```

---

# PASSO 9 — Batch Anomaly Detection

## Objetivo

Processar múltiplos painéis automaticamente.

---

## Ficheiro

```text
batch_anomaly_detection.py
```

---

## O que faz

* percorre todos os painéis
* executa PatchCore
* gera heatmaps
* gera scores
* cria CSV final

---

## Execução

```bash
python batch_anomaly_detection.py
```

---

## Resultado

```text
outputs/
├── anomalies/
└── results.csv
```

---

# PASSO 10 — Pipeline Completa

## Objetivo

Automatizar todo o sistema.

---

## Ficheiro

```text
full_pipeline.py
```

---

## O que faz

Pipeline completa:

```text
input_images/
↓
YOLO segmentation
↓
panel extraction
↓
rectification
↓
PatchCore
↓
heatmaps
↓
CSV final
```

---

## Execução

```bash
python full_pipeline.py
```

---

## Resultado Final

```text
outputs/
├── anomalies/
│   ├── panel_00001.png
│   ├── panel_00002.png
│
└── results.csv
```

---

# Ordem Correta de Execução

# Primeira vez

## 1.

Treinar YOLO

```bash
python train.py
```

---

## 2.

Gerar painéis alinhados

```bash
python generate_rectified_panels.py
```

---

## 3.

Limpeza manual

* remover painéis maus
* copiar bons para:

```text
anomaly_dataset/train/good/
```

---

## 4.

Treinar PatchCore

```bash
python train_patchcore.py
```

---

## 5.

Executar pipeline completa

```bash
python full_pipeline.py
```

---

# Fluxo Normal Futuro

Depois do sistema treinado:

```bash
python full_pipeline.py
```

é suficiente.

---

# Melhorias Futuras Recomendadas

## YOLO

* mais imagens anotadas
* objetos pequenos
* painéis inclinados
* usar yolo11m-seg
* imgsz=1280

---

## PatchCore

* mais painéis saudáveis
* dataset mais consistente
* mais diversidade térmica

---

# Próximas Evoluções Profissionais

## Dashboard Web

* Streamlit
* FastAPI
* Flask

---

## Funcionalidades Futuras

* upload imagem drone
* dashboard visual
* GPS mapping
* relatórios PDF
* tracking temporal
* severity ranking
* clustering defeitos

---

# Estado Atual do Projeto

Atualmente o sistema já consegue:

✅ segmentação automática de painéis

✅ extração automática

✅ rectificação geométrica

✅ normalização dos painéis

✅ anomaly detection com IA

✅ heatmaps térmicos

✅ scoring automático

✅ pipeline end-to-end

---

# Tecnologias Utilizadas

## Computer Vision

* OpenCV
* NumPy

---

## Deep Learning

* YOLO11
* PatchCore
* PyTorch
* Anomalib

---

## Hardware

* NVIDIA RTX 4050 Laptop GPU

---

# Pipeline Final

```text
Drone Thermal Images
↓
YOLO11 Instance Segmentation
↓
Panel Extraction
↓
Perspective Rectification
↓
Normalized Panel Dataset
↓
PatchCore Anomaly Detection
↓
Thermal Heatmaps
↓
Defect Scoring
↓
CSV Reports
```
