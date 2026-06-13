# Web App — Solar Panel Classifier

**Universidade do Minho · Mestrado em Inteligência Artificial**
Projecto em parceria com [CJR Renewables](https://cjr-renewables.pt) · © 2026

---

## Descrição

Interface web local para classificação automática de imagens de painéis solares captadas por drone. O utilizador arrasta imagens (ou pastas) para a interface, e o sistema envia cada imagem para um servidor Flask que corre um modelo **DenseNet-121** previamente treinado, devolvendo a previsão e as probabilidades por classe.

---

## Estrutura de ficheiros

```
_CB/
├── configs/
│    └── config.yaml              ← configuração global do projecto
├── src/
│    └── utils/
│         └── config_loader.py   ← carrega e resolve variáveis do config.yaml
└── web_app/
     ├── server.py                ← servidor Flask (backend + API)
     ├── solar_classifier.html   ← interface web (frontend)
     ├── requirements.txt        ← dependências Python
     └── README_web_app.md       ← este ficheiro
```

---

## O que foi implementado

### `server.py` — Backend Flask

| Componente | Detalhe |
|---|---|
| Framework | Flask + Flask-CORS |
| Modelo | DenseNet-121 (`torchvision.models.densenet121`) |
| Carregamento do config | Via `config_loader.py` — sem hardcoding de caminhos ou classes |
| Device | CUDA se disponível, CPU caso contrário |
| Modo DEMO | Se o ficheiro `.pth` não for encontrado, corre com pesos aleatórios |
| Suporte a checkpoints | Aceita `state_dict` directo, ou dicionários com chave `model_state_dict` / `state_dict` |
| Pré-processamento | `Resize(224×224)` → `ToTensor` → `Normalize` (médias e desvios ImageNet) |

**Parâmetros lidos automaticamente do `config.yaml`:**

| Chave YAML | Utilização no servidor |
|---|---|
| `MODELS_DIR` + `BEST_MODEL_NAME` | Caminho completo do ficheiro `.pth` |
| `CLASS_NAMES` | Nomes das classes para o output |
| `NUM_CLASSES` | Dimensão da cabeça do classificador |
| `IMAGE_SIZE` | Tamanho de redimensionamento da imagem (default: 224) |
| `DEFECT_THRESHOLD` | Threshold de defeito (disponível via API `/health`) |

**Endpoints da API:**

| Método | Rota | Descrição |
|---|---|---|
| `GET` | `/` | Serve a interface HTML |
| `GET` | `/solar_classifier.html` | Idem (alias) |
| `POST` | `/predict` | Recebe uma imagem, devolve previsão em JSON |
| `GET` | `/health` | Estado do servidor, modelo carregado, classes, device |
| `GET` | `/config` | Configuração activa (útil para debug) |

**Exemplo de resposta do `/predict`:**
```json
{
  "filename": "dji_20260316_0229.jpg",
  "prediction": "defect",
  "confidence": 94.3,
  "is_defect": true,
  "all_probabilities": {
    "defect": 94.3,
    "non_defect": 5.7
  }
}
```

---

### `solar_classifier.html` — Frontend

Interface de página única (HTML + CSS + JavaScript puro, sem dependências externas).

**Funcionalidades:**

- Drag-and-drop de imagens individuais ou pastas inteiras (via `webkitGetAsEntry`)
- Selecção de ficheiros pelo explorador do sistema operativo
- Formatos aceites: JPG, PNG, TIFF, WebP, BMP
- Fila de ficheiros com estado por item (`aguarda` / spinner / `ok` / `erro`)
- Barra de progresso global durante a análise
- Resultados em cartões individuais com:
  - Miniatura da imagem
  - Classe prevista e percentagem de confiança
  - Barras de probabilidade para todas as classes
  - Código de cor verde (normal) / vermelho (defeito)
- Indicador de estado do servidor (online/offline) com polling automático a cada 10 segundos
- Layout fixo ao viewport — scroll independente na fila e nos resultados, sem colapso com muitas imagens
- Rodapé de afiliação académica e institucional no cabeçalho

---

## Pré-requisitos

- Python 3.9 ou superior
- Ambiente virtual do projecto activo (`DL_Project_venv`)
- Ficheiro `best_model.pth` gerado pelo pipeline de treino

---

## Instalação de dependências

Na pasta `web_app/`, com o ambiente activo:

```powershell
pip install -r requirements.txt
```

`requirements.txt` inclui:
```
torch>=2.0.0
torchvision>=0.15.0
flask>=3.0.0
flask-cors>=4.0.0
Pillow>=10.0.0
```

> Se o `torch` já estiver instalado no ambiente do projecto, este passo pode ser ignorado.

---

## Como executar

A partir da raiz do projecto `_CB` (onde o `config_loader.py` é encontrado correctamente):

```powershell
cd C:\partilha\coding\_PIA\PIA-CJR\coding\_CB
python .\web_app\server.py
```

Output esperado na consola:

```
Config carregado com sucesso.
Modelo esperado em : C:/_PIA_CJR/outputs/_CB/models/best_model.pth
Classes            : ['non_defect', 'defect']
Num classes        : 2
Defect threshold   : 0.5
Device             : cpu

A carregar modelo...
Modelo carregado de: C:/_PIA_CJR/outputs/_CB/models/best_model.pth

─────────────────────────────────────
Abre no browser:  http://localhost:5000
─────────────────────────────────────
```

Abre o browser em **`http://localhost:5000`** — a interface é servida directamente pelo Flask.

Para parar o servidor: `Ctrl+C`

---

## Fluxo de funcionamento

```
Utilizador arrasta imagens
         │
         ▼
  solar_classifier.html
  (browser, localhost:5000)
         │
         │  POST /predict  (multipart/form-data)
         ▼
     server.py  (Flask)
         │
         ├── Lê config.yaml via config_loader.py
         ├── Carrega best_model.pth (DenseNet-121)
         ├── Pré-processa imagem (resize → tensor → normaliza)
         ├── Inferência com torch.no_grad()
         └── Devolve JSON com previsão + probabilidades
         │
         ▼
  Interface actualiza cartão de resultado
```

---

## Notas

- O servidor corre em `0.0.0.0:5000`, pelo que é acessível na rede local via `http://<IP-da-máquina>:5000` (útil para testar noutro dispositivo).
- Em modo DEMO (sem `.pth`), o servidor funciona mas as previsões são aleatórias — útil para testar a interface.
- Para alterar as classes ou o nome do modelo, basta editar o `config.yaml`; o servidor não precisa de ser modificado.
- O servidor é de desenvolvimento (Flask built-in). Para deployment em produção usar Gunicorn ou similar.
