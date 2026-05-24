# PIA-CJR — Estrutura e Pipeline

# A. Estrutura de pastas

```text
D:\_PIA\PIA-CJR\coding
```

Aqui vamos ter:

- `_CB` (pasta Carlos Bergueira)
- `_RR` (pasta Rui Rodrigues)
- etc...
- `dev` (desenvolvimento quando juntarmos tudo)
- `final` (versão final a entregar)

---

# B. Ficheiro de configurações

```text
D:\_PIA\PIA-CJR\coding\_CB\configs
```

---

# C. Pipelines

## FASE 1: IDENTIFICAR PAINEIS
O objectivo é treinar o modelo/arquitectura YOLO com um dataset previamente criado no Rodoflow.
No Rodoflow, carreguei algumas imagens de paineis solares, "desenhei" os paineis e no final gerei um dataset com train, test e validation. E é esse dataset que aqui se treina.

### Train

```powershell
(DL_Project_venv) PS D:\_PIA\PIA-CJR\coding\_CB> python .\scripts\roboflow\train.py
```

Este script vai treinar o YOLO com o dataset gerado no Roboflow.
Descarrega os pesos do modelo YOLO (2 ficheiros: ___ e ____) para treinar, e apaga-os no final porque não são mais necessários depois de construir o modelo yolo treinado.

### Generate rectified panels

```powershell
(DL_Project_venv) PS D:\_PIA\PIA-CJR\coding\_CB> python .\scripts\roboflow\generate_rectified_panels.py
```

Este script vai "partir" um array de paineis em paineis e guardar as várias imagens numa pasta (D:\_PIA_CJR\dataset\rectified_panels).

---

## FASE 2: CLASSIFICAR DEFECT / NON DEFECT VIA LINHA COMANDOS

### Train

```powershell
(DL_Project_venv) PS D:\_PIA\PIA-CJR\coding\_CB> python .\scripts\train.py
```

Treina com o dataset ELPV.
Guarda as várias versões na pasta D:\_PIA_CJR\models\densenet121__CB_20260524_132313.pth.
E os outputs (accuracy, f1, loss, etc.) em D:\_PIA_CJR\outputs\_CB\figures.
O melhor modelo fica sempre guardado aqui: D:\_PIA_CJR\models\densenet121.pth

### Evaluate

#### Com o melhor modelo construído

Modelo por defeito: D:\_PIA_CJR\models\densenet121.pth

```powershell
(DL_Project_venv) PS D:\_PIA\PIA-CJR\coding\_CB> python .\scripts\evaluate.py
```

#### Com um modelo específico

Exemplo: D:\_PIA_CJR\models\densenet121_CB_20260521_160502.pth

```powershell
(DL_Project_venv) PS C:\_PIA\PIA-CJR\coding\_CB> python .\scripts\evaluate.py --model densenet121_CB_20260521_160502.pth
```

Guarda os outputs (classification report, confusion matrix, roc_curve, etc.) em D:\_PIA_CJR\outputs\_CB\evaluation\DenseNet121.


### Predict

#### Com o melhor modelo construído e as imagens da pasta por defeito (RECTIFIED_PANELS_DIR em config.yaml)

```powershell
(DL_Project_venv) PS D:\_PIA\PIA-CJR\coding\_CB> python .\scripts\predict.py
```

#### Com o melhor modelo construído e uma imagem específica

```powershell
(DL_Project_venv) PS D:\_PIA\PIA-CJR\coding\_CB> python .\scripts\predict.py --input "data/image.jpg"
```

#### Com o melhor modelo construído e uma pasta específica (recolha em todas subfolders/recursivo)

```powershell
(DL_Project_venv) PS D:\_PIA\PIA-CJR\coding\_CB> python .\scripts\predict.py --input "path/to/folder/"
```

#### Com um modelo específico e uma imagem específica

```powershell
(DL_Project_venv) PS D:\_PIA\PIA-CJR\coding\_CB> python .\scripts\predict.py --input "data/images" --model "densenet121.pth"
```

#### Com o melhor modelo específico e uma pasta específica (recolha em todas subfolders/recursivo)

```powershell
(DL_Project_venv) PS D:\_PIA\PIA-CJR\coding\_CB> python .\scripts\predict.py --input "path/to/folder/" --model "densenet121.pth"
```

Guarda os outputs (classification report, confusion matrix, roc_curve, etc.) em D:\_PIA_CJR\outputs\_CB\evaluation\DenseNet121.


## CLASSIFICAR DEFECT / NON DEFECT VIA WEB APP

Correr
```
(DL_Project_venv) PS D:\partilha\coding\_PIA\PIA-CJR\coding\_CB> py .\web_app\server.py
```

E abrir o browser na página
```
http://localhost:5000/
```

Basta agora seguir as instruções.
