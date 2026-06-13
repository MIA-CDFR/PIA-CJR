# PIA-CJR — Estrutura e Pipeline

# A. Pasta do Dataset fornecido pela CJR
Deve estar em D:\_PIA_CJR\dataset\raw, ou seja as pastas ficam desta forma:

```text
D:\_PIA_CJR\dataset\raw\2026_05_08_VALPACOS\thermal\ (aqui os vários ficheiros)
D:\_PIA_CJR\dataset\raw\2026_05_08_VALPACOS\visible\ (aqui os vários ficheiros)
D:\_PIA_CJR\dataset\raw\2026_05_14_VALPACOS\thermal\ (aqui os vários ficheiros)
D:\_PIA_CJR\dataset\raw\2026_05_14_VALPACOS\visible\ (aqui os vários ficheiros)
```

ATT. se não for unidade D, mas sim C (por exemplo), devem alterar também no ficheiro D:\partilha\coding\_PIA\PIA-CJR\coding\_CB\configs\config.yaml
a variável "ROOT_DIR:" (na linha 13 do ficheiro confg.yaml) de "D:/_PIA_CJR" para "C:/_PIA_CJR".

---

# B. Ficheiro de configurações

```text
D:\_PIA\PIA-CJR\coding\_CB\configs
```

---

# C. Pipelines

ATT. É possível executar o pipeline completo dos treinos dos modelos (YOLO = reconhecer painéis/células para recorte e DENSENET para classificação), recorte dos paineis/células e evaluate e predict/inferência do modelo DenseNet via script:

```powershell
(DL_Project_venv) PS D:\partilha\coding\_PIA\PIA-CJR\coding\_CB> .\scripts\_run_full_pipeline.bat
```

## FASE 1: IDENTIFICAR PAINEIS (TRABALHO MANUAL NA FERRAMENTA "ROBOFLOW")
No Roboflow, foram carregadas algumas imagens de paineis solares e "desenhados" os paineis.
No final do "desenho", foi gerado um dataset com train, test e validation.
É esse dataset que vamos treinar com YOLO por forma a conseguir reconhecer paineis/células numa imagem drone com arrays de paineis.

## FASE 2: AUMENTAR AO DATASET PARA POSTERIORMENTE TREINAR O YOLO (NA FASE 3)
Uma vez que o dataset criado no Roboflow é pequeno, é possível fazer aumentar artificialmente o dataset.
Para tal, basta alterar o valor da variável (TRUE/FALSE) em config.yaml: "YOLO_DATA_FOLDER_EQUALS_AUGMENTED: FALSE"

```powershell
(DL_Project_venv) PS D:\partilha\coding\_PIA\PIA-CJR\coding\_CB> py .\scripts\augment_roboflow_dataset.py
```

## FASE 3: TREINAR O YOLO
O modelo Yolo aqui treinar visa perceber o que é um painel/célula, de modo a que, quando receber uma imagem de drone com arrays de paineis/células, consiga recortar cada um dosdeles (paineis/células).
Este script vai treinar o YOLO com o "dataset Roboflow" acabado de aumentar na FASE 2.
Descarrega os pesos do modelo YOLO (2 ficheiros: yolo11s-seg.pt e yolo26n.pt) para treinar, e apaga-os no final porque não são mais necessários depois de construir o modelo yolo treinado.

```powershell
(DL_Project_venv) PS D:\partilha\coding\_PIA\PIA-CJR\coding\_CB> python .\scripts\roboflow\train.py
```

## FASE 4: TREINAR A REDE NEURONAL CONVOLUCIONAL (CNN) COM O DATASET ELPV (DISPONÍVEL NA INTERNET)
O objectivo desta fase é treinar uma DenseNet121 com um dataset ELPV .
O modelo treinado tem como função uma classificação binária: DEFECT ou NON DEFECT (@TODO: ver se conseguimos chegar a muliclasse).

```powershell
(DL_Project_venv) PS D:\partilha\coding\_PIA\PIA-CJR\coding\_CB> python .\scripts\train.py
```

Treina com o dataset ELPV.
Guarda as várias versões na pasta D:\_PIA_CJR\models\densenet121__CB_20260524_132313.pth.
E os outputs (accuracy, f1, loss, etc.) em D:\_PIA_CJR\outputs\_CB\figures.
O melhor modelo fica sempre guardado aqui: D:\_PIA_CJR\models\densenet121.pth

## FASE 5: EVALUATE DO MODELO DE CLASSIFICAÇÃO TREINADO COM DENSENET121


### Com o melhor modelo construído

Modelo por defeito: D:\_PIA_CJR\models\densenet121.pth

```powershell
(DL_Project_venv) PS D:\partilha\coding\_PIA\PIA-CJR\coding\_CB> python .\scripts\evaluate.py
```

### Com um modelo específico

Exemplo: D:\_PIA_CJR\models\densenet121_CB_20260521_160502.pth

```powershell
(DL_Project_venv) PS D:\partilha\coding\_PIA\PIA-CJR\coding\_CB> python .\scripts\evaluate.py --model densenet121_CB_20260521_160502.pth
```

Guarda os outputs (classification report, confusion matrix, roc_curve, etc.) em D:\_PIA_CJR\outputs\_CB\evaluation\DenseNet121.


## FASE 6: CRIAR OS PAINEIS/CÉLULAS (Generate Rectified Panels)
Agora precisamos de preparar o dataset real (RAW) para poder ser submetido ao modelo DenseNet treinado com dataset ELPV.
Para isso, precisamos de recortar os paineis/céludas do dataset real (RAW).
Ou seja, precisamos de inferir/submeter o dataset real (RAW = imagens de drones fornecidas pela empresa) ao modelo YOLO treinado para reconhecer paineis/células, e quando degtecta, recorta e gera uma imagem que guarda na pasta "raw_rectified_panels".
Este script vai "partir" um array de paineis em paineis/células e guardar as várias imagens numa pasta (D:\_PIA_CJR\dataset\raw_rectified_panels).

```powershell
(DL_Project_venv) PS D:\partilha\coding\_PIA\PIA-CJR\coding\_CB> py .\scripts\roboflow\generate_rectified_panels.py
```

## FASE 7: PREDICT/INFERÊNCIA DO MODELO DE CLASSIFICAÇÃO TREINADO COM DENSENET121

### Com o melhor modelo construído e as imagens da pasta por defeito (RAW_RECTIFIED_PANELS_DIR em config.yaml)

```powershell
(DL_Project_venv) PS D:\partilha\coding\_PIA\PIA-CJR\coding\_CB> python .\scripts\predict.py
```

### Com o melhor modelo construído e uma imagem específica

```powershell
(DL_Project_venv) PS D:\partilha\coding\_PIA\PIA-CJR\coding\_CB> python .\scripts\predict.py --input "data/image.jpg"
```

### Com o melhor modelo construído e uma pasta específica (recolha em todas subfolders/recursivo)

```powershell
(DL_Project_venv) PS D:\partilha\coding\_PIA\PIA-CJR\coding\_CB> python .\scripts\predict.py --input "path/to/folder/"
```

### Com um modelo específico e uma imagem específica

```powershell
(DL_Project_venv) PS D:\partilha\coding\_PIA\PIA-CJR\coding\_CB> python .\scripts\predict.py --input "data/images" --model "densenet121.pth"
```

### Com o melhor modelo específico e uma pasta específica (recolha em todas subfolders/recursivo)

```powershell
(DL_Project_venv) PS D:\partilha\coding\_PIA\PIA-CJR\coding\_CB> python .\scripts\predict.py --input "path/to/folder/" --model "densenet121.pth"
```

Guarda os outputs (classification report, confusion matrix, roc_curve, etc.) em D:\_PIA_CJR\outputs\_CB\evaluation\DenseNet121.

## FASE 8: INFERÊNCIA/CLASSIFICAR DEFECT / NON DEFECT VIA WEB APP

### Classificação, via upload de imagens "painel" (submetidas previamente ao pipeline generate_rectified_panels)

Correr
```
(DL_Project_venv) PS D:\partilha\coding\_PIA\PIA-CJR\coding\_CB> py .\web_app\server.py
```

E abrir o browser na página
```
http://localhost:5000/
```

Seguir as instruções. Dá para arrastar ficheiros ou pastas inteiras (leitura de subpastas recursiva) e analisar.
Neste caso, as imagens a analisar são as imagens que o modelo recortou, algures em D:\_PIA_CJR\dataset\raw_rectified_panels


### Classificação, via upload de imagem original (do drone)
Correr
```
(DL_Project_venv) PS D:\partilha\coding\_PIA\PIA-CJR\coding\_CB> py .\web_app\server_drone.py
```

E abrir o browser na página
```
http://localhost:5001/
```

Seguir as instruções. Seleccionar uma imagem e analisar.

Seleccionar por exemplo imagem: "dji_20260316104949_0125_t_51675bb4-7b38-44a5-b69b-f82abf4934fa.jpg" porque esta vai gerar muitos paineis.