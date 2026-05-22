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

# C. Pipeline

# Train

```powershell
(DL_Project_venv) PS D:\_PIA\PIA-CJR\coding\_CB> python .\scripts\train.py
```

---

# Evaluate

## Opção A

Assume por defeito o modelo `densenet121.pth`

```powershell
(DL_Project_venv) PS D:\_PIA\PIA-CJR\coding\_CB> python .\scripts\evaluate.py --model
```

---

## Opção B

Especifica o modelo `densenet121__CB_20260521_180149.pth`

```powershell
(DL_Project_venv) PS D:\_PIA\PIA-CJR\coding\_CB> python .\scripts\evaluate.py --model "densenet121__CB_20260521_180149.pth"
```

---

# Predict

## Opção A — Single Image

```powershell
(DL_Project_venv) PS D:\_PIA\PIA-CJR\coding\_CB> python .\scripts\predict.py --image "D:\_PIA_CJR\dataset\raw\2026_05_14_VALPACOS\visible\dji_20260316104948_0125_v_d342e642-80f1-46ba-802a-17df7e884296.jpg" --model "D:\_PIA_CJR\outputs\_CB\models\densenet121.pth"
```

---

## Opção B — Folder recursivo (até ao "infinito")

```powershell
(DL_Project_venv) PS D:\_PIA\PIA-CJR\coding\_CB> python .\scripts\predict.py --folder "D:\_PIA_CJR\dataset\raw" --model "D:\_PIA_CJR\outputs\_CB\models\densenet121.pth"
```
