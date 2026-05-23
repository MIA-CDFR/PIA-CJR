# Pipeline Completo — DenseNet121 para Deteção de Defeitos em Painéis Fotovoltaicos

## 1. Introdução

O presente projeto teve como objetivo desenvolver um sistema de Deep Learning capaz de detetar automaticamente defeitos em painéis fotovoltaicos, utilizando imagens eletroluminescentes (ELPV Dataset).

A arquitetura escolhida para a primeira fase experimental foi a DenseNet121, devido:

- à sua elevada capacidade de extração de características;
- ao bom equilíbrio entre performance e custo computacional;
- à forte utilização em problemas de classificação de imagem;
- à sua eficiência relativamente a redes CNN tradicionais.

O objetivo principal consistiu em construir um pipeline robusto, reproduzível e cientificamente válido, incluindo:

- pré-processamento;
- data augmentation;
- treino supervisionado;
- avaliação quantitativa;
- otimização de threshold;
- análise ROC/PR;
- inferência automatizada.

---

# 2. Dataset Utilizado

## 2.1 Dataset ELPV

Foi utilizado o dataset ELPV (Electroluminescence Photovoltaic Dataset), contendo imagens de painéis solares obtidas através de eletroluminescência.

As imagens representam diferentes tipos de defeitos em células fotovoltaicas.

---

## 2.2 Problema de Classificação

O problema foi simplificado para classificação binária:

| Classe Original | Classe Final |
|---|---|
| normal | non_defect |
| defect classes | defect |

Assim:

- Classe 0 → non_defect
- Classe 1 → defect

---

## 2.3 Dimensão do Dataset

O dataset utilizado continha:

- 2624 imagens totais.

O dataset foi dividido em:

- treino;
- validação;
- teste.

---

# 3. Estratégia de Divisão do Dataset

## 3.1 Stratified Split

Foi utilizada divisão estratificada (Stratified Split), garantindo a manutenção da distribuição das classes em todos os subconjuntos.

A divisão utilizada foi:

| Subconjunto | Percentagem |
|---|---|
| Train | 70% |
| Validation | 15% |
| Test | 15% |

---

## 3.2 Persistência dos Splits

Os índices dos subconjuntos foram guardados em ficheiros JSON.

Isto permitiu:

- reprodutibilidade experimental;
- reutilização consistente do mesmo split;
- comparação justa entre arquiteturas.

---

# 4. Reprodutibilidade Experimental

Foi implementado controlo de reprodutibilidade através da definição explícita de sementes aleatórias.

Foram configuradas seeds para:

- Python;
- NumPy;
- PyTorch.

Além disso:

- cudnn deterministic foi ativado;
- benchmark foi desativado.

Objetivo:

- reduzir variabilidade experimental;
- garantir repetibilidade dos resultados.

---

# 5. Pré-processamento das Imagens

## 5.1 Resize

Todas as imagens foram redimensionadas para:

```python
224 x 224
```

Compatível com:

- DenseNet121;
- ImageNet pretrained weights.

---

## 5.2 Normalização

Foi utilizada normalização compatível com ImageNet:

```python
mean=[0.485, 0.456, 0.406]
std=[0.229, 0.224, 0.225]
```

Objetivos:

- estabilizar treino;
- acelerar convergência;
- compatibilidade com transfer learning.

---

# 6. Data Augmentation

Foi implementado um pipeline de data augmentation apenas no conjunto de treino.

Os conjuntos de validação e teste permaneceram sem augmentation.

---

## 6.1 Técnicas Utilizadas

### Random Horizontal Flip

```python
p=0.5
```

Objetivo:

- aumentar variabilidade espacial.

---

### Random Rotation

```python
degrees=10
```

Objetivo:

- aumentar robustez a pequenas rotações.

---

### Color Jitter

```python
brightness=0.10
contrast=0.10
```

Objetivo:

- aumentar robustez a variações de contraste.

---

### Random Affine

```python
translate=(0.03, 0.03)
scale=(0.95, 1.05)
```

Objetivo:

- aumentar robustez geométrica.

---

## 6.2 Técnicas Não Utilizadas

As seguintes técnicas NÃO foram utilizadas por poderem destruir a semântica física das imagens ELPV:

- vertical flip;
- rotações agressivas;
- blur forte;
- crops destrutivos.

---

# 7. Arquitetura DenseNet121

## 7.1 Transfer Learning

Foi utilizada DenseNet121 pré-treinada em ImageNet.

A utilização de transfer learning permitiu:

- acelerar convergência;
- melhorar generalização;
- reduzir necessidade de dados.

---

## 7.2 Camada Final

A camada classificadora final foi substituída por:

```python
Linear(in_features, 2)
```

para classificação binária.

---

## 7.3 Dropout

Foi introduzido dropout na cabeça classificadora.

Objetivo:

- reduzir overfitting;
- melhorar generalização.

---

# 8. Função de Loss

## 8.1 CrossEntropyLoss

Foi utilizada:

```python
CrossEntropyLoss
```

---

## 8.2 Weighted Loss

Foram utilizados pesos automáticos de classes.

Os pesos foram calculados dinamicamente através da distribuição das classes.

Objetivos:

- compensar desequilíbrio entre classes;
- melhorar recall da classe defect.

---

# 9. Otimizador

Foi utilizado:

```python
Adam
```

com:

```python
learning_rate = 1e-4
```

Motivos:

- rápida convergência;
- boa estabilidade;
- elevada eficácia em CNNs.

---

# 10. Learning Rate Scheduler

Foi utilizado:

```python
ReduceLROnPlateau
```

Parâmetros:

```python
factor = 0.5
patience = 2
```

Objetivo:

- reduzir learning rate quando a validação estabiliza.

Durante o treino observou-se:

```text
1e-4 → 5e-5 → 2.5e-5
```

O scheduler contribuiu para:

- refinamento final;
- melhoria da estabilidade.

---

# 11. Early Stopping

Foi implementado early stopping.

Objetivos:

- evitar overfitting;
- interromper treino sem melhorias.

O critério principal monitorizado foi:

```python
validation_f1
```

---

# 12. Métricas Utilizadas

## 12.1 Accuracy

Mede percentagem global de classificações corretas.

---

## 12.2 Precision

Mede a proporção de predições positivas corretas.

---

## 12.3 Recall

Mede capacidade de detetar defeitos reais.

Extremamente importante em inspeção industrial.

---

## 12.4 F1 Score

Média harmónica entre precision e recall.

Foi utilizada como principal métrica de otimização.

---

## 12.5 ROC AUC

Mede capacidade discriminativa global do classificador.

Resultado final:

```text
ROC AUC ≈ 0.90
```

Indicando forte separação entre classes.

---

# 13. Threshold Tuning

## 13.1 Problema

Inicialmente o sistema utilizava threshold implícito:

```python
0.50
```

No entanto, em problemas de defect detection:

- recall elevado é frequentemente mais importante;
- falsos negativos são mais críticos.

---

## 13.2 Threshold Analysis

Foi implementado um pipeline de análise de thresholds.

Thresholds analisados:

```python
0.30 → 0.70
```

---

## 13.3 Resultado Final

O threshold ótimo encontrado foi:

```python
0.30
```

Com melhoria significativa em:

- defect recall;
- F1 defect;
- equilíbrio geral.

---

# 14. Avaliação Final

## 14.1 Resultados Obtidos

Resultados finais com threshold otimizado:

| Métrica | Valor |
|---|---|
| Accuracy | 84.01% |
| Weighted F1 | 0.8406 |
| Macro F1 | 0.8374 |
| ROC AUC | 0.9007 |
| Recall defect | 0.8383 |
| F1 defect | 0.8163 |

---

## 14.2 Confusion Matrix

A confusion matrix final demonstrou:

- equilíbrio entre classes;
- boa capacidade de deteção;
- redução significativa de falsos negativos.

---

## 14.3 ROC Curve

A curva ROC demonstrou:

- elevada separação entre classes;
- forte capacidade discriminativa.

---

## 14.4 Precision-Recall Curve

A PR curve demonstrou:

- boa estabilidade;
- comportamento robusto;
- capacidade consistente de deteção de defeitos.

---

# 15. Sistema de Inferência

Foi desenvolvido um pipeline completo de inferência.

Capacidades:

- inferência individual;
- inferência em batch;
- processamento recursivo de diretórios;
- exportação CSV;
- imagens anotadas;
- confidence score.

---

# 16. Ferramentas e Bibliotecas Utilizadas

## Linguagem

- Python

---

## Frameworks e Bibliotecas

- PyTorch
- Torchvision
- Scikit-learn
- NumPy
- Pandas
- Matplotlib
- OpenCV
- tqdm

---

## Ambiente

- CUDA
- GPU NVIDIA
- Visual Studio Code

---

# 17. Organização do Projeto

O projeto foi estruturado modularmente.

Principais módulos:

```text
src/
    datasets/
    training/
    evaluation/
    inference/
    visualization/
    utils/
    models/
```

Objetivos:

- reutilização;
- manutenção;
- escalabilidade;
- clareza arquitetural.

---

# 18. Conclusões

O pipeline DenseNet121 desenvolvido demonstrou:

- forte capacidade de deteção de defeitos;
- boa generalização;
- elevada robustez;
- métricas consistentes.

O sistema final apresentou:

- ROC AUC ≈ 0.90;
- balanced behaviour;
- threshold otimizado;
- pipeline reprodutível.

Os resultados obtidos demonstram que a abordagem baseada em Deep Learning é adequada para inspeção automática de painéis fotovoltaicos utilizando imagens eletroluminescentes.

---

# 19. Trabalho Futuro

Como trabalho futuro poderão ser exploradas:

- EfficientNet-B0;
- ConvNeXt-Tiny;
- CLAHE preprocessing;
- mixed precision training;
- focal loss;
- ensemble models;
- explainability (Grad-CAM);
- classificação multi-classe.

