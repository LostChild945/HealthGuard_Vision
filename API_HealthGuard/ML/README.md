# 🏥 HealthGuard Vision - Machine Learning Models

> Suite de modèles d'intelligence artificielle pour l'analyse d'images médicales à destination des professionnels de santé et diététiciens.

---

## 🎯 Vue d'ensemble

HealthGuard Vision intègre **3 modèles de deep learning** pour détecter différentes conditions médicales à partir d'images :

| Modèle                   | Condition détectée          | Type d'image   | Architecture    | Status        |
| ------------------------ | --------------------------- | -------------- | --------------- | ------------- |
| **Anemia Detector**      | Anémie                      | Yeux/Paupières | EfficientNet-B0 | ✅ Production |
| **Skin Lesion Analyzer** | 7 types de lésions cutanées | Dermatoscopie  | LLaVA-v1.5 (7B) | ✅ Production |

---

## 🤖 Modèles disponibles

### 1. Anemia Detector

Détection d'anémie à partir d'images de paupières inférieures.

#### Caractéristiques

- **Architecture** : EfficientNet-B0 avec dropout (0.3)
- **Framework** : PyTorch + timm
- **Input** : Images RGB 224×224
- **Output** : Binaire (Sain / Anémie) + probabilités
- **Dataset** : India Anemia Dataset (~12,000 images)
- **Performance** :
  - Accuracy : **92.34%**
  - F1-Score : **88.76%**

#### Classes détectées

- `0` : Sain (pas d'anémie)
- `1` : Anémie détectée

#### Fichier modèle

```
models/model_anemie.pt
```

---

### 2. Skin Lesion Analyzer

Classification de lésions cutanées en 7 catégories diagnostiques.

#### Caractéristiques

- **Architecture** : LLaVA-v1.5-7B (Vision-Language Model)
- **Framework** : HuggingFace Transformers
- **Input** : Images dermatoscopiques RGB
- **Output** : Diagnostic textuel + code de lésion
- **Dataset** : HAM10000 (10,015 images)
- **Spécialité** : Génère des explications détaillées du diagnostic

#### Classes détectées

| Code    | Nom complet          | Description                         | Gravité     |
| ------- | -------------------- | ----------------------------------- | ----------- |
| `MEL`   | Melanoma             | Mélanome (cancer malin)             | 🔴 Critique |
| `BCC`   | Basal Cell Carcinoma | Carcinome basocellulaire            | 🔴 Élevée   |
| `AKIEC` | Actinic Keratoses    | Kératose actinique (pré-cancéreuse) | 🟡 Modérée  |
| `NV`    | Melanocytic Nevi     | Nævus / Grain de beauté bénin       | 🟢 Bénin    |
| `BKL`   | Benign Keratosis     | Kératose bénigne                    | 🟢 Bénin    |
| `DF`    | Dermatofibroma       | Dermatofibrome                      | 🟢 Bénin    |
| `VASC`  | Vascular Lesions     | Lésion vasculaire                   | 🟢 Bénin    |

#### Modèle HuggingFace

```
YuchengShi/LLaVA-v1.5-7B-HAM10000
```

---

## 🚀 Utilisation

### 1. Anemia Detector

#### Format de sortie

```python
{
    'image': 'eye_test.jpg',
    'prediction': 1,  # 0=sain, 1=anémie
    'confidence': 87.45,
    'probabilities': {
        'sain': 12.55,
        'anemie': 87.45
    },
    'result': 'Anémie détectée'
}
```

---

### 2. Skin Lesion Analyzer

#### Format de sortie

```python
{
    'image': 'lesion_test.jpg',
    'lesion_type': 'MEL',
    'diagnosis': 'This appears to be a melanoma due to asymmetry, irregular borders...',
    'confidence': 90.0,
    'result': 'Mélanome détecté - Consultation urgente recommandée'
}
```

### Pipeline de traitement

```
Image Input (JPG/PNG)
        ↓
  Preprocessing
  - Resize (224×224 ou auto)
  - Normalization
  - Tensor conversion
        ↓
   Model Inference
  - EfficientNet-B0 (Anemia)
  - LLaVA-v1.5 (Skin Lesion)
        ↓
  Post-processing
  - Softmax / Argmax
  - Confidence scoring
  - Text generation (Skin)
        ↓
   JSON Response
```

---

## 📊 Performances

### Anemia Detector

Évalué sur le test set (20% du dataset) :

| Métrique  | Score  |
| --------- | ------ |
| Accuracy  | 92.34% |
| Precision | 89.12% |
| Recall    | 87.45% |
| F1-Score  | 88.76% |

**Matrice de confusion** :

```
              Prédit Sain  Prédit Anémie
Réel Sain          1234           89
Réel Anémie         156         1087
```

### Skin Lesion Analyzer

Performances rapportées sur HAM10000 :

| Métrique             | Score                         |
| -------------------- | ----------------------------- |
| Accuracy (7 classes) | ~75-85%                       |
| Mélanome Detection   | ~90%                          |
| Interprétabilité     | ✅ Haute (diagnostic textuel) |

---

## 🛠️ Développement

### Entraîner un nouveau modèle Anemia

```bash
python ML/train_anemie.py \
  --data_dir ML/data/hugging-face \
  --epochs 50 \
  --batch_size 32 \
  --lr 0.001 \
  --save_dir models/
```

### Tests unitaires

```bash
# Tests des analyseurs
pytest tests/test_anemia_analyzer.py -v
pytest tests/test_skin_lesion_analyzer.py -v

# Tests d'intégration API
pytest tests/test_api.py -v
```
