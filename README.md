# 🏥 HealthGuard_Vision

Plateforme médicale complète pour l'analyse d'images via intelligence artificielle. Détection d'anémie et analyse de lésions cutanées pour les professionnels de santé.

---

## 🎯 Vue d'ensemble

HealthGuard_Vision est composée de **3 modules intégrés** :

| Module               | Technologie         | Rôle                                              |
| -------------------- | ------------------- | ------------------------------------------------- |
| **Frontend**         | React Native + Expo | Application mobile pour capture et envoi d'images |
| **Backend**          | FastAPI + Python    | API de gestion des images et modèles ML           |
| **Machine Learning** | PyTorch             | 2 modèles d'IA pour diagnostic                    |

---

## 🚀 Fonctionnalités principales

### 📱 Application Mobile (HealGuard)

- Capture d'images via caméra
- Galerie de photos
- Envoi sécurisé à l'API
- Affichage des résultats d'analyse

### 🔬 Modèles ML

- **Anemia Detector** : Détection d'anémie via images de paupières (EfficientNet-B0, 92% accuracy)
- **Skin Lesion Analyzer** : Classification de 7 types de lésions cutanées (LLaVA-v1.5)

### 🌐 API Backend

- Upload d'images vectorielles
- Stockage PostgreSQL avec pgvector
- Routes FastAPI optimisées
- Tests unitaires

---

## 📁 Structure du projet

```
HealthGuard_Vision/
├── HealGuard/              # Frontend mobile (React Native/Expo)
├── API_HealthGuard/        # Backend API
│   ├── API/                # Service API (FastAPI)
│   ├── ML/                 # Modèles Machine Learning
│   │   ├── anemia/         # Modèle détection anémie
│   │   ├── skin_cancer/    # Modèle lésions cutanées
│   │   └── data/           # Datasets d'entraînement
│   └── POSTGRES/           # Configuration base de données
└── compose.yml             # Orchestration Docker
```

---

## 🛠️ Installation & Démarrage

### Prérequis

- Docker & Docker Compose
- GPU NVIDIA (recommandé pour ML)
- Python 3.9+
- Node.js 18+

### Démarrage avec Docker Compose

```bash
# Cloner le projet
git clone https://github.com/LostChild945/M1Proj.git
cd M1Proj

# Configurer les variables d'environnement
cp .env.example .env

# Démarrer tous les services
docker-compose up -d

# Services disponibles :
# - API: http://localhost:8000
# - PostgreSQL: localhost:5432
# - Adminer: http://localhost:8080
```

### Installation locale

#### Backend API

```bash
cd API_HealthGuard/API
python -m venv venv
source venv/bin/activate  # ou venv\Scripts\activate sur Windows
pip install -r requirements.txt
python -m uvicorn src.main:app --reload
```

#### Frontend Mobile

```bash
cd HealGuard
npm install
npx expo start
```

---

## 📚 Documentation détaillée

- [Documentation API](API_HealthGuard/API/README.md)
- [Documentation ML](API_HealthGuard/ML/README.md)
- [Documentation Frontend](HealGuard/README.md)
- [Lien gestion de projet](https://sleepy-cart-f71.notion.site/Gestion-de-projet-19d4d62f1fee804dbb4dd28dccbc7521?source=copy_link)

---

## 🔧 Stack technique

| Composant          | Technologies                              |
| ------------------ | ----------------------------------------- |
| **Backend**        | FastAPI, SQLAlchemy, PostgreSQL, pgvector |
| **ML**             | PyTorch, EfficientNet, LLaVA, timm        |
| **Frontend**       | React Native, Expo, TypeScript            |
| **Infrastructure** | Docker, Docker Compose, NVIDIA GPU        |

---

## 🤝 Contribution

Les pull requests sont bienvenues. Pour les changements majeurs, ouvrez d'abord une issue pour discuter des modifications.

## 📄 Licence

Ce projet est sous licence MIT.
