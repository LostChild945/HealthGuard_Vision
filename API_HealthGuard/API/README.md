# 🌐 HealthGuard API

**Service backend pour l'analyse d'images médicales** développé en **Python** avec **FastAPI**, permettant l'upload, le stockage vectorisé et la gestion des analyses via modèles ML.

---

## 🚀 Fonctionnalités

- 📸 **Upload d'images** : Téléchargement sécurisé d'images médicales via POST
- 🔬 **Vectorisation** : Conversion d'images en embeddings vectoriels pour recherche sémantique
- 🗄️ **Base de données** : Stockage PostgreSQL avec pgvector pour indexation performante
- 🤖 **Intégration ML** : Connexion aux modèles d'IA (Anemia Detector, Skin Lesion Analyzer)
- ✅ **Validation** : Validation des données avec Pydantic
- 🧪 **Tests** : Suite de tests unitaires avec pytest
- 🔒 **Middleware** : Gestion sécurisée des requêtes et authentification

---

## 🧱 Stack technique

| Dépendance      | Version | Usage                     |
| --------------- | ------- | ------------------------- |
| Python          | 3.9+    | Runtime                   |
| FastAPI         | 0.128.5 | Framework web/API         |
| Uvicorn         | 0.40.0  | Serveur ASGI              |
| SQLAlchemy      | 2.0.46  | ORM base de données       |
| psycopg2-binary | 2.9.11  | Driver PostgreSQL         |
| Pydantic        | 2.12.5  | Validation schémas        |
| python-dotenv   | 1.2.1   | Variables d'environnement |
| pytest          | 9.0.2   | Framework testing         |

---

## 📁 Structure du projet

```
API/
├── src/
│   ├── main.py                 # Point d'entrée FastAPI
│   ├── database/
│   │   └── database.py         # Configuration connexion BD
│   ├── middleware/
│   │   └── middleware.py       # Middleware personnalisée
│   ├── models/
│   │   ├── analyse.py          # Modèle Analyse (ORM)
│   │   └── image.py            # Modèle Image (ORM)
│   └── schema/
│       ├── analyse_schema.py   # Schémas Analyse (Pydantic)
│       └── image_schema.py     # Schémas Image (Pydantic)
├── __test__/                   # Tests unitaires
├── requirements.txt            # Dépendances
├── Dockerfile                  # Configuration Docker
└── README.md
```

---

## ⚙️ Installation & Lancement

### Prérequis

- Python 3.9 ou supérieur
- PostgreSQL 12+
- pip ou conda

### 1️⃣ Installation locale

**Créer l'environnement virtuel :**

```bash
python -m venv venv
```

**Activer l'environnement :**

```bash
# Sur Linux/macOS
source venv/bin/activate

# Sur Windows
venv\Scripts\activate
```

**Installer les dépendances :**

```bash
pip install -r requirements.txt
```

### 2️⃣ Configuration

**Créer un fichier `.env` :**

```env
DATABASE_URL=postgresql://user:password@localhost:5432/healthguard
API_PORT=8000
LOG_LEVEL=info
```

### 3️⃣ Lancement du serveur

**Mode développement (avec rechargement automatique) :**

```bash
python -m uvicorn src.main:app --reload --host 0.0.0.0 --port 8000
```

**Mode production :**

```bash
python -m uvicorn src.main:app --host 0.0.0.0 --port 8000 --workers 4
```

L'API sera accessible sur **http://localhost:8000**

### 4️⃣ Accéder à la documentation interactive

- **Swagger UI** : http://localhost:8000/docs
- **ReDoc** : http://localhost:8000/redoc

---

## 🐳 Lancement avec Docker

```bash
# Build l'image
docker build -t healthguard-api .

# Lancer le conteneur
docker run -p 8000:8000 --env-file .env healthguard-api
```

---

## 🔗 Endpoints principaux

### Images

- `POST /api/images/upload` - Télécharger une image
- `GET /api/images/{id}` - Récupérer une image
- `DELETE /api/images/{id}` - Supprimer une image

### Analyses

- `POST /api/analyses` - Créer une analyse
- `GET /api/analyses/{id}` - Récupérer les résultats
- `GET /api/analyses` - Lister les analyses

---

## 🧪 Tests

**Exécuter tous les tests :**

```bash
pytest
```

**Tests avec couverture :**

```bash
pytest --cov=src
```

**Tests spécifiques :**

```bash
pytest __test__/test_images.py -v
```

---

## 📚 Documentation supplémentaire

- Voir le fichier [main.py](src/main.py) pour la configuration globale
- Voir [database.py](src/database/database.py) pour la connexion BD
- Voir [models/](src/models/) pour les modèles de données

---
