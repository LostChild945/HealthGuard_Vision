# HealthGuard-api

Api pour **l'upload des images**, développée en **Python** avec **Fast-API**, **Uvicorn** et **SqlAlchemy**.

# 🚀 Fonctionnalités

- 🔐 Upload d’images vectorielles (via méthode POST)
- 🗄️ Gestion de la base de données (SQLAlchemy, Postgres SQL)
- 🌐 Gestion des routes API (FastAPI, Uvicorn)
- 🧪 Test Unitaire (pytest)

---

# 🧱 Stack technique

- Python (24.3.1)
- fastapi (0.128.5)
- psycopg2-binary (2.9.11)
- pydantic (2.12.5)
- python-dotenv (1.2.1)
- SqlAlchemy (2.0.46)
- uvicorn (0.40.0)
- pytest (9.0.2)
---

# 📁 Arborescence principale

````
API/
├── src/
│ ├── models/
│ │ ├── main.py
│ │ └──
│ ├── routes/
│ │ ├── main.py
│ │ └──
│ └── main.py
├── __test__/
│ ├──
├── README.md
└── requirements.txt

````
# ⚙️ Installation & lancement
## Prérequis
```
python -m venv venv
venv/Scripts/activate
```

```
pip install -r requirements.txt
```

## Lancement
```
python src/main.py
```
