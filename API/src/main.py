from fastapi import FastAPI
from database.database import engine
from models.image import Base
from models.Analyse import Base

app = FastAPI(title="HealthGuard Vision API")

Base.metadata.create_all(bind=engine)

@app.get("/")
def root():
    return {"message": "API fonctionne 🚀"}