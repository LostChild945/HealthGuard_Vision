from fastapi import FastAPI
from database.database import engine
from models.image import Base
from models.Analyse import Base
from middleware.middleware import middleware_api_key

app = FastAPI(title="HealthGuard Vision API")

Base.metadata.create_all(bind=engine)

app.middleware("http")(middleware_api_key)

@app.get("/update")
async def update_route():
    return {"message": "Accés vers l'API autorisé."}
