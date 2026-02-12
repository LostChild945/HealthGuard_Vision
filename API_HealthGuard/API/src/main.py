from database.database import engine, Base
from middleware.middleware import middleware_api_key
from fastapi import FastAPI, UploadFile, File, Depends, HTTPException
from sqlalchemy.orm import Session
from database.database import get_db
from models.image import Image
from models.analyse import Analyse

app = FastAPI(title="HealthGuard Vision API")

Base.metadata.create_all(bind=engine)

app.middleware("http")(middleware_api_key)

@app.post("/upload")
async def upload_image(
    file: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Le fichier doit être une image")

    image_bytes = await file.read()

    image = Image(data=image_bytes)

    db.add(image)
    db.commit()
    db.refresh(image)

    return {
        "message": "Image enregistrée avec succès",
        "image_id": image.id
    }