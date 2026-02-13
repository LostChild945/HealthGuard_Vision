from API.src.database.database import engine, Base, get_db
from API.src.middleware.middleware import middleware_api_key
from fastapi import FastAPI, UploadFile, Depends, HTTPException
from sqlalchemy.orm import Session
from API.src.models.image import Image
from API.src.models.analyse import Analyse
from ML.anemia.run_anemia import AnemiaAnalyzer

app = FastAPI(title="HealthGuard Vision API")

Base.metadata.create_all(bind=engine)

app.middleware("http")(middleware_api_key)

@app.post("/upload")
async def upload_image(
    files: list[UploadFile],
    db: Session = Depends(get_db)
):
    results = []
    total_sain = 0.0
    total_anemie = 0.0
    for file in files:
        if not file.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="Le fichier doit être une image")

        image_bytes = await file.read()
        image = Image(data=image_bytes)

        db.add(image)
        db.commit()
        db.refresh(image)

        anemia = AnemiaAnalyzer()
        result_anemia = anemia.analyze(image_bytes)
        results.append(result_anemia)
        result_anemia_sql = Analyse(result=result_anemia["result"], image_id=image.id)

        db.add(result_anemia_sql)
        db.commit()
        db.refresh(result_anemia_sql)

    for r in results:
        total_sain += r["probabilities"]["sain"]
        total_anemie += r["probabilities"]["anemie"]

    moyenne_sain = total_sain / len(results)
    moyenne_anemie = total_anemie / len(results)

    if moyenne_anemie > moyenne_sain:
        return {
            "result": "Anémie détectée"
        }
    else:
        return {
            "result": "Pas d'Anémie détectée",
        }
