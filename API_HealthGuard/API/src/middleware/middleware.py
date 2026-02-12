from fastapi import Request
from fastapi.responses import JSONResponse
from dotenv import load_dotenv
import os

load_dotenv()
MY_API_KEY = os.getenv("CLE_API")

async def middleware_api_key(request: Request, call_next):
    client_key = (
        request.headers.get("x-api-key") or
        request.query_params.get("api_key") or
        request.query_params.get("key-api")
    )

    if client_key != MY_API_KEY:
        return JSONResponse(status_code=401, content={"message": "Autorisation refusée !"})

    return await call_next(request)

