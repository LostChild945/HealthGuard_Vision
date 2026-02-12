import os
from sqlalchemy import create_engine, event
from sqlalchemy.orm import declarative_base
from dotenv import load_dotenv

Base = declarative_base()

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")
SECRET_KEY = os.getenv("ENCRYPTION_KEY")

if not DATABASE_URL:
    raise ValueError("DATABASE_URL non définie dans le .env")

if not SECRET_KEY:
    raise ValueError("ENCRYPTION_KEY non définie dans le .env")

engine = create_engine(DATABASE_URL)

@event.listens_for(engine, "connect")
def set_encryption_key(dbapi_connection, connection_record):
    cursor = dbapi_connection.cursor()
    cursor.execute("SET my.encryption_key = %s", (SECRET_KEY,))
    cursor.close()

print("Engine PostgreSQL initialisé avec clé de chiffrement session.")