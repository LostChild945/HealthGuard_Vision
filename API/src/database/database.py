# Import SQLAlchemy's create_engine function
import os
from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base
from dotenv import load_dotenv

Base = declarative_base()

load_dotenv()

DATABASE_URL = str(os.getenv("DATABASE_URL", "test.fr"))

engine = create_engine(DATABASE_URL)

connection = engine.connect()
print("Connection a la base PostgreSQL réussi !")
connection.close()
