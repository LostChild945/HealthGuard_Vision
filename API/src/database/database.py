# Import SQLAlchemy's create_engine function
import os
from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base
Base = declarative_base()
from dotenv import load_dotenv

# Charger le .env
load_dotenv()

DATABASE_URL = str(os.getenv("DATABASE_URL", "test.fr"))

# Create a connection string
engine = create_engine(DATABASE_URL)

# Test the connection
connection = engine.connect()
print("Connection a la base PostgreSQL réussi !")
connection.close()

Base = declarative_base()