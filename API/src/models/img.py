from sqlalchemy import Column, Integer, DateTime
from sqlalchemy.orm import relationship
from pgvector.sqlalchemy import VECTOR
from datetime import datetime
from database.database import Base

class Image(Base):
    __tablename__ = "image"

    id = Column(Integer, primary_key=True, index=True)
    embedding = Column(VECTOR(512))  # ou VECTOR_DIM
    date = Column(DateTime, default=datetime.utcnow)

    # relation avec Analyse
    analyses = relationship("Analyse", back_populates="image", cascade="all, delete")

    def __repr__(self):
        return f"<Image id={self.id}>"
