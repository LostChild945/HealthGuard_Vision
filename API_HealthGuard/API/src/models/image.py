from sqlalchemy import Column, Integer, DateTime, LargeBinary
from sqlalchemy.orm import relationship
from datetime import datetime
from API.src.database.database import Base

class Image(Base):
    __tablename__ = "image"

    id = Column(Integer, primary_key=True, index=True)
    data = Column(LargeBinary, nullable=False)
    date = Column(DateTime, default=datetime.utcnow)

    analyses = relationship("Analyse", back_populates="image", cascade="all, delete")

    def __repr__(self):
        return f"<Image id={self.id}>"


