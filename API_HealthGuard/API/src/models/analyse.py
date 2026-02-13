from sqlalchemy import Column, Integer, DateTime, String, ForeignKey
from sqlalchemy.orm import relationship
from datetime import datetime
from API.src.database.database import Base

class Analyse(Base):
    __tablename__ = "analyse"

    id = Column(Integer, primary_key=True, index=True)
    result = Column(String)
    image_id = Column(Integer, ForeignKey("image.id", ondelete="CASCADE"))
    date = Column(DateTime, default=datetime.utcnow)

    image = relationship("Image", back_populates="analyses")

    def __repr__(self):
        return f"<Analyse id={self.id} image_id={self.image_id}>"
