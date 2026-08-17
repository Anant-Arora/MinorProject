from sqlalchemy import Column, Integer, String, Float, Date
from app.database import Base

class Transaction(Base):
    __tablename__ = "transactions"

    id = Column(Integer, primary_key=True, index=True)
    description = Column(String)
    amount = Column(Float)
    category = Column(String, nullable=True)
    date = Column(Date)