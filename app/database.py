from sqlalchemy import create_engine, Column, Integer, Float, String, DateTime, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime
import os
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(bind=engine)
Base = declarative_base()

class PredictionLog(Base):
    __tablename__ = "prediction_logs"
    id = Column(Integer, primary_key=True, index=True)
    input_data = Column(JSON)
    prediction = Column(Integer)
    class_name = Column(String)
    confidence = Column(Float)
    created_at = Column(DateTime, default=datetime.utcnow)

# Create table
Base.metadata.create_all(bind=engine)

def save_prediction(input_data, prediction, class_name, confidence):
    db = SessionLocal()
    try:
        log = PredictionLog(
            input_data=input_data,
            prediction=prediction,
            class_name=class_name,
            confidence=confidence
        )
        db.add(log)
        db.commit()
        db.refresh(log)
        return log.id
    finally:
        db.close()

def get_predictions(limit: int = 10):
    db = SessionLocal()
    try:
        return db.query(PredictionLog)\
                 .order_by(PredictionLog.created_at.desc())\
                 .limit(limit).all()
    finally:
        db.close()