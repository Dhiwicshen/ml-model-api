from fastapi import FastAPI, HTTPException
from app.schemas import PredictionRequest, PredictionResponse
from app.model import ml_model
from app.database import save_prediction, get_predictions
from datetime import datetime

app = FastAPI(
    title="ML Model API",
    description="Iris flower classification API with prediction logging",
    version="1.0.0"
)

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat()
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    try:
        features = [
            request.sepal_length,
            request.sepal_width,
            request.petal_length,
            request.petal_width
        ]
        result = ml_model.predict(features)

        # Log to PostgreSQL
        prediction_id = save_prediction(
            input_data=request.dict(),
            prediction=result["prediction"],
            class_name=result["class_name"],
            confidence=result["confidence"]
        )

        return PredictionResponse(
            prediction=result["prediction"],
            class_name=result["class_name"],
            confidence=result["confidence"],
            prediction_id=prediction_id
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/predictions")
async def get_all_predictions(limit: int = 10):
    return get_predictions(limit=limit)

@app.get("/")
async def root():
    return {"message": "ML Model API is running. Visit /docs for API reference."}