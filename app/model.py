import joblib
import numpy as np
import os

CLASS_NAMES = ["setosa", "versicolor", "virginica"]

class MLModel:
    def __init__(self):
        model_path = os.getenv("MODEL_PATH", "ml/model.pkl")
        self.model = joblib.load(model_path)

    def predict(self, features: list) -> dict:
        input_array = np.array(features).reshape(1, -1)
        prediction = int(self.model.predict(input_array)[0])
        probabilities = self.model.predict_proba(input_array)[0]
        confidence = float(probabilities[prediction])
        return {
            "prediction": prediction,
            "class_name": CLASS_NAMES[prediction],
            "confidence": round(confidence, 4)
        }

# Singleton instance
ml_model = MLModel()