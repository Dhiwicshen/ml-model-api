from app.model import ml_model

def test_model_prediction_returns_valid_class():
    result = ml_model.predict([5.1, 3.5, 1.4, 0.2])
    assert result["prediction"] in [0, 1, 2]
    assert result["class_name"] in ["setosa", "versicolor", "virginica"]

def test_model_confidence_range():
    result = ml_model.predict([5.1, 3.5, 1.4, 0.2])
    assert 0.0 <= result["confidence"] <= 1.0