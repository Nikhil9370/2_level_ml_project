import os
import pytest
from src.predict import predict_age

def test_model_file_exists():
    assert os.path.exists("age_model.pkl")

def test_predict_method():
    from src import predict
    model = predict_age
    assert callable(model)

def test_age_predictions():
    assert predict_age(20) == "Adult"
    assert predict_age(10) == "Minor"
