from fastapi import FastAPI
from pydantic import BaseModel

from utils.model import DepressionPredictionModel

# Простое REST API приложение на FastAPI для тестирования модели
app = FastAPI()

# Путь к модели
model_load_path = "model"

# Загрузка обученной модели
model = DepressionPredictionModel(model_load_path)


class PredictRequest(BaseModel):
    text: str


@app.post("/predict")
async def get_prediction(request: PredictRequest):
    text = request.text
    probability = round(model.get_prediction(text), 3)
    print(probability)

    return {
        "prediction": "suicidal" if probability >= 0.5 else "non-suicidal",
        "probability": round(abs(probability - 0.5) + 0.5, 3)
    }