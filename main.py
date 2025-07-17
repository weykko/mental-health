from fastapi import FastAPI
from pydantic import BaseModel

from utils.model import DepressionPredictionModel

# Инициализация FastAPI
app = FastAPI()

model_load_path = "model"
model = DepressionPredictionModel(model_load_path)

# Модель для входных данных
class PredictRequest(BaseModel):
    text: str

@app.post("/predict")
async def get_prediction(request: PredictRequest):
    text = request.text
    probability = model.get_prediction(text)

    return {
        'prediction': 'Suicidal' if round(probability) == 1 else 'Non-suicidal',
        'probability': probability if round(probability) == 1 else 1 - probability
    }