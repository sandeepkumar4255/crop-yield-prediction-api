from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import pickle

# Create app
app = FastAPI(title="Crop Yield Prediction API")

# Load saved model and encoder
model = pickle.load(open("model.pkl", "rb"))
encoder = pickle.load(open("encoder.pkl", "rb"))

# Input schema
class CropInput(BaseModel):
    soil_type: str
    rainfall: float
    temperature: float
    fertilizer: float
    area: float
    humidity: float

# Home route
@app.get("/")
def home():
    return {"message": "Crop Yield Prediction API is working"}

# Prediction route
@app.post("/predict")
def predict(data: CropInput):

    # Encode soil type
    soil = encoder.transform([data.soil_type.capitalize()])[0]

    # Prepare input
    input_data = np.array([[soil,
                            data.rainfall,
                            data.temperature,
                            data.fertilizer,
                            data.area,
                            data.humidity]])

    # Predict
    prediction = model.predict(input_data)

    return {
        "predicted_yield_kg_per_acre": float(prediction[0])
    }