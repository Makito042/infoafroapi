from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, conint, confloat
import joblib
import numpy as np
import pandas as pd

# Load model and preprocessing objects
model = joblib.load('models/best_inflation_model.pkl')
scaler = joblib.load('models/inflation_scaler.pkl')
features = joblib.load('models/inflation_features.pkl')
le_country = joblib.load('models/inflation_le_country.pkl')

# FastAPI app
app = FastAPI()

# Enable CORS (allow all origins for demo; restrict for production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change to your frontend domain in production!
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    return {"message": "Welcome to the African Inflation Prediction API. Visit /docs for documentation."}


# Define Pydantic input model with data types and constraints
class InflationInput(BaseModel):
    country: str
    year: conint(ge=1800, le=2100)
    systemic_crisis: conint(ge=0, le=1)
    exch_usd: confloat(ge=0)
    domestic_debt_in_default: conint(ge=0, le=1)
    sovereign_external_debt_default: conint(ge=0, le=1)
    gdp_weighted_default: confloat(ge=0)
    independence: conint(ge=0, le=1)
    currency_crises: conint(ge=0, le=1)
    inflation_crises: conint(ge=0, le=1)

@app.post("/predict")
def predict_inflation(input: InflationInput):
    # Encode country
    try:
        country_enc = le_country.transform([input.country])[0]
    except Exception:
        raise HTTPException(status_code=400, detail="Unknown country. Allowed: " + ", ".join(le_country.classes_))
    
    # Prepare feature vector
    input_dict = {
        'country_enc': country_enc,
        'year': input.year,
        'systemic_crisis': input.systemic_crisis,
        'exch_usd': input.exch_usd,
        'domestic_debt_in_default': input.domestic_debt_in_default,
        'sovereign_external_debt_default': input.sovereign_external_debt_default,
        'gdp_weighted_default': input.gdp_weighted_default,
        'independence': input.independence,
        'currency_crises': input.currency_crises,
        'inflation_crises': input.inflation_crises
    }
    X = pd.DataFrame([input_dict])[features]
    X_scaled = scaler.transform(X)
    pred_log = model.predict(X_scaled)[0]
    pred = float(np.expm1(pred_log))  # Convert log1p back to original scale

    return {"predicted_inflation": pred}