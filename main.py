from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
from fastapi.middleware.cors import CORSMiddleware
from catboost import CatBoostClassifier

app = FastAPI()

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─────────────────────────────
# LOAD MODELS
# ─────────────────────────────
model = joblib.load("catboost_model.pkl")  # next period

phase_model = CatBoostClassifier()
phase_model.load_model("cycle_phase_model.cbm")  # phases model


# ─────────────────────────────
# INPUT MODEL
# ─────────────────────────────
class InputData(BaseModel):
    is_flow: int
    is_first_flow: int
    last_cycle_length: float
    cycle_progress: float
    is_last_flow: int
    flow_length: float
    cycle_length_deviation: float
    avg_cycle_length: float


# ─────────────────────────────
# ROOT
# ─────────────────────────────
@app.get("/")
def root():
    return {"status": "ok"}


# ─────────────────────────────
# NEXT PERIOD
# ─────────────────────────────
@app.post("/predict")
def predict(data: InputData):

    x = np.array([[
        data.is_flow,
        data.is_first_flow,
        data.last_cycle_length,
        data.cycle_progress,
        data.is_last_flow,
        data.flow_length,
        data.cycle_length_deviation,
        data.avg_cycle_length
    ]])

    pred = model.predict(x)[0]

    return {"prediction": float(pred)}


# ─────────────────────────────
# PHASE PROBABILITIES (SAFE VERSION)
# ─────────────────────────────
@app.post("/predict_phases")
def predict_phases(data: InputData):

    x = np.array([[
        data.is_flow,
        data.is_first_flow,
        data.last_cycle_length,
        data.cycle_progress,
        data.is_last_flow,
        data.flow_length,
        data.cycle_length_deviation,
        data.avg_cycle_length
    ]])

    proba = phase_model.predict_proba(x)[0]

    # 🔥 ВАЖНО: НЕ ДЕЛАЕМ ПРЕДПОЛОЖЕНИЯ О ПОРЯДКЕ КЛАССОВ
    return {
        "raw": [float(p) for p in proba]
    }