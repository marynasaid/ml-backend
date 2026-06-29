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

fert_model = CatBoostClassifier()
fert_model.load_model("fertility_binary.cbm")

fol_model = CatBoostClassifier()
fol_model.load_model("follicular_binary.cbm")

lut_model = CatBoostClassifier()
lut_model.load_model("luteal_binary.cbm")

final_model = CatBoostClassifier()
final_model.load_model("cycle_phase_model.cbm")


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
    avg_cycle_length: float
    cycle_length_deviation: float


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
# PHASE PREDICTION (FIXED PIPELINE)
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
        data.avg_cycle_length,
        data.cycle_length_deviation
    ]])

    # binary models
    p_fertility = fert_model.predict_proba(x)[0][1]
    p_follicular = fol_model.predict_proba(x)[0][1]
    p_luteal = lut_model.predict_proba(x)[0][1]

    # final stacked model
    x_final = np.array([[
        *x[0],
        p_fertility,
        p_follicular,
        p_luteal
    ]])

    proba = final_model.predict_proba(x_final)[0]

    return {
        "follicular": float(proba[0]),
        "fertility": float(proba[1]),
        "luteal": float(proba[2])
    }