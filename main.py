from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
from fastapi.middleware.cors import CORSMiddleware
from catboost import CatBoostClassifier
from ai_service import router as ai_router

app = FastAPI()

app.include_router(ai_router)

# ─────────────────────────────
# CORS
# ─────────────────────────────

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

model = joblib.load("catboost_model.pkl")

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
# PHASE PREDICTION
# ─────────────────────────────

@app.post("/predict_phases")
def predict_phases(data: InputData):

    # 🔥 DEBUG BLOCK (IMPORTANT)
    print("======================================")
    print("RAW is_flow:", data.is_flow)
    print("TYPE is_flow:", type(data.is_flow))
    print("======================================")

    is_flow_flag = int(data.is_flow) == 1
    print("NORMALIZED is_flow_flag:", is_flow_flag)
    print("======================================")

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

    # ─────────────────────────
     # FLOW OVERRIDE
    if int(data.is_flow) == 1:
        return {
            "menstrual": 1.0,
            "follicular": 0.0,
            "fertility": 0.0,
            "luteal": 0.0
        }
    # ─────────────────────────
    # BASE MODELS
    # ─────────────────────────

    p_fertility = fert_model.predict_proba(x)[0][1]
    p_follicular = fol_model.predict_proba(x)[0][1]
    p_luteal = lut_model.predict_proba(x)[0][1]

    x_final = np.array([[
        *x[0],
        p_fertility,
        p_follicular,
        p_luteal
    ]])

    proba = final_model.predict_proba(x_final)[0]

    return {
        "menstrual": 0.0,
        "follicular": float(proba[0]),
        "fertility": float(proba[1]),
        "luteal": float(proba[2])
    }