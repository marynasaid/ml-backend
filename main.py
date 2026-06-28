from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# ✅ CORS ДОЛЖЕН БЫТЬ СРАЗУ ПОСЛЕ app
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# загрузка модели
model = joblib.load("catboost_model.pkl")


class InputData(BaseModel):
    is_flow: int
    is_first_flow: int
    last_cycle_length: float
    cycle_progress: float
    is_last_flow: int
    flow_length: float
    cycle_length_deviation: float
    avg_cycle_length: float


@app.get("/")
def root():
    return {"status": "ok"}


@app.post("/predict")
def predict(data: InputData):

    try:
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

    except Exception as e:
        return {"error": str(e)}