from fastapi import FastAPI
from pydantic import BaseModel
import pickle

app = FastAPI()


class LiverInput(BaseModel):
    Age: int
    Gender: int   # 1 = Male, 0 = Female
    TB: float
    DB: float
    Alkphos: float
    Sgpt: float
    Sgot: float
    TP: float
    ALB: float
    A_G_Ratio: float


try:
    liver_model = pickle.load(open('liver_disease_model.pkl', 'rb'))
except Exception as e:
    liver_model = None
    print("MODEL LOAD ERROR:", e)

@app.get("/")
def home():
    return {"status": "Liver Disease API is running"}


@app.post("/liver_disease_prediction")
def predict(data: LiverInput):

    if liver_model is None:
        return {"error": "Model not loaded properly"}

    try:
        input_list = [
            data.Age,
            data.Gender,
            data.TB,
            data.DB,
            data.Alkphos,
            data.Sgpt,
            data.Sgot,
            data.TP,
            data.ALB,
            data.A_G_Ratio
        ]

        prediction = liver_model.predict([input_list])

        return {
            "prediction": int(prediction[0]),
            "result": "Liver Disease" if prediction[0] == 1 else "Healthy"
        }

    except Exception as e:
        return {"error": str(e)}
