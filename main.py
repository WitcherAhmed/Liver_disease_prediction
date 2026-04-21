from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import json

app = FastAPI()

# Define input schema based on your dataset
class LiverInput(BaseModel):
    Age: int
    Gender: int   # Encode: Male=1, Female=0 (or whatever you used)
    TB: float
    DB: float
    Alkphos: float
    Sgpt: float
    Sgot: float
    TP: float
    ALB: float
    A_G_Ratio: float   # renamed because "/" is invalid in variable names

# Load your trained model
liver_model = pickle.load(open('E:/Work/Python/Liver_pred_API/liver_disease_model.pkl', 'rb'))

@app.post('/liver_disease_prediction')
def liver_pred(input_parameters: LiverInput):
    
    input_data = input_parameters.json()  # type: ignore
    input_dictionary = json.loads(input_data)

    age = input_dictionary['Age']
    gender = input_dictionary['Gender']
    tb = input_dictionary['TB']
    db = input_dictionary['DB']
    alkphos = input_dictionary['Alkphos']
    sgpt = input_dictionary['Sgpt']
    sgot = input_dictionary['Sgot']
    tp = input_dictionary['TP']
    alb = input_dictionary['ALB']
    ag_ratio = input_dictionary['A_G_Ratio']

    # IMPORTANT: match the exact order used during training
    input_list = [age, gender, tb, db, alkphos, sgpt, sgot, tp, alb, ag_ratio]

    prediction = liver_model.predict([input_list])

    if prediction[0] == 1:
        return {"result": "The person has liver disease"}
    else:
        return {"result": "The person is healthy"}