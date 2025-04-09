from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import pandas as pd
from fastapi.middleware.cors import CORSMiddleware

# fastapi ve pydantic kütüphanesini araştıralım.

app = FastAPI(title="Hearth Disease Prediction API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Frontend adresi buraya!
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# .pkl dosyasına önceden kaydedilmiş modeli bu python dosyasına import et.
# artık aşağıdaki definitionlarda (fonksiyonlarda) modelimi kullanabilirim.
with open("logistic_model.pkl","rb") as f:
    model = pickle.load(f)

# kullanıcıdan isteyeceğim json veriyi pythonda temsil et.
class PatientData(BaseModel):
    Age: int
    Sex: int
    ChestPainType: int
    RestingBP: int
    Cholesterol: int
    FastingBS: int
    RestingECG: int
    MaxHR: int
    ExerciseAngina: int
    Oldpeak: float
    ST_Slope: int

@app.get("/")
def get():
    return { "Message":"Merhaba" }

@app.post("/predict")
def predict(data: PatientData):
    df = pd.DataFrame([data.dict()])
    prediction = model.predict(df)[0]
    return {"prediction":int(prediction)}