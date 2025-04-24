from fastapi import FastAPI
from pydantic import BaseModel
import joblib  

app = FastAPI()

model = joblib.load('models/baseline_model.pkl') 
vectorizer = joblib.load('models/baseline_vectorizer.pkl')

class JobDescription(BaseModel):
    description: str

@app.post("/predict")
def predict(job: JobDescription):
    description = job.description
    vectorized_desc = vectorizer.transform([description])
    prediction = model.predict(vectorized_desc)
    fraudulent = int(prediction[0])
    print(f"Fraudulent: {fraudulent}")
    return {"fraudulent": fraudulent}