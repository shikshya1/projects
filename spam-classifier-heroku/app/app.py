from fastapi import FastAPI
from pydantic import BaseModel
from model.model import predict_label
import uvicorn


app = FastAPI()


class InputText(BaseModel):
    text: str


@app.get("/")
def home():
    return {"health_check": "OK"}


@app.post("/predict")
def predict(userinput: InputText):
    prediction_text = predict_label(userinput.text)
    return (prediction_text)

if __name__ == '__main__':
    uvicorn.run(app, port=8080, host='0.0.0.0')