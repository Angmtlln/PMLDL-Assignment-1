from fastapi import FastAPI
import joblib
import uvicorn
import pandas as pd

app = FastAPI()


with open('/api/model.pkl', 'rb') as f:
    model = joblib.load(f)


@app.post('/predict')
def predict(features: dict):
    X = pd.DataFrame([features["features"]])
    prediction = model.predict(X)
    print(prediction)
    return {'prediction': int(prediction[0])}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port = 3500)

