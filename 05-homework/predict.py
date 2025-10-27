import pickle

from fastapi import FastAPI
import uvicorn

app = FastAPI(title="Lead Conversion Predictor")

with open('pipeline_v1.bin', 'rb') as f_in:
    pipeline = pickle.load(f_in)

datapoint = {
    "lead_source": "paid_ads",
    "number_of_courses_viewed": 2,
    "annual_income": 79276.0
}

y_pred = pipeline.predict_proba(datapoint)[0, 1]

# Question 3

print(f'The probability that this lead will convert is {y_pred}')
@app.post("/predict")
def predict(client: dict, model_version: str = 'v1'):
    print("Model version:", model_version)

    with open(f"pipeline_{model_version}.bin", 'rb') as f_in:
        pipeline = pickle.load(f_in)

    y_pred = pipeline.predict_proba(client)[0, 1]
    return {
        'conversion_probability': float(y_pred)
    }

if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=9696)