import requests

API_URL = 'http://localhost:9697/predict'

model_version = 'v2'

client = {
    "lead_source": "organic_search",
    "number_of_courses_viewed": 4,
    "annual_income": 80304.0
}

response = requests.post(API_URL, json=client, params={'model_version': model_version}).json()

# Question 4
print(response)