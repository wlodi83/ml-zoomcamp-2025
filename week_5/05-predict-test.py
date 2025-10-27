import requests

#API_URL = "http://localhost:9696/predict"
API_URL = 'http://churn-serving-env.eba-29vyifik.eu-central-1.elasticbeanstalk.com/predict'

customer_id = "xyz"
customer = {
    'gender': 'female',
    'seniorcitizen': 0,
    'partner': 'yes',
    'dependents': 'no',
    'phoneservice': 'no',
    'multiplelines': 'no_phone_service',
    'internetservice': 'dsl',
    'onlinesecurity': 'no',
    'onlinebackup': 'yes',
    'deviceprotection': 'no',
    'techsupport': 'no',
    'streamingtv': 'no',
    'streamingmovies': 'no',
    'contract': 'month-to-month',
    'paperlessbilling': 'yes',
    'paymentmethod': 'electronic_check',
    'tenure': 1,
    'monthlycharges': 29.85,
    'totalcharges': 29.85
}

response = requests.post(API_URL, json=customer).json()

print(response)

if response[0]["churn"] == True:
    print("sending promo offer to client id:", customer_id)
else:
    print("no offer sent to client id:", customer_id)