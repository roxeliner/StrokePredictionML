import requests

# Define the data to be sent
data = {
    "age": 90.0,
    "hypertension": 1,
    "heart_disease": 1,
    "avg_glucose_level": 200.5,
    "bmi": 28.9,
    "gender": "Male",
    "ever_married": "Yes",
    "work_type": "Private",
    "Residence_type": "Urban",
    "smoking_status": "smoked"
}

# Send a POST request
response = requests.post("https://strokepredictionapp-sqzplglaaq-nw.a.run.app/predict", json=data)

# Check if the request was successful
if response.status_code == 200:
    print("Prediction successful!")
    print("Response:", response.json())
else:
    print("Prediction failed!")
    print("Status Code:", response.status_code)
    print("Response:", response.text)
