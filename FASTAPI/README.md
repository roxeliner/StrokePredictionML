Stroke Prediction API
This repository contains the implementation of a RESTful API for stroke prediction using FastAPI, XGBoost, and various preprocessing techniques.

Overview
The API accepts input data in JSON format, applies preprocessing and feature engineering, then uses a pre-trained XGBoost classifier to predict whether the input data indicates a high risk of stroke.

Setup
Requirements
To install the required dependencies, run the following command:

bash
Copy code
pip install -r requirements.txt
Make sure you have the following Python packages installed:

fastapi
pydantic
pandas
numpy
scikit-learn
xgboost
joblib
uvicorn
Also, ensure you have the preprocessor and model files (preprocessor.pkl and tuned_xgb_model.pkl) in your working directory.

Running the API
To start the API, run:

bash
Copy code
python filename.py
Replace filename.py with the name of your Python script. This will start the API on http://127.0.0.1:8000.

Usage
Endpoints
POST /predict: Accepts input data, applies preprocessing and feature engineering, then returns the prediction result.
Input Format
The input data should be a JSON object with the following fields:

age: float
hypertension: int (0 or 1)
heart_disease: int (0 or 1)
avg_glucose_level: float
bmi: float
gender: str
ever_married: str
work_type: str
Residence_type: str
smoking_status: str
Example:

json
Copy code
{
  "age": 67.0,
  "hypertension": 0,
  "heart_disease": 1,
  "avg_glucose_level": 228.69,
  "bmi": 36.6,
  "gender": "Male",
  "ever_married": "Yes",
  "work_type": "Private",
  "Residence_type": "Urban",
  "smoking_status": "formerly smoked"
}
Output Format
The API returns a JSON object with a single field prediction, which can be either "Yes" or "No".

Example:

json
Copy code
{
  "prediction": "Yes"
}
Development
The main components of the code include:

Model and Preprocessor Loading: The XGBoost model and the preprocessing pipeline are loaded from disk.
FastAPI App Initialization: A FastAPI app is created.
Data Models: Pydantic models are defined for input validation.
Endpoint Definition: A POST endpoint is defined for making predictions.
Preprocessing Functions: Functions for data preprocessing and feature engineering are implemented.
Main Function: If the script is run directly, the API server is started.
Make sure to handle and test any changes carefully, especially if you modify the preprocessing functions or the data models, as these are critical for the API’s correct operation.

Contributing
If you wish to contribute to this project, please fork the repository and submit a pull request. Any contributions would be greatly appreciated.

License
This project is open-source and available under the MIT License.