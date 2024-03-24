from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import KNNImputer
from xgboost import XGBClassifier
import joblib

# Load the preprocessorsdock
preprocessor = joblib.load('preprocessor.pkl')


# Load the tuned XGBoost model
model = joblib.load("tuned_xgb_model.pkl")
# Define a FastAPI app
app = FastAPI()

# Define your input data model
class InputData(BaseModel):
    age: float
    hypertension: int
    heart_disease: int
    avg_glucose_level: float
    bmi: float
    gender: str
    ever_married: str
    work_type: str
    Residence_type: str
    smoking_status: str



@app.post("/predict")
async def make_prediction(input_data: InputData):
    # Convert input data to DataFrame

    input_data_dict = input_data.dict()
    df = pd.DataFrame([input_data_dict])
    print(f"input data : {df}")



    try:
        # Convert input data to DataFrame

        df = pd.DataFrame([input_data.dict()])
        print(f"try_df: {df}")



        # Apply preprocessing
        df = preprocess_data(df)


        # Create features
        df = create_features(df)

        # Apply feature transformation

        X_prepared = preprocessor.transform(df)


        # Make prediction
        predictions = model.predict(X_prepared)
        print(type(predictions))
        # Return the prediction result
        return {"prediction": "Yes" if int(predictions) == 1 else "No"}
    except Exception as e:


        raise HTTPException(status_code=500, detail=str(e))





def preprocess_data(df):

    df["hypertension"] = (
        df["hypertension"]
            .astype(str)
            .apply(lambda x: "No" if x == "0" else ("Yes" if x == "1" else "Unknown"))
    )

    df["heart_disease"] = (
        df["heart_disease"]
            .astype(str)
            .apply(lambda x: "No" if x == "0" else ("Yes" if x == "1" else "Unknown"))
    )

    # Handling Missing Value
    df.replace('N/A', np.nan, inplace=True)
    encoders = {}

    for col in df.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].fillna('missing').astype(str))
        encoders[col] = le


    imputer = KNNImputer(n_neighbors=5)
    df = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)

    for col, le in encoders.items():
        df[col] = le.inverse_transform(df[col].astype(int))

    return df

# Define a function to create features
def create_features(df):

    def age_group(age):

        age = np.array(age).flatten()

        bins = [0, 18, 35, 50, 65, 80, 100]
        labels = ['0-18', '19-35', '36-50', '51-65', '66-80', '81-100']
        value = pd.cut(age, bins=bins, labels=labels, right=False)
        result = value[0]
        return result

    def bmi_category(bmi):
        if bmi < 18.5:
            return "Underweight"
        elif 18.5 <= bmi < 24.9:
            return "Normal weight"
        elif 25 <= bmi < 29.9:
            return "Overweight"
        else:
            return "Obesity"

    def is_senior_citizen(age):
        return 1 if age >= 60 else 0

    def glucose_level_category(glucose):
        if glucose < 90:
            return "Low"
        elif 90 <= glucose <= 120:
            return "Normal"
        else:
            return "High"

    def marriage_work_type_interaction(ever_married, work_type):
        return f"{ever_married}_{work_type}"

    def gender_smoking_status_interaction(gender, smoking_status):
        return f"{gender}_{smoking_status}"

    def hypertension_heart_disease_interaction(hypertension, heart_disease):
        return f"{hypertension}_{heart_disease}"

    def bmi_heart_disease_interaction(bmi_category, heart_disease):
        return f"{bmi_category}_{heart_disease}"

    def glucose_level_hypertension_interaction(glucose_level_category, hypertension):
        return f"{glucose_level_category}_{hypertension}"

    def glucose_level_heart_disease_interaction(glucose_level_category, heart_disease):
        return f"{glucose_level_category}_{heart_disease}"

    def bmi_hypertension_interaction(bmi_category, hypertension):
        return f"{bmi_category}_{hypertension}"

    def hypertension_age_group_interaction(hypertension, age_group):
        return f"{hypertension}_{age_group}"

    def heart_disease_age_group_interaction(heart_disease, age_group):
        return f"{heart_disease}_{age_group}"

    print(f"inside_preprocess_data_before_age_group: {df}")
    df['age_group'] = df['age'].astype(int).apply(lambda x: age_group(x))
    print(f"inside_preprocess_data_after_age_group: {df}")
    df['bmi_category'] = df['bmi'].apply(bmi_category)
    print(df)
    df['is_senior_citizen'] = df['age'].apply(is_senior_citizen)
    print(df)
    df['glucose_level_category'] = df['avg_glucose_level'].apply(glucose_level_category)
    print(df)
    df['married_work_type'] = df.apply(lambda x: marriage_work_type_interaction(x['ever_married'], x['work_type']), axis=1)
    print(df)
    df['gender_smoking_status'] = df.apply(lambda x: gender_smoking_status_interaction(x['gender'], x['smoking_status']), axis=1)
    print(df)
    df['hypertension_heart_disease'] = df.apply(lambda x: hypertension_heart_disease_interaction(x['hypertension'], x['heart_disease']), axis=1)
    print(df)
    df['bmi_heart_disease'] = df.apply(lambda x: bmi_heart_disease_interaction(x['bmi_category'], x['heart_disease']), axis=1)
    print(df)
    df['glucose_level_hypertension'] = df.apply(lambda x: glucose_level_hypertension_interaction(x['glucose_level_category'], x['hypertension']), axis=1)
    print(df)
    df['glucose_level_heart_disease'] = df.apply(lambda x: glucose_level_heart_disease_interaction(x['glucose_level_category'], x['heart_disease']), axis=1)
    print(df)
    df['bmi_hypertension'] = df.apply(lambda x: bmi_hypertension_interaction(x['bmi_category'], x['hypertension']), axis=1)
    print(df)
    df['hypertension_age_group'] = df.apply(lambda x: hypertension_age_group_interaction(x['hypertension'], x['age_group']), axis=1)
    print(df)
    df['heart_disease_age_group'] = df.apply(lambda x: heart_disease_age_group_interaction(x['heart_disease'], x['age_group']), axis=1)
    print(df)
    return df





if __name__ == "__main__":

    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)