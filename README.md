# Stroke Prediction with Machine Learning

## Overview
This project focuses on developing a reliable machine learning model to predict the likelihood of strokes in patients, leveraging the Stroke Prediction Dataset. As part of Module 3's capstone project, it encompasses data preprocessing, exploratory data analysis (EDA), statistical inference, and the deployment of various ML models, including ensemble methods, to identify patients at high risk of stroke.

## Objectives
- Perform comprehensive EDA to understand the dataset's characteristics and uncover patterns.
- Apply statistical inference techniques to draw meaningful conclusions from the data.
- Evaluate and deploy multiple machine learning models to predict stroke occurrence effectively.
- Utilize model ensembling and hyperparameter tuning to improve prediction accuracy.

## Dataset
The dataset includes patient information like age, BMI, glucose levels, and more, aiming to predict stroke likelihood. Access the dataset [here](https://www.kaggle.com/fedesoriano/stroke-prediction-dataset).

## Analysis Highlights
- EDA revealed crucial insights into age, BMI, and glucose levels' roles in stroke risk.
- Statistical tests highlighted significant differences in health metrics between patients with and without stroke history.
- Ensemble models, particularly XGBoost, showed promising results in stroke prediction.

## Key Findings
- The XGBoost model, after tuning, emerged as the best performer with balanced accuracy, precision, and recall.
- Feature importance analysis via SHAP values indicated the critical predictors for stroke risk.

## Future Directions
- Further research could explore more sophisticated feature engineering, tackle class imbalance, and investigate model fairness to refine prediction capabilities.
- Continuous validation of the model with new patient data to ensure its reliability and effectiveness.

## How to Use
- Clone the repository and explore the Jupyter notebooks for detailed analysis and model implementation steps.
- Follow the instructions to deploy the model for real-time stroke prediction.

## Contributions
Feedback, suggestions, and contributions are welcome to enhance the project's scope and accuracy.

## Acknowledgements
Thanks to The Johns Hopkins Hospital for providing the context for this project and to all contributors of the Stroke Prediction Dataset.
