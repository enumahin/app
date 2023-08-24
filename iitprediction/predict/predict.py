import joblib
import logging
import pandas as pd
from sklearn.preprocessing import LabelEncoder

LOG_LEVEL = logging.INFO

LOG_FILENAME = "prediction.log"

logging.basicConfig(
    level=LOG_LEVEL,
    force=True,
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.FileHandler(LOG_FILENAME, "a", "utf-8"),
              logging.StreamHandler()]
)


# Predict Status_at_18th_Month
def predict_status(data):
    data = pd.read_excel(data)
    ltfu_data = data.drop(["regimen_Line_At_Start", "regimen_At_Start", "regimen_Switch","drug_Duration", "inh_Start_to_Now", "regimen_Dispensed", "current_Regimen_Line"], axis=1)
    for column in ltfu_data.select_dtypes(include='object').columns:
        le = LabelEncoder()
        ltfu_data[column] = le.fit_transform(ltfu_data[column].astype(str))
    clf = joblib.load("random_forest_model_status_at_18th_month.pkl")
    print(ltfu_data)
    return clf.predict(ltfu_data[:1])


# Predict Number of Visit Before Defaulting
def predict_number_of_visits(data):
    data = pd.read_excel(data)
    if data.regimen_Line_At_Start.empty or data.regimen_At_Start.empty or data.regimen_Switch.empty or data.drug_Duration.empty:
        return ['Incomplete Data']
    tv_data = data # .drop(["Total_Visits","status_at_18_months"], axis=1)
    for column in tv_data.select_dtypes(include='object').columns:
        le = LabelEncoder()
        tv_data[column] = le.fit_transform(tv_data[column].astype(str))
    clf = joblib.load("random_forest_model_number_of_visit_before_interruption.pkl")
    print(tv_data)
    return clf.predict(tv_data[:1])