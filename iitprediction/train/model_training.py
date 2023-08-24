import logging
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
import random
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV
import seaborn as sns
import matplotlib.pyplot as plt

from time import sleep


SEED = 1

LOG_LEVEL = logging.INFO

LOG_FILENAME = "training.log"

logging.basicConfig(
    level=LOG_LEVEL,
    force=True,
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.FileHandler(LOG_FILENAME, "a", "utf-8"),
              logging.StreamHandler()]
)

# Reading dataset file
# df = pd.read_excel("real_data_app_enumah.xlsx")


def train(linelist):

    logging.info("# Reading dataset file")

    df = pd.read_excel(linelist)
    logging.info(df)

    # Saving a copy of original file
    data = df.copy(deep=True)

    data["careEntryPoint"] = data["careEntryPoint"].replace("Outreach-program","Outreach")
    data["careEntryPoint"] = data["careEntryPoint"].replace("COMMUNITY TESTING","COMMUNITY")
    data["careEntryPoint"] = data["careEntryPoint"].replace("community","COMMUNITY")
    data["careEntryPoint"] = data["careEntryPoint"].replace("COMMUNITY TESTERS","COMMUNITY")
    data["careEntryPoint"] = data["careEntryPoint"].replace("COMMUNITY TESTER","COMMUNITY")
    data["careEntryPoint"] = data["careEntryPoint"].replace("Community-ART","COMMUNITY")
    data["careEntryPoint"] = data["careEntryPoint"].replace("Community","COMMUNITY")
    data["careEntryPoint"] = data["careEntryPoint"].replace("ct","COMMUNITY")
    data["careEntryPoint"] = data["careEntryPoint"].replace("CB","CBO")
    data["careEntryPoint"] = data["careEntryPoint"].replace("cbo","CBO")
    data["careEntryPoint"] = data["careEntryPoint"].replace("CB O","CBO")
    data["careEntryPoint"] = data["careEntryPoint"].replace("cb","CBO")
    data["careEntryPoint"] = data["careEntryPoint"].replace("CBO TESTING","CBO")
    data["careEntryPoint"] = data["careEntryPoint"].replace("CBO KWALE","CBO")
    data["careEntryPoint"] = data["careEntryPoint"].replace("CB0","CBO")
    data["careEntryPoint"] = data["careEntryPoint"].replace("COB","CBO")
    data["careEntryPoint"] = data["careEntryPoint"].replace("SPOKE","SPOKE SITE")
    data["careEntryPoint"] = data["careEntryPoint"].replace("SPOKES","SPOKE SITE")
    data["careEntryPoint"] = data["careEntryPoint"].replace("SPOKE TESTING","SPOKE SITE")
    data["careEntryPoint"] = data["careEntryPoint"].replace("SPOKE INDEX","SPOKE SITE")
    data["careEntryPoint"] = data["careEntryPoint"].replace("ANC/PMTCT","ANC")


    logging.info("Use weight and Height to Calculate BMI")

    def calc_bmi(row):
        weight = row['weight']
        height = row['height']
        if np.isnan(weight) or np.isnan(height):
            # return "invalid"
            return np.random.choice(["underweight", "healthy"])
        bmi = weight / (height / 100) / (height / 100);
        if bmi < 18.5:
            return "underweight"
        if 18.5 <= bmi <= 24.9:
            return "healthy"
        if 25 <= bmi <= 29.9:
            return "overweight"
        if 30 <= bmi <= 39.9:
            return "obesity"
        if bmi >= 40:
            return "severe obesity"

    data["bmi"] = data.apply(calc_bmi, axis=1)

    logging.info("Remove unnecessary columns such as IDs, dates etc")
    data = data.drop(['defaultID', 'patientID', 'facilityID', 'art_Start_Date', 'patientLga',
                      'last_Drug_Pickup_Date', 'facilityLgaID', 'number_of_Days_Missed_Appointment', "status_at", "current_Age","weight","height"], axis = 1)

    logging.info("Iterate over each column in the DataFrame")
    for column in data.columns:
        logging.info("Check if the column is of categorical data type")
        if data[column].dtype == 'object':
            logging.info("Print the column name")
            logging.info(f"Column: {column}")
            logging.info("Print the unique values in the column")
            logging.info(data[column].unique())
            logging.info("")

    logging.info("Cleaning the categorical columns")
    data['status_at_18_months'] = data['status_at_18_months'].replace('Inactive', 'InActive')
    data['regimen_At_Start'] = data['regimen_At_Start'].str.replace('/r', '')
    data['current_Regimen_Line'] = data['current_Regimen_Line'].str.replace('/r', '')
    data['regimen_Dispensed'] = data['regimen_Dispensed'].str.replace('/r', '')

    logging.info("Removing Invalid age groups from the data")
    valid_age_groups = ['25 - 29', '30 - 34', '35 - 39', '20 - 24', '40 - 44', '15 - 19', '45 - 49','>=50', 'nan']
    data = data[data['age_Group'].isin(valid_age_groups)]

    logging.info("Remove Invalid Entry Point")
    valid_cep = ["OPD","Outreach", "CBO","VCT","COMMUNITY","ANC","Other","L&D","TB","SPOKE SITE","PMTCT"]
    data = data[data['careEntryPoint'].isin(valid_cep)]

    logging.info("Separate numeric and categorical columns")
    numeric_cols = data.select_dtypes(include=np.number).columns
    categorical_cols = data.select_dtypes(exclude=np.number).columns

    logging.info("Impute missing values in numeric columns with mean")
    numeric_imputer = SimpleImputer(strategy='mean')
    imputed_numeric = numeric_imputer.fit_transform(data[numeric_cols])

    logging.info("Impute missing values in categorical columns with most frequent")
    categorical_imputer = SimpleImputer(strategy='most_frequent')
    imputed_categorical = categorical_imputer.fit_transform(data[categorical_cols])

    logging.info("Create a DataFrame with imputed numeric columns")
    imputed_df_num = pd.DataFrame(imputed_numeric, columns=numeric_cols)

    logging.info("Create a DataFrame with imputed categorical columns")
    imputed_df_cat = pd.DataFrame(imputed_categorical, columns=categorical_cols)

    logging.info("Concatenate imputed numeric columns with categorical columns")
    data_imputed = pd.concat([imputed_df_num, imputed_df_cat], axis=1)

    logging.info("Converting current status to two possible categories that are active and inactive")
    data_imputed['status_at_18_months'] = data_imputed['status_at_18_months'].replace('Missed Appointment', 'Active')
    data_imputed['status_at_18_months'] = data_imputed['status_at_18_months'].replace('Restarted', 'Active')
    data_imputed['status_at_18_months'] = data_imputed['status_at_18_months'].replace('Active', 'Active')
    data_imputed['status_at_18_months'] = data_imputed['status_at_18_months'].replace('Returned to care', 'Active')
    data_imputed['status_at_18_months'] = data_imputed['status_at_18_months'].replace('Death', 'InActive')
    data_imputed['status_at_18_months'] = data_imputed['status_at_18_months'].replace('Discontinued Care', 'InActive')
    data_imputed['status_at_18_months'] = data_imputed['status_at_18_months'].replace('Transferred out', 'InActive')
    data_imputed['status_at_18_months'] = data_imputed['status_at_18_months'].replace('LTFU', 'InActive')

    logging.info("Calculate z-scores for each numerical column")
    z_scores = data_imputed.select_dtypes(include=np.number).apply(lambda x: np.abs((x - x.mean()) / x.std()))

    logging.info("Define a threshold for outlier detection (e.g., z-score > 3)")
    threshold = 3

    logging.info("Remove rows with outliers in numerical columns")
    df_no_outliers = data_imputed[(z_scores < threshold).all(axis=1)]

    logging.info("Check final shape of data")
    logging.info(df_no_outliers.shape)

    logging.info("Copy out data for the second model")
    df_model_2 = df_no_outliers.copy(deep=True)
    df_no_outliers.to_csv("Final_Cleaned_Data.csv")
    logging.info("X_current_Status = df_no_outliers[['current_Status']]")
    X_status_at_18_months = df_no_outliers[['status_at_18_months']]
    df_no_outliers = df_no_outliers.drop(['status_at_18_months', 'Total_Visits'], axis= 1)

    logging.info("Label encoding all numeric columns")
    logging.info("Iterate over each column in the DataFrame")
    for column in df_no_outliers.select_dtypes(include='object').columns:
        le = LabelEncoder()
        df_no_outliers[column] = le.fit_transform(df_no_outliers[column].astype(str))

    ## Machine Learning Algorithms

    ### Status at 18th Month in Treatment Prediction
    logging.info("Training model for Status at 18th Month in Treatment Prediction")
    ltfu_data = df_no_outliers.drop(["regimen_Line_At_Start", "regimen_At_Start", "regimen_Switch","drug_Duration", "inh_Start_to_Now", "regimen_Dispensed", "current_Regimen_Line"], axis=1)

    y = ltfu_data

    #SPlit the dataset into training and testing sets, with a ratio of 70% and 30% respectively.
    X_train, X_test, y_train, y_test = train_test_split(y, X_status_at_18_months, test_size=0.3)

    y_train = np.array(y_train)
    y_test = np.array(y_test)

    # Set the random seed for NumPy
    np.random.seed(42)

    # Initializing a random forest classifier and parameters with random values to find the best one
    rfc = RandomForestClassifier(random_state = 42)

    parameters = {
        "n_estimators":[5,10,50,100,250],
        "max_depth":[2,4,8,16,32,None]
    }
    logging.info("Fitting the model")
    # Performing 5 fold cross validation to find best set of parameters using grid seacr cv
    np.random.seed(124)
    cv = GridSearchCV(rfc,parameters,cv=5)
    cv.fit(X_train,y_train.ravel())

    # Display CV results
    def display(results):
        logging.info(f'Best parameters are: {results.best_params_}')
        logging.info("\n")
        mean_score = results.cv_results_['mean_test_score']
        std_score = results.cv_results_['std_test_score']
        params = results.cv_results_['params']
        for mean,std,params in zip(mean_score,std_score,params):
            logging.info(f'{round(mean,3)} + or -{round(std,3)} for the {params}')
    display(cv)

    #Import Random Forest Classifier and fit the model.
    logging.info("#Import Random Forest Classifier and fit the model.")
    random.seed(1)
    clf = RandomForestClassifier(max_depth = 16, n_estimators = 250)
    clf.fit(X_train,y_train)

    y_pred = clf.predict(X_test)
    logging.info("Accuracy:",metrics.accuracy_score(y_test, y_pred))

    #Printing classification report
    logging.info("#Printing classification report")
    logging.info(confusion_matrix(y_test, y_pred))

    # Show features with their importance value. The higer the value the more important the variable is in estimation of our target
    # variable.
    feature_imp = pd.Series(clf.feature_importances_, index = ltfu_data.columns).sort_values(ascending=False)

    #Creating a bar plot for features importance.
    # Set the figure size
    # plt.figure(figsize=(10, 6))
    sns.barplot(x=feature_imp, y=feature_imp.index)
    #Add labels to your graph
    plt.xlabel('Feature Importance Score')
    plt.ylabel('Features')
    plt.title("Visualizing Important Features for Status Prediction")
    #plt.legend()
    plt.savefig("iit.png")

    import joblib
    joblib.dump(clf, 'random_forest_model_status_at_18th_month.pkl')

    ### Predicting Patient Interruption point
    logging.info("### Predicting Patient Interruption point")
    useable_df_model_2 = df_model_2[df_model_2["status_at_18_months"] == "InActive"]
    X_Total_Visits = useable_df_model_2[['Total_Visits']]
    useable_df_model_2 = useable_df_model_2.drop(['status_at_18_months', 'Total_Visits'], axis = 1)

    # Label encoding all numeric columns
    # Iterate over each column in the DataFrame
    logging.info("# Iterate over each column in the DataFrame")
    for column in useable_df_model_2.select_dtypes(include='object').columns:
        le = LabelEncoder()
        useable_df_model_2[column] = le.fit_transform(useable_df_model_2[column].astype(str))
    logging.info(useable_df_model_2.columns)
    useable_df_model_2.to_csv("Model2_Cleaned_Data.csv")

    logging.info("#SPlit the dataset into training and testing sets, with a ratio of 70% and 30% respectively.")
    X_train, X_test, y_train, y_test = train_test_split(useable_df_model_2, X_Total_Visits, test_size=0.3)

    # Performing 5 fold cross validation to find best set of parameters using grid seacr cv
    logging.info("# Performing 5 fold cross validation to find best set of parameters using grid seacr cv")
    np.random.seed(124)
    cv = GridSearchCV(rfc,parameters,cv=5)
    cv.fit(X_train,y_train.values.ravel())
    display(cv)

    useable_df_model_2.head()

    # Performing 5 fold cross validation to find best set of parameters using grid seacr cv
    logging.info("# Performing 5 fold cross validation to find best set of parameters using grid seacr cv")
    np.random.seed(124)
    cv = GridSearchCV(rfc,parameters,cv=5)
    cv.fit(X_train,y_train.values.ravel())
    display(cv)

    #Import Random Forest Classifier and fit the model.
    logging.info("#Import Random Forest Classifier and fit the model.")
    random.seed(1)
    clf2 = RandomForestClassifier(max_depth = 2, n_estimators = 10)
    clf2.fit(X_train,y_train)

    y_pred = clf2.predict(X_test)
    logging.info("Accuracy:",metrics.accuracy_score(y_test, y_pred))

    logging.info(clf2.predict(useable_df_model_2[:1]))

    #Printing classification report
    logging.info("#Printing classification report")
    logging.info(confusion_matrix(y_test, y_pred))

    #Printing classification report
    logging.info("#Printing classification report")
    logging.info(classification_report(y_test, y_pred))

    # Show features with their importance value. The higer the value the more important the variable is in estimation of our target
    # variable.
    feature_imp = pd.Series(clf2.feature_importances_, index = df_no_outliers.columns).sort_values(ascending=False)

    #Creating a bar plot for features importance.
    # Set the figure size

    # plt.figure(figsize=(10, 6))
    sns.barplot(x=feature_imp, y=feature_imp.index)
    #Add labels to your graph
    plt.xlabel('Feature Importance Score')
    plt.ylabel('Features')
    plt.title("Visualizing Important Features for Total Visits Prediction")
    #plt.legend()
    plt.savefig("total_visit.png")

    joblib.dump(clf2, 'random_forest_model_number_of_visit_before_interruption.pkl')


