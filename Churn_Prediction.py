import pandas as pd
import numpy as np
df = pd.read_csv("Churn_analysis_project.csv")

print("Dataset Loaded Successfully")
print("Shape",df.shape)

print("\ncolumns")
print(df.columns)

print(df.head())
print(df.info())

print(df.isnull().sum())


print("Customer Status")
print(df["Customer_Status"].value_counts())

print(df["Customer_Status"].value_counts(normalize=True)*100.0)

from sklearn.model_selection import train_test_split  # model ko 2 parts mai todna
from sklearn.preprocessing import OneHotEncoder, StandardScaler    # OneHotEncoder - text ko binary maii dalnaa, Standardscaler - Normalizing the numbers
from sklearn.compose import ColumnTransformer  #ColumnTransformer - Act as a manager to OneHotCoder & Stadardscaler

from sklearn.pipeline import Pipeline     # preprocessing & train the model

from sklearn.linear_model import LogisticRegression     # for probability prediction but limited
from sklearn.ensemble import RandomForestClassifier     # better probability prediction  

from sklearn.metrics import (          # importing to check models performance
    accuracy_score,                    
    precision_score,                   # Positive prediction accuracy
    recall_score,                      # Actual Positives caught
    f1_score,                          # Balance of precision score & recall score
    roc_auc_score,                     # Churn prediction ability
    confusion_matrix,                  # Prediction error breakdown
    classification_report              # Overall Summary of prediction/full report
)

df = pd.read_csv("Churn_analysis_project.csv")
print("dataset loaded successfully")
print("shape",df.shape)

print("\nCustomer_Status distribution")
print(df["Customer_Status"].value_counts())       # checking distribution

df_model = df[df["Customer_Status"].isin(["Churned","Stayed"])].copy()  # keeping useful metrix 

print("\nModel dataset after removing Joined Customers:",df_model.shape) 
print(df_model["Customer_Status"].value_counts())
df_model["Churn"] = df_model["Customer_Status"].map({"Churned": 1, "Stayed": 0}) # mapping the customer_status to 1s and 0s

drop_cols = ["Customer_ID","Customer_Status","Churn_Category","Churn_Reason"]

df_model = df_model.drop(columns=drop_cols,errors= "ignore")       # prevents data leakage by removing columns that directly reveal churn outcome

categorical_cols = df_model.select_dtypes(include=["object","str"]).columns          

for cols in categorical_cols:
    df_model[cols] = df_model[cols].fillna("Not Applicable")        #Filling missing values in strings/object columns

numeric_cols = df_model.select_dtypes(include=["int64","float64"]).columns

for cols in numeric_cols:
    df_model[cols] = df_model[cols].fillna(0)                       #Filling missing values in numerical columns

x = df_model.drop("Churn",axis=1)                                   # dropping churn from input features
y = df_model["Churn"]                                               # targeted output

# Idetify numeric and categorical columns
numeric_features = x.select_dtypes(["int64","float64"]).columns      
categorical_features = x.select_dtypes(["object","str"]).columns

print("\nNumeric Features",len(numeric_features))
print("\nCategorical Features",len(categorical_features))

# Train test split
x_train, x_test, y_train, y_test = train_test_split(x,
                                                    y,
                                                    test_size=0.2,
                                                    random_state=42, # split koo same rakhnaa
                                                    stratify=y)   # keeps class balance

preprocessor = ColumnTransformer(transformers=[("num",StandardScaler(),numeric_features),   # preprocessing the data turning raw data into standarscaler and onehotcoder
                                               ("cat",OneHotEncoder(handle_unknown="ignore"),categorical_features)])

# Logistic Regression Model
logistic_model = Pipeline(steps=[("preprocessor",preprocessor),
                                 ("model",LogisticRegression(max_iter=1000,class_weight="balanced"))  # preprocessing and training of the model
                                 ])

logistic_model.fit(x_train,y_train)      

log_pred = logistic_model.predict(x_test)                          
log_prob = logistic_model.predict_proba(x_test)[:,1]

print("\nlogistic regression score")
print("Accuracy: ",accuracy_score(y_test,log_pred))
print("Precision: ",precision_score(y_test,log_pred))
print("recall",recall_score(y_test,log_pred))
print("F1 score",f1_score(y_test,log_pred))
print("ROC AUC",roc_auc_score(y_test,log_prob))

print("\nconfusion_matrix:")                # 1. TN = Stayed ko stayed kaha 2. FP = Stayed ko churn bola
print(confusion_matrix(y_test,log_pred))    # 1. FN = Churn ko stayed bola  3. TN = Churn ko Churn bola

print("\nClassificationn report: ")
print(classification_report(y_test,log_pred))

rf_model = Pipeline(steps=[("preprocessor",preprocessor),
                           ("model",RandomForestClassifier(n_estimators=300,
                                                           random_state=42,
                                                           class_weight="balanced",
                                                           n_jobs=-1
                                                           ))])

rf_model.fit(x_train,y_train)

rf_pred = rf_model.predict(x_test)
rf_prob = rf_model.predict_proba(x_test)[:,1]

print("\Random Forest Result")
print("Accuracy",accuracy_score(y_test,rf_pred))
print("Precision",precision_score(y_test,rf_pred))
print("recall",recall_score(y_test,rf_pred))
print("f1_score",f1_score(y_test,rf_pred))
print("ROC AUC",roc_auc_score(y_test,rf_prob))

print("\confusion matrix:")
print(confusion_matrix(y_test,rf_pred))

print("\nclassification report:")
print(classification_report(y_test,rf_pred))

# Feature importance from random forest
feature_names = rf_model.named_steps["preprocessor"].get_feature_names_out()
importances = rf_model.named_steps["model"].feature_importances_

feature_importance_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": importances
}).sort_values(by="Importance", ascending=False)

print("\Top 15 important Features")
print(feature_importance_df.head(15))

# Create prediction output for power bi
# Use Same df_model features
prediction_features = df_model.drop("Churn",axis=1)

df_output = df[df["Customer_Status"].isin(["Churned","Stayed"])].copy()

df_output["churn_probability"] = rf_model.predict_proba(prediction_features)[:,1]

def risk_level(prob):
    if prob >= 0.70:
        return "high risk"
    elif prob >= 0.4:
        return "medium risk"
    else:
        return "low risk"
    
df_output["risk_level"] = df_output["churn_probability"].apply(risk_level)

df_output["predicted_churn"] = np.where(df_output["churn_probability"]>=0.50,"Yes","No")
import os

os.makedirs("output",exist_ok=True)

# Export output
df_output.to_csv("output/customer_churn_predictions.csv",index=False)
print("output/customer_churn_predictions.csv")
print("\nRisk Level Distribution")

print(df_output["risk_level"].value_counts())

print(pd.crosstab(df_output["risk_level"], df_output["Customer_Status"]))


      
