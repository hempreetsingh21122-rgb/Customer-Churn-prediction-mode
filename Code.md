# Imported Libraries

``` python

import pandas as pd

import numpy as np

```

# Imported dataset

``` pyhon

df = pd.read_csv("Churn_analysis_project.csv")

print("Dataset Loaded Successfully")

```

# Exploring Dataset

``` python

print("Shape",df.shape)


print("\ncolumns")
print(df.columns)

print(df.head())
print(df.info())

print(df.isnull().sum())

print("Customer Status")
print(df["Customer_Status"].value_counts())

print(df["Customer_Status"].value_counts(normalize=True)*100.0)

```

imported preprocessing tools for feature encoding and scaling.

``` python

from sklearn.model_selection import train_test_split

```

standardizing column for machine learning

``` python
from sklearn.preprocessing import OneHotEncoder, StandardScaler    
from sklearn.compose import ColumnTransformer  

```

imported pipeline for preprocessing and model training

``` python

from sklearn.pipeline import Pipeline     

```

imported machine learning algorithims for classifications

``` python

from sklearn.linear_model import LogisticRegression     
from sklearn.ensemble import RandomForestClassifier     

```

imported metrics to evaluate model performance

```python

from sklearn.metrics import (          # importing to check models performance
    accuracy_score,                    
    precision_score,                   # Positive prediction accuracy
    recall_score,                      # Actual Positives caught
    f1_score,                          # Balance of precision score & recall score
    roc_auc_score,                     # Churn prediction ability
    confusion_matrix,                  # Prediction error breakdown
    classification_report              # Overall Summary of prediction/full report
)

```

dataset imported

''' python
df = pd.read_csv("Churn_analysis_project.csv")
print("dataset loaded successfully")

``` python

Validating the dataset

``` python

print("shape",df.shape)

print("\nCustomer_Status distribution")
print(df["Customer_Status"].value_counts())

```

filtered the dataset to keep the records required for model training

``` python

df_model = df[df["Customer_Status"].isin(["Churned","Stayed"])].copy()  # keeping useful metrix 

print("\nModel dataset after removing Joined Customers:",df_model.shape)
 
print(df_model["Customer_Status"].value_counts())

```

mapped the customer status to 1s and 0s 

``` python

df_model["Churn"] = df_model["Customer_Status"].map({"Churned": 1, "Stayed": 0}) # mapping the customer_status to 1s and 0s

``` removed extra columns to preserve data intergrity and to prevent data leakage

``` python

drop_cols = ["Customer_ID","Customer_Status","Churn_Category","Churn_Reason"]

df_model = df_model.drop(columns=drop_cols,errors= "ignore") 

``` 

Handled missing values in both numerical and categorical columns

``` python

categorical_cols = df_model.select_dtypes(include=["object","str"]).columns          

for cols in categorical_cols:
    df_model[cols] = df_model[cols].fillna("Not Applicable")      

numeric_cols = df_model.select_dtypes(include=["int64","float64"]).columns

for cols in numeric_cols:
    df_model[cols] = df_model[cols].fillna(0)                 

splitted the columns into two variables for training and testing

x = df_model.drop("Churn",axis=1)             
y = df_model["Churn"]                                               

# seperated the features in categorical and numerical groups for further preprocessing.

numeric_features = x.select_dtypes(["int64","float64"]).columns      
categorical_features = x.select_dtypes(["object","str"]).columns

print("\nNumeric Features",len(numeric_features))
print("\nCategorical Features",len(categorical_features))

```

seperated the dataset into training and testing while preserving class balance

```python

# Train test split
x_train, x_test, y_train, y_test = train_test_split(x,
                                                    y,
                                                    test_size=0.2,
                                                    random_state=42, # split koo same rakhnaa
                                                    stratify=y)   # keeps class balance

```

created a preprocessing pipeline to scale numerical features and categorical features.
 
```python

preprocessor = ColumnTransformer(transformers=[("num",StandardScaler(),numeric_features),  
                                               ("cat",OneHotEncoder(handle_unknown="ignore"),categorical_features)])

```

trained the logistic regression model using the training data

``` python

Logistic Regression Model
logistic_model = Pipeline(steps=[("preprocessor",preprocessor),
                                 ("model",LogisticRegression(max_iter=1000,class_weight="balanced")) 
                                 ])

logistic_model.fit(x_train,y_train)      

```

Generated predictions on the training data

``` python

log_pred = logistic_model.predict(x_test)                          
log_prob = logistic_model.predict_proba(x_test)[:,1]

``` 

Evaluated the model's performance using classification metrics

``` python

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

```

trained the model using random forest classifier while maintaining class balance.

```python

rf_model = Pipeline(steps=[("preprocessor",preprocessor),
                           ("model",RandomForestClassifier(n_estimators=300,
                                                           random_state=42,
                                                           class_weight="balanced",
                                                           n_jobs=-1
                                                           ))])

rf_model.fit(x_train,y_train)

```

Generated predictions on the test data

``` python

rf_pred = rf_model.predict(x_test)
rf_prob = rf_model.predict_proba(x_test)[:,1]

```

Evaluated the models performance using classificaton metrics

``` python

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

```

Exported the Feature importance from the trained random forest model.

``` python

feature_names = rf_model.named_steps["preprocessor"].get_feature_names_out()
importances = rf_model.named_steps["model"].feature_importances_

feature_importance_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": importances
}).sort_values(by="Importance", ascending=False)

print("\Top 15 important Features")
print(feature_importance_df.head(15))

```

Created churn prediction output for power bi visualizattion 

``` python

#Use Same df_model features

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

```

Exported output

``` python

df_output.to_csv("output/customer_churn_predictions.csv",index=False)
print("output/customer_churn_predictions.csv")
print("\nRisk Level Distribution")

print(df_output["risk_level"].value_counts())

print(pd.crosstab(df_output["risk_level"], df_output["Customer_Status"]))
