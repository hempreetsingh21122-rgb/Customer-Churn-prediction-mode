# Customer-Churn-prediction-mode
Machine learning project to predict telecom customer churn using Python, Pandas, and Scikit-learn.

# Telecom Customer Churn Prediction

## Project Overview

This project focuses on predicting telecom customer churn using machine learning. The goal is to identify customers who are likely to leave the company so that businesses can take proactive retention actions.

The project uses Python for data preprocessing, model training, model evaluation, and generating churn prediction outputs that can be used for further analysis or dashboard reporting.

## Objective

- Predict whether a customer is likely to churn or stay.
- Compare machine learning models for churn prediction.
- Identify important customer behavior patterns related to churn risk.
- Generate customer-level churn probability and risk level output for reporting.

## Dataset

The dataset contains telecom customer information, including customer status, demographic details, service-related features, and account information.

For model building, only customers with the following statuses were used:

- Churned
- Stayed

Customers with `Joined` status were removed because they do not represent a clear churn/stay outcome for model training.

## Tools and Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Logistic Regression
- Random Forest Classifier
- OneHotEncoder
- StandardScaler
- ColumnTransformer
- Pipeline

## Project Workflow

### 1. Data Loading and Initial Exploration

The dataset was loaded using Pandas and basic checks were performed, including:

- Dataset shape
- Column names
- First few rows
- Data types
- Missing values
- Customer status distribution

### 2. Data Preparation

The dataset was filtered to keep only customers with `Churned` and `Stayed` status.

A new target column named `Churn` was created:

- Churned = 1
- Stayed = 0

Columns that could directly reveal the churn outcome were removed to avoid data leakage:

- Customer_ID
- Customer_Status
- Churn_Category
- Churn_Reason

### 3. Missing Value Handling

Missing values were handled separately for categorical and numerical columns:

- Categorical missing values were filled with `Not Applicable`
- Numerical missing values were filled with `0`

### 4. Feature Selection

The target variable was separated from the input features:

- `X` = customer features
- `y` = churn target

Numerical and categorical columns were identified for preprocessing.

### 5. Train-Test Split

The dataset was split into training and testing sets using an 80-20 split.

Stratified sampling was used to maintain the same churn/stay class balance in both training and testing data.

### 6. Data Preprocessing

A preprocessing pipeline was created using `ColumnTransformer`:

- Numerical features were scaled using `StandardScaler`
- Categorical features were encoded using `OneHotEncoder`

This helped prepare the data properly before model training.

### 7. Model Training

Two classification models were trained and compared:

#### Logistic Regression

Logistic Regression was used as a baseline classification model.

#### Random Forest Classifier

Random Forest Classifier was used to capture more complex patterns in customer churn behavior.

Both models were trained using Scikit-learn pipelines.

### 8. Model Evaluation

The models were evaluated using the following metrics:

- Accuracy
- Precision
- Recall
- F1 Score
- ROC AUC Score
- Confusion Matrix
- Classification Report

These metrics helped compare model performance and understand prediction quality.

### 9. Feature Importance

Feature importance was extracted from the Random Forest model to identify which customer attributes had the highest impact on churn prediction.

The top important features were displayed to better understand churn-driving factors.

### 10. Churn Probability and Risk Level Output

The Random Forest model was used to generate churn probability for each customer.

Customers were categorized into risk levels:

- High Risk: Churn probability >= 70%
- Medium Risk: Churn probability >= 40%
- Low Risk: Churn probability < 40%

A final prediction output file was created:

```text
output/customer_churn_predictions.csv
