# Telecom Customer Churn Prediction

## Project Objective

The objective of this project is to predict whether a telecom customer is likely to churn or stay using machine learning models. The project also focuses on identifying high-risk customers and generating churn probability outputs for business reporting and retention analysis.

---

# Step 1: Importing Required Libraries

Imported the required Python libraries for:

- Data analysis
- Data preprocessing
- Machine learning
- Model evaluation
- Output generation

Libraries used:

- Pandas
- NumPy
- Scikit-learn

---

# Step 2: Loading the Dataset

The telecom customer dataset was loaded using Pandas.

Basic dataset checks were performed:

- Dataset shape
- Column names
- Data types
- Missing values
- Customer status distribution

---

# Step 3: Filtering Relevant Customer Records

Only customers with the following statuses were selected for model training:

- Churned
- Stayed

Customers with `Joined` status were removed because they do not represent a completed churn outcome.

A new target column named `Churn` was created:

- Churned = 1
- Stayed = 0

---

# Step 4: Preventing Data Leakage

Columns that could directly reveal churn information were removed before training the model.

Removed columns:

- Customer_ID
- Customer_Status
- Churn_Category
- Churn_Reason

This step helps prevent data leakage and improves model reliability.

---

# Step 5: Handling Missing Values

Missing values were handled separately for categorical and numerical columns.

### Categorical Columns
Missing values were replaced with:

```python
"Not Applicable"
```

### Numerical Columns
Missing values were replaced with:

```python
0
```

---

# Step 6: Splitting Features and Target Variable

The dataset was divided into:

### Input Features (X)
Customer-related information used for prediction.

### Target Variable (y)
Churn outcome:

- 1 = Churned
- 0 = Stayed

---

# Step 7: Identifying Numerical and Categorical Features

The project automatically identified:

- Numerical columns
- Categorical columns

This helped prepare the preprocessing pipeline correctly.

---

# Step 8: Train-Test Split

The dataset was divided into:

- 80% Training Data
- 20% Testing Data

Stratified sampling was used to maintain churn class balance.

```python
stratify=y
```

---

# Step 9: Data Preprocessing

A preprocessing pipeline was created using `ColumnTransformer`.

### Numerical Features
Processed using:

- StandardScaler

### Categorical Features
Processed using:

- OneHotEncoder

This ensured proper feature transformation before model training.

---

# Step 10: Building the Logistic Regression Model

A Logistic Regression classification model was trained using a Scikit-learn pipeline.

The model was used as a baseline model for churn prediction.

### Evaluation Metrics Used

- Accuracy
- Precision
- Recall
- F1 Score
- ROC AUC Score
- Confusion Matrix
- Classification Report

---

# Step 11: Building the Random Forest Model

A Random Forest Classifier model was trained and compared with Logistic Regression.

Random Forest was used because it can capture more complex customer behavior patterns.

### Random Forest Configuration

- n_estimators = 300
- class_weight = balanced
- random_state = 42

---

# Step 12: Evaluating Model Performance

Both models were evaluated using classification metrics.

The following evaluation techniques were used:

- Accuracy Score
- Precision Score
- Recall Score
- F1 Score
- ROC AUC Score
- Confusion Matrix
- Classification Report

These metrics helped understand prediction quality and model performance.

---

# Step 13: Feature Importance Analysis

Feature importance was extracted from the Random Forest model.

This helped identify which customer attributes had the highest impact on churn prediction.

Top important features were displayed for analysis.

---

# Step 14: Generating Customer Churn Probability

The trained Random Forest model was used to generate churn probability for each customer.

A churn probability score was created for business analysis and reporting.

---

# Step 15: Creating Risk Levels

Customers were categorized into different risk groups based on churn probability.

### Risk Level Logic

- High Risk → Probability >= 70%
- Medium Risk → Probability >= 40%
- Low Risk → Probability < 40%

---

# Step 16: Predicting Customer Churn

Customers were classified as:

- Yes → Likely to churn
- No → Likely to stay

Prediction logic was created using churn probability thresholds.

---

# Step 17: Exporting Final Output

The final prediction dataset was exported as a CSV file.

### Output File

```text
output/customer_churn_predictions.csv
```

The output contains:

- Churn probability
- Risk level
- Predicted churn status

This output can be used in Power BI dashboards and business reporting.

---

# Business Use Case

This project helps telecom companies:

- Identify high-risk customers
- Improve customer retention strategies
- Target customers with retention offers
- Reduce customer churn
- Support business decision-making using predictive analysis

---

# Tools & Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Logistic Regression
- Random Forest Classifier
- OneHotEncoder
- StandardScaler
- ColumnTransformer
- Machine Learning Pipeline

---

# Future Improvements

Possible future improvements for the project:

- Hyperparameter tuning
- Cross-validation
- Additional feature engineering
- XGBoost implementation
- Power BI integration
- Model deployment using Flask or Streamlit
