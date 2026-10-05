# 🔮 ChurnGuard — Telecom Customer Churn Prediction

A complete **Machine Learning and Deep Learning project** for predicting whether a telecom customer is likely to leave the company (**Churn**) or continue using its services (**No Churn**).

The project compares multiple machine learning algorithms with a Deep Learning neural network and provides an interactive customer prediction system.

---

## 📌 Project Overview

Customer churn is one of the most important problems in the telecommunications industry.

When a customer leaves a telecom company, the company loses recurring revenue and may also need to spend additional resources to acquire a new customer.

The goal of this project is to build a predictive system that can analyze customer information such as:

* Customer demographics
* Contract type
* Internet service
* Phone services
* Technical support
* Online security
* Streaming services
* Payment method
* Monthly charges
* Total charges
* Customer tenure

and predict whether the customer is likely to **Churn**.

### Prediction Classes

| Value | Meaning                                    |
| ----- | ------------------------------------------ |
| `0`   | 🟢 No Churn — Customer is expected to stay |
| `1`   | 🔴 Churn — Customer is expected to leave   |

---

# 📊 Dataset

The project uses the **Telco Customer Churn** dataset.

The dataset contains information about telecom customers, their subscribed services, billing information, contracts, and whether they eventually churned.

### Main Dataset Information

* **Rows:** 7,043 customers
* **Target:** `Churn`
* **Target Type:** Binary Classification
* **Original Target Values:**

  * `Yes` → `1`
  * `No` → `0`

The dataset contains customer information from several categories including demographics, services, contracts, and billing.

---

# 🧩 Features

The project uses customer information such as:

### 👤 Customer Information

* `gender`
* `SeniorCitizen`
* `Partner`
* `Dependents`
* `tenure`

### 📞 Phone Services

* `PhoneService`
* `MultipleLines`

### 🌐 Internet Services

* `InternetService`
* `OnlineSecurity`
* `OnlineBackup`
* `DeviceProtection`
* `TechSupport`

### 📺 Streaming Services

* `StreamingTV`
* `StreamingMovies`

### 📄 Contract & Billing

* `Contract`
* `PaperlessBilling`
* `PaymentMethod`

### 💰 Charges

* `MonthlyCharges`
* `TotalCharges`

---

# 🛠️ Data Preprocessing

Several preprocessing steps were applied before training the models.

## 1. Remove Unnecessary Columns

The `customerID` column was removed because it is an identifier and does not provide useful predictive information.

```python
df.drop(columns=["customerID"], inplace=True)
```

## 2. Convert `TotalCharges`

`TotalCharges` was originally stored as an object/string column.

It was converted to numeric values:

```python
df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"],
    errors="coerce"
)
```

Missing values were then handled using the median:

```python
df["TotalCharges"].fillna(
    df["TotalCharges"].median(),
    inplace=True
)
```

## 3. Encode the Target

The target variable was converted into binary values:

```python
df["Churn"] = df["Churn"].map({
    "Yes": 1,
    "No": 0
})
```

## 4. Remove Duplicate Rows

Duplicate records were checked and removed:

```python
df.drop_duplicates(inplace=True)
```

## 5. Categorical Encoding

Categorical features were converted into numerical features using **One-Hot Encoding**:

```python
df_encoded = pd.get_dummies(
    df,
    columns=categorical_cols,
    drop_first=True
)
```

This allows the machine learning models to work with categorical customer information.

---

# 🔀 Train / Test Split

The dataset was divided into training and testing sets.

```python
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)
```

### Split

* **80%** → Training
* **20%** → Testing

`stratify=y` was used to maintain a similar class distribution in both training and testing datasets.

---

# 🤖 Machine Learning Models

The project compares multiple classification algorithms.

## 1. Decision Tree

A Decision Tree classifier was trained using entropy as the splitting criterion.

```python
DecisionTreeClassifier(
    criterion="entropy",
    splitter="best",
    max_depth=5,
    min_samples_split=3,
    min_samples_leaf=1,
    random_state=32
)
```

---

## 2. Random Forest

A Random Forest classifier was trained using multiple decision trees.

```python
RandomForestClassifier(
    n_estimators=400,
    max_depth=16,
    criterion="entropy",
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)
```

`class_weight="balanced"` was used to give additional importance to the minority class during training.

---

## 3. Support Vector Machine

An SVM classifier with an RBF kernel was also trained.

```python
SVC(
    C=5.0,
    kernel="rbf",
    gamma="auto"
)
```

---

# 🧠 Deep Learning Model

In addition to traditional machine learning algorithms, a **Deep Neural Network (MLP)** was developed using TensorFlow/Keras.

Before training the neural network, numerical features were standardized using `StandardScaler`.

```python
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

The neural network architecture is:

```text
Input Layer
     ↓
Dense(128, ReLU)
     ↓
Dropout(0.3)
     ↓
Dense(64, ReLU)
     ↓
Dropout(0.2)
     ↓
Dense(32, ReLU)
     ↓
Dense(1, Sigmoid)
     ↓
Churn Probability
```

### Model Configuration

```python
dl_model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)
```

The model uses:

* **ReLU** activation for hidden layers
* **Sigmoid** activation for binary classification
* **Binary Cross-Entropy** loss
* **Adam** optimizer
* **Dropout** for regularization

---

# ⏹️ Early Stopping

Early stopping was used to reduce overfitting.

```python
EarlyStopping(
    monitor="val_loss",
    patience=10,
    restore_best_weights=True
)
```

The model stops training when the validation loss stops improving and restores the best-performing weights.

---

# 📈 Model Evaluation

The models are evaluated using several classification metrics rather than relying only on accuracy.

### Metrics

* **Accuracy**
* **Precision**
* **Recall**
* **F1-Score**

### Why multiple metrics?

Customer churn datasets are often imbalanced.

Therefore, accuracy alone may not provide a complete picture of model performance.

For example, a model could achieve high accuracy by correctly predicting the majority class while performing poorly on customers who actually churn.

For this reason, **Precision, Recall, and F1-Score** are also considered.

---

# 📊 Model Comparison

The project generates a comparison table containing:

```text
Model
Accuracy
Precision
Recall
F1-Score
```

The models are also visualized using a comparison chart.

The current comparison includes:

* Decision Tree
* Random Forest
* SVM
* Deep Learning (MLP)

This makes it easier to compare traditional Machine Learning approaches with Deep Learning.

---

# 🔮 Interactive Customer Churn Predictor

The project also includes an interactive prediction form.

Users can enter customer information such as:

```text
Tenure
Monthly Charges
Total Charges
Gender
Senior Citizen
Partner
Dependents
Phone Service
Multiple Lines
Internet Service
Online Security
Online Backup
Device Protection
Tech Support
Streaming TV
Streaming Movies
Contract
Paperless Billing
Payment Method
```

The input is automatically converted into the same feature representation used during model training.

The system then sends the processed customer data to the trained models.

---

# 🎯 Prediction Output

The system returns predictions such as:

```text
📊 MODEL PREDICTION RESULTS

• Decision Tree    : 🔴 Churn
• Random Forest    : 🟢 No Churn
• SVM Classifier   : 🔴 Churn
• Deep Learning   : 🔴 Churn (Probability: 78.4%)
```

For the Deep Learning model, the system also displays the predicted churn probability.

For example:

```text
Churn Probability: 78.4%
```

This means the model estimates a relatively high likelihood that the customer will churn.

---

# 🌐 Streamlit Application

The project was also prepared for deployment as an interactive **Streamlit web application**.

The application allows users to enter customer information through a graphical interface instead of manually editing Python variables.

The trained models and preprocessing objects can be saved and loaded by the Streamlit application.

### Saved Resources

```text
decision_tree_model.pkl
random_forest_model.pkl
svc_model.pkl
scaler.pkl
dl_model.keras
```

The application can then load these resources and perform predictions for new customers.

---

# 💾 Model Saving

Traditional Machine Learning models were saved using `joblib`:

```python
joblib.dump(Model, "decision_tree_model.pkl")
joblib.dump(rf_model, "random_forest_model.pkl")
joblib.dump(Model_SVC, "svc_model.pkl")
joblib.dump(scaler, "scaler.pkl")
```

The Deep Learning model was saved using the Keras format:

```python
dl_model.save("dl_model.keras")
```

---

# 🔄 Complete Workflow

The complete project workflow can be summarized as:

```text
Raw Dataset
     ↓
Data Understanding
     ↓
Data Cleaning
     ↓
Missing Value Handling
     ↓
Duplicate Removal
     ↓
Target Encoding
     ↓
Categorical Encoding
     ↓
Train / Test Split
     ↓
Feature Scaling
     ↓
┌─────────────────────────────────┐
│                                 │
│     Machine Learning Models     │
│                                 │
│  Decision Tree                  │
│  Random Forest                 │
│  SVM                           │
│                                 │
└─────────────────────────────────┘
     ↓
┌─────────────────────────────────┐
│                                 │
│       Deep Learning Model       │
│                                 │
│       Neural Network (MLP)      │
│                                 │
└─────────────────────────────────┘
     ↓
Model Evaluation
     ↓
Model Comparison
     ↓
Customer Input
     ↓
Preprocessing
     ↓
Prediction
     ↓
Churn / No Churn
```

---

# 🧰 Technologies Used

| Technology             | Purpose                                |
| ---------------------- | -------------------------------------- |
| Python                 | Main programming language              |
| Pandas                 | Data manipulation and preprocessing    |
| NumPy                  | Numerical operations                   |
| Scikit-learn           | Machine Learning models and evaluation |
| TensorFlow / Keras     | Deep Learning                          |
| Matplotlib             | Data visualization                     |
| Seaborn                | Model comparison visualization         |
| Joblib                 | Saving Machine Learning models         |
| Streamlit              | Interactive web application            |
| Jupyter / Google Colab | Development and experimentation        |



---

# 🚀 Future Improvements

Possible future improvements include:

* Hyperparameter optimization for all models
* Better handling of class imbalance
* Threshold optimization for churn prediction
* More advanced Deep Learning architectures
* Explainable AI using SHAP
* Feature importance analysis
* Customer churn risk levels
* Model deployment
* Streamlit Cloud deployment
* API deployment using FastAPI
* Real-time prediction system

---

# 🎯 Project Goals

The main goals of this project are to:

1. Understand and preprocess real-world customer data.
2. Build multiple Machine Learning classification models.
3. Develop a Deep Learning model for binary classification.
4. Compare traditional Machine Learning with Deep Learning.
5. Handle categorical variables and class imbalance.
6. Evaluate models using multiple classification metrics.
7. Build an interactive customer prediction system.
8. Prepare the trained models for deployment.

---

# 👨‍💻 Author

**Moaz**



# ⭐ Conclusion

**ChurnGuard** demonstrates a complete end-to-end Machine Learning workflow for telecom customer churn prediction, starting from raw customer data and preprocessing, through multiple Machine Learning and Deep Learning models, evaluation and comparison, and finally interactive customer-level prediction.

The project focuses not only on achieving a good accuracy score, but also on understanding the complete process of building a practical classification system.
