# 📊 Employee Attrition Prediction

## 🚀 Project Overview

This project focuses on predicting whether an employee will leave a company (**Attrition**) using Machine Learning models.

The goal is to help organizations identify employees at risk of leaving and take preventive actions.

---

## 📁 Dataset

The project uses two datasets:

* `train.csv` → Used for training the model
* `test.csv` → Used for making final predictions

---

## ⚙️ Workflow

### 1️⃣ Data Preprocessing

* Removed unnecessary columns (EmployeeCount, EmployeeNumber, etc.)
* Converted categorical data into numerical format:

  * Binary Encoding (Gender, OverTime)
  * Ordinal Encoding (BusinessTravel)
  * One-Hot Encoding (Department, JobRole, etc.)

---

### 2️⃣ Feature Scaling

* Used **StandardScaler** to normalize data
* Ensures all features are on the same scale

---

### 3️⃣ Model Training

Trained multiple Machine Learning models:

* Logistic Regression
* K-Nearest Neighbors (KNN)
* Decision Tree
* Support Vector Machine (SVM)
* Neural Network (MLP)
* Random Forest
* Gradient Boosting

---

### 4️⃣ Model Evaluation

* Compared models using **accuracy**
* Best model:

  ```
  Logistic Regression → ~87% accuracy
  ```

---

### 5️⃣ Final Prediction

* Applied model on `test.csv`
* Output:

  * `0` → Employee stays
  * `1` → Employee leaves

---

## 🧠 Key Concepts Used

* Data Cleaning
* Feature Engineering
* Encoding Techniques
* Feature Scaling
* Model Training & Evaluation

---

## 💾 Model Saving

The trained model is saved using pickle:

```python
pickle.dump(model, open('model.pkl', 'wb'))
```

---

## 📌 Project Structure

```
Employee Attrition Prediction/
│
├── Datasets/
│   ├── train.csv
│   ├── test.csv
│
├── Employee_Attrition_Prediction.ipynb
├── model.pkl
└── README.md
```

---

## 🎯 Conclusion

* Built a complete ML pipeline
* Achieved ~87% accuracy
* Successfully predicted employee attrition

---

## 🔥 One-Line Summary

👉 Predict whether an employee will leave using Machine Learning

---

