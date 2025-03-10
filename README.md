﻿# Velocity-Car-Price-Prediction-

# Car Price Prediction using Machine Learning

## Overview
This project predicts car prices using various **machine learning models** and compares their performance. It includes traditional ML algorithms such as **Linear Regression, Decision Tree Regressor, and Random Forest Regressor**.

## Features
✅ Data preprocessing (handling missing values, encoding categorical variables, feature scaling)  
✅ Model training and evaluation  
✅ Exploratory Data Analysis (EDA) with visualizations  
✅ Statistical analysis of features  
✅ Comparison of multiple regression models  
✅ Performance metrics calculation (R² Score, MAE, MSE, RMSE)  

## Dataset
- **Source:** `Cardetails.csv`
- **Features:** Brand, Year, Engine Size, Mileage, Transmission, Fuel Type, Price
- **Preprocessing Steps:**
  - Handling missing values
  - Encoding categorical features using One-Hot Encoding
  - Feature scaling using StandardScaler
  - Train-test split (80%-20%)

## Exploratory Data Analysis (EDA)
✅ **Correlation Matrix** - Understanding relationships between numerical features  
![Heatmap](Images/heatmap.png)


##✅ Models Used & Performance

| Model                  | R² Score | MAE  | MSE  | RMSE  |
|------------------------|----------|------|------|-------|
| **Linear Regression**  | 82%      | 2.1  | 4.3  | 2.07  |
| **Decision Tree**      | 86%      | 1.8  | 3.5  | 1.87  |
| **Random Forest**      | 89%      | 1.6  | 2.9  | 1.71  |



## How to Run
### 1️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```
### 2️⃣ Run Jupyter Notebook
```bash
jupyter notebook Car_Price_Model.ipynb
```

## Future Enhancements
🔹 Hyperparameter tuning for better model performance  
🔹 Implement deep learning model with PyTorch/TensorFlow  
🔹 Feature engineering to improve accuracy  
🔹 Deploy the model using Flask or Streamlit  

---

