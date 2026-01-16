# Explaratory-Data-Analysis-for-Machine-Learning


## Titanic Survival Prediction using Logistic Regression
1. Project Overview

This project performs Exploratory Data Analysis (EDA) on the Titanic dataset and builds a Logistic Regression model to predict passenger survival. The goal is to understand the data, handle missing values appropriately, engineer useful features, and train a baseline classification model.

2. Dataset

Dataset: Titanic Dataset

Source: Kaggle

Target Variable: Survived

0 → Did not survive

1 → Survived

3. Exploratory Data Analysis (EDA)

During EDA, the following insights were observed:

The dataset contains missing values, mainly in:

Age

Cabin

Survival rates differ across:

Passenger class (Pclass)

Gender (Sex)

Some features show class imbalance (e.g. SibSp)

Continuous variables such as Age and Fare are right-skewed

Visualizations used:

Count plots

Histograms

Heatmaps

Boxplots

4. Data Cleaning & Preprocessing

Key preprocessing steps include:

Missing Values

Age: Imputed using the average age within each passenger class (Pclass)

Cabin: Dropped due to a large number of missing values

Embarked: Converted into dummy variables

5. Feature Encoding

Categorical variables converted using pd.get_dummies()

drop_first=True used to avoid multicollinearity

Feature Selection

Dropped non-informative columns such as:

Name

Ticket

6. Model Building

Model: Logistic Regression

Why Logistic Regression?

Suitable for binary classification problems

Simple and interpretable baseline model

Train-Test Split

Features (X) separated from target (y)

Data split into training and test sets

To address convergence warnings:

Increased the number of iterations (max_iter)

(Optional) Feature scaling can further improve convergence

7. Model Evaluation

The trained model is evaluated using:

Accuracy score

Confusion matrix

Classification report

This provides insight into how well the model predicts survival outcomes.

- Project Structure
- Titanic-Logistic-Regression
 ┣ Titanic_EDA_LogisticRegression.ipynb
 ┣ README.md
 ┗ titanic_train.csv

### How to Run

Clone the repository

Install required libraries:

pip install pandas numpy matplotlib seaborn scikit-learn


Open the notebook:

jupyter notebook Titanic_EDA_LogisticRegression.ipynb

- Notes

This project focuses on clarity and correctness rather than advanced optimization

It serves as a baseline ML project and can be extended with:

Feature scaling

Cross-validation

More advanced models (Random Forest, XGBoost)
