# Credit Card Fraud Detection

A supervised machine learning project for detecting fraudulent credit card transactions. This project was developed as practice while completing Andrew Ng's **Machine Learning Specialization**.

## Project Overview

The goal of this project is to compare different supervised machine learning algorithms for identifying fraudulent credit card transactions.

The models evaluated are:

* Logistic Regression
* Decision Tree
* Random Forest
* XGBoost

Because fraudulent transactions represent a very small proportion of the dataset, **F1 score** was used as the primary evaluation metric rather than accuracy.

## Dataset

* **Source:** [Kaggle Credit Card Fraud Detection Dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
* **Transactions:** 284,807
* **Features:** 30
* **Fraud rate:** approximately 0.17%
* Features V1–V28 are anonymized PCA-transformed features.

## Approach

### 1. Data Splitting

The dataset was divided into:

* **60%** training set
* **20%** cross-validation set
* **20%** test set

### 2. Preprocessing

For the Logistic Regression model, features were standardized using `StandardScaler`, with the scaler fitted only on the training data and then applied to the cross-validation and test sets.

### 3. Model Training and Evaluation

Four supervised learning algorithms were compared:

* **Logistic Regression** — used as a baseline model
* **Decision Tree** — evaluated with different `max_depth` and `min_samples_split` values
* **Random Forest** — evaluated with different `max_depth`, `min_samples_split`, and `n_estimators` values
* **XGBoost** — trained using 500 estimators with early stopping

F1 score was calculated on the training, cross-validation, and test sets for each model.

## Results

| Model               |   Train F1 |      CV F1 |    Test F1 |
| ------------------- | ---------: | ---------: | ---------: |
| Logistic Regression |     0.7571 |     0.7467 |     0.6627 |
| Decision Tree       |     0.7796 |     0.7861 |     0.6919 |
| Random Forest       |     0.8940 |     0.8690 |     0.7955 |
| XGBoost             | **0.9562** | **0.9024** | **0.8023** |

## Conclusion

Among the four models tested, **XGBoost achieved the highest F1 score on the test set (0.8023)**, followed closely by Random Forest (0.7955).

This project provided practical experience with data preprocessing, supervised classification, model comparison, hyperparameter tuning, and evaluation on an imbalanced classification problem.

## Tools & Technologies

* Python
* Pandas
* NumPy
* Scikit-learn
* XGBoost
* Matplotlib
* Google Colab
