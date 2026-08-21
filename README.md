# Credit Card Fraud Detection

A machine learning project focused on detecting fraudulent credit card transactions. The project compares several classification models and evaluates their performance using the F1-score.

## Project Overview

The goal of this project is to build and compare machine learning models capable of distinguishing between legitimate and fraudulent credit card transactions.

The models evaluated are:

* Logistic Regression
* Decision Tree
* Random Forest
* XGBoost Classifier

The project was developed using Python in Google Colab.

## Dataset

The project uses a credit card fraud dataset from Kaggle containing legitimate and fraudulent transactions.

Because fraudulent transactions represent a much smaller proportion of the dataset, the classification problem is highly imbalanced.

## Models

### Logistic Regression

Used as a baseline classification model.

### Decision Tree

A Decision Tree was trained and tuned using:

* `max_depth`
* `min_samples_split`

### Random Forest

The Random Forest model was tuned using:

* `max_depth`
* `min_samples_split`
* `n_estimators`

The `n_estimators` parameter controls the number of decision trees used in the forest.

### XGBoost Classifier

XGBoost Classifier was included as a gradient-boosting model for comparison with the other classification approaches.

## Evaluation Metric

Because the dataset is highly imbalanced, accuracy alone can be misleading for fraud detection.

The primary evaluation metric used in this project is the **F1-score**. The F1-score balances precision and recall into a single metric, making it useful for evaluating performance when both false positives and false negatives are important.

The models were therefore compared based on their **F1-score** to determine which approach performed best at identifying fraudulent transactions.

## Project Workflow

1. Load and inspect the dataset
2. Perform exploratory data analysis
3. Preprocess the data
4. Perform feature engineering
5. Split the data for training and evaluation
6. Train multiple classification models
7. Tune selected model hyperparameters
8. Evaluate models using F1-score
9. Compare model performance

## Technologies

* Python
* Pandas
* NumPy
* Scikit-learn
* XGBoost
* Matplotlib
* Google Colab

## Files

* `credit_card_fraud_detection.ipynb` — Complete analysis, preprocessing, model training, tuning, and evaluation.

## Conclusion

This project compares multiple machine learning approaches for credit card fraud detection and evaluates their ability to identify fraudulent transactions using the F1-score.

The **XGBoost Classifier** achieved the highest performance among the evaluated models, with an F1-score of **0.95 on the training set** and **0.90 on the cross-validation set**. The cross-validation result indicates strong performance on unseen validation data while helping assess the model's ability to generalize beyond the training data.
