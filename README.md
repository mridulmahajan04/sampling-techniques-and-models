Introduction
This project evaluates the impact of various sampling techniques on the performance of machine learning models. It demonstrates how different data sampling methods affect classification accuracy using multiple models.

Key Features
Implementation of SMOTE (Synthetic Minority Oversampling Technique) to handle class imbalance.
Comparison of five sampling techniques:
Simple Random Sampling
Stratified Sampling
Systematic Sampling
Cluster Sampling
Bootstrap Sampling

Evaluation of the following machine learning models:
Logistic Regression
Decision Tree
Random Forest
Support Vector Machine (SVM)
K-Nearest Neighbors (KNN)


Dataset
The project assumes a dataset named sample_dataset.csv with the following structure:

Input Features: Columns representing independent variables (e.g., V1, V2, etc.).
Target Variable: A column named Class, which is imbalanced.
You can replace sample_dataset.csv with your own dataset, ensuring it has the same structure.


Sampling Techniques
The following sampling techniques are implemented to create balanced datasets:

Simple Random Sampling: Randomly selects a subset of the data.
Stratified Sampling: Ensures the proportion of each class in the sample matches the original dataset.
Systematic Sampling: Selects data points at regular intervals (e.g., every 5th record).
Cluster Sampling: Divides data into clusters and selects samples from one or more clusters.
Bootstrap Sampling: Samples data with replacement to create a new dataset.

Machine Learning Models
The project evaluates the following classifiers:

Logistic Regression
Decision Tree
Random Forest
Support Vector Machine (SVM)
K-Nearest Neighbors (KNN)
Each model is trained and tested on datasets created using the above sampling techniques, and their classification accuracy is compared.
