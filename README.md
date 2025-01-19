# sampling-techniques-and-models
Camparison of sampling techniques
Introduction
This project evaluates the impact of various sampling techniques on the performance of machine learning models. It demonstrates how different data sampling methods affect classification accuracy using multiple models.

The project includes:

Implementation of SMOTE (Synthetic Minority Oversampling Technique) to handle class imbalance.
Comparison of five sampling techniques: Simple Random Sampling, Stratified Sampling, Systematic Sampling, Cluster Sampling, and Bootstrap Sampling.
Evaluation of models like Logistic Regression, Decision Tree, Random Forest, Support Vector Machine, and K-Nearest Neighbors.
Dataset
The project assumes a dataset named sample_dataset.csv with:
Input features (e.g., V1, V2, etc.).
A target variable named Class with imbalanced classes.
You can replace sample_dataset.csv with your dataset while ensuring the column structure matches.

Sampling Techniques
The following sampling methods are implemented:

Simple Random Sampling: Randomly selects a subset of the data.
Stratified Sampling: Ensures the proportion of each class remains consistent.
Systematic Sampling: Selects data points at regular intervals (e.g., every 5th record).
Cluster Sampling: Divides data into clusters and selects samples from one cluster.
Bootstrap Sampling: Samples data with replacement to create a new dataset.
Machine Learning Models
The following classifiers are used to evaluate performance:

Logistic Regression
Decision Tree
Random Forest
Support Vector Machine (SVM)
K-Nearest Neighbors (KNN)
Each model is trained and evaluated on datasets created using the above sampling techniques
