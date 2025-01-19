import pandas as pd
from imblearn.over_sampling import SMOTE
from collections import Counter
import numpy as np

dataset = pd.read_csv('sample_dataset.csv')

print("Columns in the dataset:", dataset.columns)

X_features = dataset.drop(columns=['Class'])  
y_target = dataset['Class']  

smote_oversampler = SMOTE(random_state=42)

X_resampled, y_resampled = smote_oversampler.fit_resample(X_features, y_target)

resampled_dataset = pd.DataFrame(X_resampled, columns=X_features.columns)
resampled_dataset['Class'] = y_resampled

class_dist_after_smote = Counter(y_resampled)
print(f"\nClass distribution after applying SMOTE:\n{class_dist_after_smote}")

simple_random_sample = resampled_dataset.sample(frac=0.2, random_state=42)
print("\nSimple Random Sampling (20% of the data):")
print(simple_random_sample['Class'].value_counts())

stratified_sample = resampled_dataset.groupby('Class', group_keys=False).apply(lambda x: x.sample(frac=0.2, random_state=42))
print("\nStratified Sampling (20% of the data):")
print(stratified_sample['Class'].value_counts())

systematic_indices = np.arange(0, len(resampled_dataset), step=5)  
systematic_sample = resampled_dataset.iloc[systematic_indices]
print("\nSystematic Sampling (Select every 5th record):")
print(systematic_sample['Class'].value_counts())

resampled_dataset['Cluster'] = pd.qcut(resampled_dataset['V1'], q=5, labels=False)  
cluster_sample = resampled_dataset[resampled_dataset['Cluster'] == 0] 
print("\nCluster Sampling (Select Cluster 0):")
print(cluster_sample['Class'].value_counts())

bootstrap_sample = resampled_dataset.sample(n=len(resampled_dataset), replace=True, random_state=42)
print("\nBootstrap Sampling (Sample with replacement):")
print(bootstrap_sample['Class'].value_counts())

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

models = {
    "Logistic Regression": LogisticRegression(random_state=42, solver='saga', max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42, n_estimators=1000),
    "Support Vector Machine": SVC(random_state=42),
    "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5)
}

def evaluate_models(X_train, y_train, X_test, y_test):
    results = {}
    for model_name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        results[model_name] = accuracy
    return results

X_simple = simple_random_sample.drop(columns=['Class'])
y_simple = simple_random_sample['Class']
X_train_simple, X_test_simple, y_train_simple, y_test_simple = train_test_split(X_simple, y_simple, test_size=0.2, random_state=42, stratify=y_simple)
results_simple = evaluate_models(X_train_simple, y_train_simple, X_test_simple, y_test_simple)

X_stratified = stratified_sample.drop(columns=['Class'])
y_stratified = stratified_sample['Class']
X_train_stratified, X_test_stratified, y_train_stratified, y_test_stratified = train_test_split(X_stratified, y_stratified, test_size=0.2, random_state=42, stratify=y_stratified)
results_stratified = evaluate_models(X_train_stratified, y_train_stratified, X_test_stratified, y_test_stratified)

X_systematic = systematic_sample.drop(columns=['Class'])
y_systematic = systematic_sample['Class']
X_train_systematic, X_test_systematic, y_train_systematic, y_test_systematic = train_test_split(X_systematic, y_systematic, test_size=0.2, random_state=42, stratify=y_systematic)
results_systematic = evaluate_models(X_train_systematic, y_train_systematic, X_test_systematic, y_test_systematic)

X_cluster = cluster_sample.drop(columns=['Class', 'Cluster'])  # Drop 'Cluster' column
y_cluster = cluster_sample['Class']
X_train_cluster, X_test_cluster, y_train_cluster, y_test_cluster = train_test_split(X_cluster, y_cluster, test_size=0.2, random_state=42, stratify=y_cluster)
results_cluster = evaluate_models(X_train_cluster, y_train_cluster, X_test_cluster, y_test_cluster)

X_bootstrap = bootstrap_sample.drop(columns=['Class'])
y_bootstrap = bootstrap_sample['Class']
X_train_bootstrap, X_test_bootstrap, y_train_bootstrap, y_test_bootstrap = train_test_split(X_bootstrap, y_bootstrap, test_size=0.2, random_state=42, stratify=y_bootstrap)
results_bootstrap = evaluate_models(X_train_bootstrap, y_train_bootstrap, X_test_bootstrap, y_test_bootstrap)

print("\nModel Performance on Simple Random Sampling:")
for model, accuracy in results_simple.items():
    print(f"{model}: {accuracy:.2f}")

print("\nModel Performance on Stratified Sampling:")
for model, accuracy in results_stratified.items():
    print(f"{model}: {accuracy:.2f}")

print("\nModel Performance on Systematic Sampling:")
for model, accuracy in results_systematic.items():
    print(f"{model}: {accuracy:.2f}")

print("\nModel Performance on Cluster Sampling:")
for model, accuracy in results_cluster.items():
    print(f"{model}: {accuracy:.2f}")

print("\nModel Performance on Bootstrap Sampling:")
for model, accuracy in results_bootstrap.items():
    print(f"{model}: {accuracy:.2f}")

all_results = {
    "Simple Random Sampling": results_simple,
    "Stratified Sampling": results_stratified,
    "Systematic Sampling": results_systematic,
    "Cluster Sampling": results_cluster,
    "Bootstrap Sampling": results_bootstrap,
}

best_sampling_for_model = {}
for model in models.keys():
    best_sampling = max(all_results, key=lambda sampling: all_results[sampling][model])
    best_accuracy = all_results[best_sampling][model]
    best_sampling_for_model[model] = (best_sampling, best_accuracy)

print("\nBest Sampling Technique for Each Model:")
for model, (sampling, accuracy) in best_sampling_for_model.items():
    print(f"{model}: Best Sampling - {sampling}, Accuracy - {accuracy:.2f}")
