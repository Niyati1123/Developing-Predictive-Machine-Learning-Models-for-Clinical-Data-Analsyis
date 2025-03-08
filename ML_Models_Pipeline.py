# Clustering Analysis

##1 Hierarchical Clustering

plt.figure(figsize=(15, 10))
linkage_matrix = linkage(X_train_outliers_removed, method='ward')
dendrogram(linkage_matrix)
plt.title('Hierarchical Clustering Dendrogram')
plt.show()
k = 2
clusters = fcluster(linkage_matrix, k, criterion='maxclust')

X_train_scaled

##2 K-means Clustering

num_clusters = 2
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
kmeans.fit(X_train_prepared)
# Assign cluster labels to the original data
X_train['cluster'] = kmeans.labels_
# Combine the cluster labels with the original training set
patient_df_with_clusters = patient_df.loc[X_train.index].copy()
patient_df_with_clusters['cluster'] = X_train['cluster']

# Explore characteristics of each cluster
for cluster_label in range(num_clusters):
    cluster_data = patient_df_with_clusters[patient_df_with_clusters['cluster'] == cluster_label]
    print(f'\nCluster {cluster_label} Characteristics:')
    print(cluster_data.describe())

# Calculate the mean of each feature for each cluster
# Exclude non-numeric columns from the calculation
cluster_means = patient_df_with_clusters.select_dtypes(include=['number']).groupby('cluster').mean()
print("\nMean Value of Features for Each Cluster:")
print(cluster_means)

plt.scatter(X_train_pca[:, 0], X_train_pca[:, 1], c=kmeans.labels_, cmap='viridis')
plt.title('Clustering Analysis')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.show()

# Classifier: The Model (Logistic Regression, RandomForestClassifier, GradientBoostingClassifier, SVM)

## Run the below code (line 47 to 91) in one chunk

# Prepare the classifiers
classifiers = {
    "Logistic Regression": LogisticRegression(),
    "Random Forest": RandomForestClassifier(random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(random_state=42),
    "Support Vector Machine": SVC(probability=True, random_state=42)
}
# Train and evaluate the classifiers
results = {}
for name, clf in classifiers.items():
    # Train the classifier
    clf.fit(X_train_scaled, y_train)
    # Predict on the test set
    y_pred = clf.predict(X_test_scaled)
    y_proba = clf.predict_proba(X_test_scaled)[:, 1]
    # Evaluate the classifier
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_proba)
    cv_scores = cross_val_score(clf, X_train_scaled, y_train, cv=5)
    # Store results
    results[name] = {
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1,
        "ROC-AUC Score": roc_auc,
        "Cross-validation scores": cv_scores.tolist(),  # Convert to list for printing

    }
    # Print the performance
    print(f"Results for {name}:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"ROC-AUC Score: {roc_auc:.4f}\n")
    print(f"Cross-validation scores: {cv_scores}\n")

# Compare results
results_df = pd.DataFrame(results).transpose()
print("Comparison of Classifiers:")
print(results_df)

## Finding out the most important feature?

# Feature importance
# Train the classifier
clf =GradientBoostingClassifier(random_state=42)
clf.fit(X_train_prepared, y_train)

feature_importance = clf.feature_importances_
important_features = X_train_prepared.columns[np.argsort(feature_importance)[::-1]]
print("Most Important Feature:", important_features[0])

# 6.2 Exclude the best feature and retrain the classifier
X_train_subset = X_train_prepared.drop(important_features[0], axis=1)
X_test_subset = X_test_prepared.drop(important_features[0], axis=1)
clf_subset = RandomForestClassifier(random_state=42)
clf_subset.fit(X_train_subset, y_train)

y_pred_test_subset = clf_subset.predict(X_test_subset)
print("Test Accuracy (Excluding Best Feature):", accuracy_score(y_test, y_pred_test_subset))

# Train the classifier
clf =RandomForestClassifier(random_state=42)
clf.fit(X_train_prepared, y_train)

feature_importance = clf.feature_importances_
important_features = X_train_prepared.columns[np.argsort(feature_importance)[::-1]]

print("Most Important Feature:", important_features[0])

# 6.2 Exclude the best feature and retrain the classifier
X_train_subset = X_train_prepared.drop(important_features[0], axis=1)
X_test_subset = X_test_prepared.drop(important_features[0], axis=1)

clf_subset = RandomForestClassifier(random_state=42)
clf_subset.fit(X_train_subset, y_train)

y_pred_test_subset = clf_subset.predict(X_test_subset)
print("Test Accuracy (Excluding Best Feature):", accuracy_score(y_test, y_pred_test_subset))
