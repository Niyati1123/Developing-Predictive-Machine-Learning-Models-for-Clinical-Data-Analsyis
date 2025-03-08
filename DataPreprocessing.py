#4 Correlation Analysis

* We want to see if there is any correlation between the features (Not all the feature are numeric so we can only use the numeric ones)
* The code is generating a heatmap and the results will include numbers: these numbers shows the correlation coefficient between the corresponding pair of features
* The colors are showing the the magnitude of the correlation coefficient. Darker colors represent stronger correlations, while lighter colors represent weaker correlations

correlation_matrix = patient_df.corr(numeric_only=True)
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title('Correlation Matrix')
plt.show()

#5 Train and Test Split 

X = patient_df.drop(['dissease'], axis=1)
y = patient_df['dissease']
X_train, X_test, y_train, y_test = train_test_split(X,y,random_state=15)

X_train

#5.1 Dealing with Missing Values

# Dealing with missing data in patient_df:
# Identify numeric and non-numeric columns
numeric_columns_df = patient_df.select_dtypes(include='number').columns
non_numeric_columns_df = patient_df.columns.difference(numeric_columns_df)
# Convert numeric columns to numeric type
patient_df[numeric_columns_df] = patient_df[numeric_columns_df].apply(pd.to_numeric, errors='coerce')
# Handle non-numeric columns (drop them for simplicity)
patient_df = patient_df.drop(columns=non_numeric_columns_df)
# Fill missing values for numeric columns
patient_df.loc[:, numeric_columns_df] = patient_df[numeric_columns_df].fillna(patient_df[numeric_columns_df].mean())

#5.2 Dealing with missing data in the X_train and X_test:

# Imputation
numeric_columns = X_train.select_dtypes(include=['number']).columns
categorical_columns = X_train.columns.difference(numeric_columns)

imputer_numeric = SimpleImputer(strategy='mean')
imputer_categorical = SimpleImputer(strategy='most_frequent')

X_train_numeric = pd.DataFrame(imputer_numeric.fit_transform(X_train[numeric_columns]), columns=numeric_columns)
X_train_categorical = pd.DataFrame(imputer_categorical.fit_transform(X_train[categorical_columns]), columns=categorical_columns)

X_test_numeric = pd.DataFrame(imputer_numeric.transform(X_test[numeric_columns]), columns=numeric_columns)
X_test_categorical = pd.DataFrame(imputer_categorical.transform(X_test[categorical_columns]), columns=categorical_columns)

# Concatenating imputed data
X_train_imputed = pd.concat([X_train_numeric, X_train_categorical], axis=1)
X_test_imputed = pd.concat([X_test_numeric, X_test_categorical], axis=1)

# One-Hot Encoding for categorical variables
encoder = OneHotEncoder(sparse=False)
X_train_encoded = pd.DataFrame(encoder.fit_transform(X_train_imputed[categorical_columns]), columns=encoder.get_feature_names_out())
X_test_encoded = pd.DataFrame(encoder.transform(X_test_imputed[categorical_columns]), columns=encoder.get_feature_names_out())

X_train_prepared = pd.concat([X_train_imputed.drop(categorical_columns, axis=1), X_train_encoded], axis=1)
X_test_prepared = pd.concat([X_test_imputed.drop(categorical_columns, axis=1), X_test_encoded], axis=1)

# Standardise data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_prepared)
X_test_scaled = scaler.transform(X_test_prepared)

# PCA
pca = PCA(n_components=2)
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

# Visualising PCA results
plt.scatter(X_train_pca[:, 0], X_train_pca[:, 1], c=y_train, cmap='viridis')
plt.title('PCA Analysis - Training Data')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.show()

plt.scatter(X_test_pca[:, 0], X_test_pca[:, 1], c=y_test, cmap='plasma')
plt.title('PCA Analysis - Testing Data')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.show()

X_train_prepared
X_train_encoded
categorical_columns

# Outliers Removal

# Outlier Removal on patient_df
# Performing outlier detection using Z-scores
p_num = patient_df[['age', 'bmi', 'alcohol_misuse', 'health_gen', 'health_ment', 'health_phys']]
z1 = np.abs(stats.zscore(p_num))
print('\nZ-Score Array:\n', z1)
print(p_num.shape)
threshold = 3
print('\nOutliers:\n', np.where(z1 > threshold))

p_num

# Ploting the all the  outliers
plt.figure(figsize=(10,7))
sns.boxplot(data=p_num)
plt.title('Outlier Visualization', fontsize=20)
plt.xticks(rotation=90)
plt.show()

# Removing outliers
p_outliers = (z1 < threshold).all(axis=1)
print(p_outliers.shape)
print('\nAfter Removing Outliers:\n', np.where(z1 < threshold))
p_outliers_removed = p_num[p_outliers]

p_outliers

p_outliers_removed

# Outlier Removal on X_train
# Splitting a training dataset
X_train_num = X_train_imputed[['age', 'bmi', 'alcohol_misuse', 'health_gen', 'health_ment', 'health_phys']]
X_train_cat = X_train_imputed[['gender', 'high_bp', 'high_chol', 'chol_check', 'history_smoking', 'history_stroke', 'history_heart_disease', 'amount_activity', 'fruits', 'vegetables', 'walking_diff']]
X_train_num

# Calculating the Z-scores for each element in the X_train_num
z1 = np.abs(stats.zscore(X_train_num))
print('\nZ-Score Array:\n', z1)
print(X_train_num.shape)
threshold = 3
print('\nOutliers:\n', np.where(z1 < threshold))

# Identifying outliers in X_train
X_train_outliers = (z1 < threshold).all(axis=1)
print(X_train_outliers.shape)
X_train_outliers

# Remove the outliers in X_train
X_train_outliers_removed = X_train_num[X_train_outliers]
print('\nAfter Removing Outliers:\n', np.where(z1 < threshold))

X_train_outliers_removed

# Ploting the train data outliers
plt.figure(figsize=(20,10))
sns.boxplot(data=X_train_num)
plt.title('Trian Data Outlier Visualization', fontsize=20)
plt.xticks(rotation=90)
plt.show()

print('\nAfter Removing Outliers:\n', np.where(z1 < threshold))

# Ploting the train data after removing the outliers
plt.figure(figsize=(20,10))
sns.boxplot(data=X_train_outliers_removed)
plt.title('Train Data After Outlier Removal', fontsize=20)
plt.xticks(rotation=90)
plt.show()

# Outlier Removal on X_test

# Calculating the Z-scores for each element in the X_test_num
X_test_num = X_test_imputed[['age', 'bmi', 'alcohol_misuse', 'health_gen', 'health_ment', 'health_phys']]
X_test_cat = X_test_imputed[['gender', 'high_bp', 'high_chol', 'chol_check', 'history_smoking', 'history_stroke', 'history_heart_disease', 'amount_activity', 'fruits', 'vegetables', 'walking_diff']]
z2 = np.abs(stats.zscore(X_test_prepared))
print('\nZ-Score Array:\n', z2)
print(X_test_num.shape)
threshold = 3
print('\nOutliers:\n', np.where(z2 < threshold))

plt.figure(figsize=(20,10))
sns.boxplot(data=X_test_num)
plt.title('Test Data Outlier Visualization', fontsize=20)
plt.xticks(rotation=90)
plt.show()

# Identifying outliers in X_test
X_test_outliers = (z2 < threshold).all(axis=1)
print(X_test_outliers.shape)
z2 = np.abs(stats.zscore(X_test_outliers))
print('\nAfter Removing Outliers:\n', np.where(z2 < threshold))

X_test_outliers_removed = X_test_prepared[X_test_outliers]
print('\nAfter Removing Outliers:\n', np.where(z1 < threshold))

X_test_outliers_removed

plt.figure(figsize=(20,10))
sns.boxplot(data=X_test_outliers_removed)
plt.title('Test Data After Outlier Removal', fontsize=20)
plt.xticks(rotation=90)
plt.show()

