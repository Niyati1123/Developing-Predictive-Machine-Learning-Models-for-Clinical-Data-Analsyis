#1 Importing Needed Libraries 

import pandas as pd
import seaborn as sns
import numpy as np
from matplotlib import pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, IterativeImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.model_selection import cross_val_score
from scipy import stats
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from keras.models import Model
from keras.layers import Input, Dense
import missingno as msno

#2 Loading and Exploring the Data

patient_df = pd.read_csv ('patients_01.csv')
#Exploring the Dataset
print(patient_df.describe())
print(patient_df.info())
# Number of samples
num_rows = len(patient_df)
print(num_rows)
# Identify the number of missing values in each feature
missing_values = patient_df.isnull().sum()
print(missing_values)
# Calculating the percentage of missing values for each feature
round(100*(1-patient_df.count()/len(patient_df)),2)
# Extracting samples with 3 or more missing values
patient_df.loc[patient_df.isnull().sum(axis=1)>=3, :]
# Extracting samples with 4 or more missing values
patient_df.loc[patient_df.isnull().sum(axis=1)>=4, :]

#3 Data Visualisation 

#3.1 Plotting the distribution of age
plt.figure(figsize=(10, 6))
# Plotting the histogram with specified bins and color
plt.hist(patient_df['age'], bins=20, color="seagreen",alpha = 0.7, edgecolor="black")
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.title('Distribution of Age')
plt.grid(axis='y', linestyle='--')
plt.show()

# Plotting the distribution of BMI
plt.figure(figsize=(10, 6))
#3.2 Plotting the histogram with specified bins and color
plt.hist(patient_df['bmi'], bins=20, color="darkorange", alpha = 0.8,edgecolor="black")
plt.xlabel('BMI')
plt.ylabel('Frequency')
plt.title('Distribution of BMI')
plt.grid(axis='y', linestyle='--')
plt.show()

#3.3 Plotting the distribution of drinking alcohol
plt.figure(figsize=(10, 6))
# Plotting the histogram with specified bins and color
plt.hist(patient_df['alcohol_misuse'], bins=20, color="steelblue",alpha = 0.9, edgecolor="black")
plt.xlabel('Amount of Drinking Alcohol')
plt.ylabel('Frequency')
plt.title('Distribution of Drinking Alcohol')
plt.grid(axis='y', linestyle='--')
plt.show()

#3.4 Plotting the distribution of having mental health issues
plt.figure(figsize=(10, 6))
# Plotting the histogram with specified bins and color
plt.hist(patient_df['health_ment'], bins=20, color="brown",alpha = 0.7, edgecolor="black")
plt.xlabel('Mental Health')
plt.ylabel('Frequency')
plt.title('Distribution of Mental Health Issues')
plt.grid(axis='y', linestyle='--')
plt.show()

#3.5 Plotting the distribution of having general health
plt.figure(figsize=(10, 6))
# Plotting the histogram with specified bins and color
plt.hist(patient_df['health_gen'], bins=10, color="red", alpha=0.7, edgecolor="black")
plt.xlabel('General Health')
plt.ylabel('Frequency')
plt.title('Distribution of General Health')
plt.grid(axis='y', linestyle='--')
plt.show()

#3.6 Plotting the distribution of having healthy body
plt.figure(figsize=(10, 6))
# Plotting the histogram with specified bins, color, transparency, and edgecolor
plt.hist(patient_df['health_phys'], bins=15, color="purple", alpha=0.7, edgecolor="black")
plt.xlabel('Physical Health')
plt.ylabel('Frequency')
plt.title('Distribution of Physical Health')
plt.grid(axis='y', linestyle='--')
plt.show()

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


