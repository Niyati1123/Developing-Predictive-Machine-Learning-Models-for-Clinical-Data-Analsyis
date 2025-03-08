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
