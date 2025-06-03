# Basic Libraries
import pandas as pd
import numpy as np

# Visualization Libraries
import matplotlib.pyplot as plt
import seaborn as sns

# Preprocessing
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler

# Model Selection
from sklearn.model_selection import train_test_split, GridSearchCV

# Machine Learning Algorithms
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

# Evaluation Metrics
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score

# Suppress warnings
import warnings
warnings.filterwarnings("ignore")

# Set seaborn style
sns.set(style="whitegrid")

import pandas as pd

# Load the dataset (replace with actual file name if different)
df = pd.read_csv("Telco-Customer-Churn.csv")

# Show shape and first few rows
print("Dataset Shape:", df.shape)
df.head()

# Check for missing values
print("Missing values per column:")
print(df.isnull().sum())

# Also check for blank strings which may be "missing" but not null
print("\nBlank strings per column:")
print((df == ' ').sum())

# Check for duplicate rows
duplicate_count = df.duplicated().sum()
print(f"\nNumber of duplicate rows: {duplicate_count}")

import matplotlib.pyplot as plt
import seaborn as sns

# Convert 'TotalCharges' to numeric if needed
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

# Select numeric columns
numeric_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']

# Create boxplots
plt.figure(figsize=(15, 4))
for i, col in enumerate(numeric_cols):
    plt.subplot(1, 3, i+1)
    sns.boxplot(y=df[col])
    plt.title(f'Boxplot of {col}')
plt.tight_layout()
plt.show()

# Drop the customerID column
df.drop('customerID', axis=1, inplace=True)

df.dtypes

# Convert 'TotalCharges' to numeric (invalid parsing will be set as NaN)
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

# Check number of missing values before dropping
print("Missing values before dropping:", df.isnull().sum().sum())

# Drop rows with missing values
df.dropna(inplace=True)

# Check shape after dropping
print("New dataset shape after dropping missing values:", df.shape)

# Dataset structure
print(df.info())

# Statistical summary of numerical features
print(df.describe())

# Distribution of target variable
print("\nChurn value counts:")
print(df['Churn'].value_counts())

# Bar plot of churn distribution
plt.figure(figsize=(6, 4))
sns.countplot(data=df, x='Churn', palette='pastel')
plt.title('Churn Distribution')
plt.show()

# List of key categorical features
cat_features = ['gender', 'SeniorCitizen', 'Partner', 'Dependents',
                'PhoneService', 'InternetService', 'Contract',
                'PaymentMethod', 'PaperlessBilling']

# Plot countplots for each categorical feature vs Churn
plt.figure(figsize=(18, 20))
for i, col in enumerate(cat_features, 1):
    plt.subplot(4, 3, i)
    sns.countplot(data=df, x=col, hue='Churn', palette='pastel')
    plt.title(f'{col} vs Churn')
    plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# List of numeric features
num_features = ['tenure', 'MonthlyCharges', 'TotalCharges']

# Boxplots comparing numeric features by Churn
plt.figure(figsize=(18, 5))
for i, col in enumerate(num_features, 1):
    plt.subplot(1, 3, i)
    sns.boxplot(data=df, x='Churn', y=col, palette='pastel')
    plt.title(f'{col} vs Churn')
plt.tight_layout()
plt.show()

# Compute correlation matrix
corr_matrix = df[['tenure', 'MonthlyCharges', 'TotalCharges']].corr()

# Heatmap
plt.figure(figsize=(6, 4))
sns.heatmap(corr_matrix, annot=True, cmap='Blues')
plt.title('Correlation Matrix')
plt.show()

# Identify categorical columns
cat_cols = df.select_dtypes(include=['object']).columns.tolist()



print("Categorical columns to encode:", cat_cols)

from sklearn.preprocessing import LabelEncoder

# Initialize label encoder
le = LabelEncoder()

# Apply to all categorical columns
for col in cat_cols:
    df[col] = le.fit_transform(df[col])

print("Encoding complete.")

# Check the transformed dataset
df.head()

# Compute correlation matrix
corr = df.corr()

# Set up the matplotlib figure
plt.figure(figsize=(14, 10))

# Draw the heatmap
sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm', square=True, cbar_kws={"shrink": 0.8})

plt.title("Correlation Heatmap of All Features", fontsize=16)
plt.tight_layout()
plt.show()

# Compute correlation with the target variable
correlation_with_target = df.corr()['Churn'].drop('Churn')

# Filter features with correlation > 0.15 or < -0.15
selected_features = correlation_with_target[correlation_with_target.abs() > 0.19].index.tolist()

# Show selected features
print("Selected features based on correlation threshold (|r| > 0.19):")
print(selected_features)

# Define new feature set
X = df[selected_features]
y = df['Churn']

# Split the filtered features into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Scale for KNN
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report

# Initialize and train Decision Tree
dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train, y_train)

# Predict
dt_preds = dt_model.predict(X_test)

# Evaluate
print("----- Decision Tree Evaluation (Filtered Features) -----")
print("Accuracy:", accuracy_score(y_test, dt_preds))
print("F1 Score:", f1_score(y_test, dt_preds))
print("\nClassification Report:\n", classification_report(y_test, dt_preds))
print("Confusion Matrix:\n", confusion_matrix(y_test, dt_preds))

from sklearn.neighbors import KNeighborsClassifier

# Initialize and train KNN
knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_train_scaled, y_train)

# Predict
knn_preds = knn_model.predict(X_test_scaled)

# Evaluate
print("----- KNN Evaluation (Filtered Features) -----")
print("Accuracy:", accuracy_score(y_test, knn_preds))
print("F1 Score:", f1_score(y_test, knn_preds))
print("\nClassification Report:\n", classification_report(y_test, knn_preds))
print("Confusion Matrix:\n", confusion_matrix(y_test, knn_preds))

from sklearn.metrics import roc_curve, auc

# For Decision Tree
y_proba_dt = dt_model.predict_proba(X_test)[:,1]
fpr_dt, tpr_dt, _ = roc_curve(y_test, y_proba_dt)
roc_auc_dt = auc(fpr_dt, tpr_dt)

# For KNN
y_proba_knn = knn_model.predict_proba(X_test_scaled)[:,1]
fpr_knn, tpr_knn, _ = roc_curve(y_test, y_proba_knn)
roc_auc_knn = auc(fpr_knn, tpr_knn)

# Plot ROC Curves
plt.figure(figsize=(8, 6))
plt.plot(fpr_dt, tpr_dt, label=f'Decision Tree (AUC = {roc_auc_dt:.2f})')
plt.plot(fpr_knn, tpr_knn, label=f'KNN (AUC = {roc_auc_knn:.2f})')
plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves for DT and KNN')
plt.legend()
plt.grid()
plt.show()

