# terrorist-attack-prediction-ML
 Predictive Analysis of Terrorist Activities using Machine Learning
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import folium

# 1. Load the dataset
df = pd.read_csv('globalterrorismdb_0718dist.csv', 
                 encoding='ISO-8859-1', 
                 low_memory=False)

# 2. Data Preprocessing
df = df.dropna(subset=['attacktype1_txt', 'country_txt', 'iyear', 'imonth', 'iday', 'longitude', 'latitude'])

le = LabelEncoder()
df['attacktype1'] = le.fit_transform(df['attacktype1_txt'])  
df['country'] = le.fit_transform(df['country_txt'])  

df['date'] = pd.to_datetime(df.apply(lambda row: f"{row['iyear']}-{row['imonth']:02d}-{row['iday']:02d}", axis=1),errors='coerce')
df['year'] = df['date'].dt.year
df['month'] = df['date'].dt.month
df['day_of_week'] = df['date'].dt.dayofweek

features = ['country', 'year', 'month', 'day_of_week', 'attacktype1', 'longitude', 'latitude']
X = df[features]
y = df['attacktype1']  

# Impute NaN values
imputer = SimpleImputer(strategy='mean')  
X_imputed = imputer.fit_transform(X)

# Standardization
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_imputed)

# Splitting the dataset
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

# Model Training
log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)
log_reg_pred = log_reg.predict(X_test)

dtree = DecisionTreeClassifier(random_state=42)
dtree.fit(X_train, y_train)
dtree_pred = dtree.predict(X_test)

rf = RandomForestClassifier(random_state=42)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)

# Model Evaluation
print("Logistic Regression Accuracy: ", accuracy_score(y_test, log_reg_pred))
print("Decision Tree Accuracy: ", accuracy_score(y_test, dtree_pred))
print("Random Forest Accuracy: ", accuracy_score(y_test, rf_pred))

# Classification Reports
print("\nLogistic Regression Classification Report:\n", classification_report(y_test, log_reg_pred))
print("\nDecision Tree Classification Report:\n", classification_report(y_test, dtree_pred))
print("\nRandom Forest Classification Report:\n", classification_report(y_test, rf_pred))

# Confusion Matrix
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

sns.heatmap(confusion_matrix(y_test, log_reg_pred), annot=True, fmt='d', cmap='Blues', ax=axes[0])
axes[0].set_title("Logistic Regression")

sns.heatmap(confusion_matrix(y_test, dtree_pred), annot=True, fmt='d', cmap='Blues', ax=axes[1])
axes[1].set_title("Decision Tree")

sns.heatmap(confusion_matrix(y_test, rf_pred), annot=True, fmt='d', cmap='Blues', ax=axes[2])
axes[2].set_title("Random Forest")

plt.show()

# Distribution of attack types
plt.figure(figsize=(12, 6))
df['attacktype1_txt'].value_counts().plot(kind='bar', color='teal')
plt.title('Distribution of Attack Types', fontsize=16)
plt.xlabel('Attack Type', fontsize=14)
plt.ylabel('Number of Attacks', fontsize=14)
plt.xticks(rotation=45, ha='right')
plt.show()
# Top 10 countries with the most attacks
plt.figure(figsize=(12, 6))
df['country_txt'].value_counts().head(10).plot(kind='bar', color='purple')
plt.title('Top 10 Countries with Most Attacks', fontsize=16)
plt.xlabel('Country', fontsize=14)
plt.ylabel('Number of Attacks', fontsize=14)
plt.xticks(rotation=45, ha='right')
plt.show()

# Correlation heatmap
plt.figure(figsize=(10, 8))
corr_matrix = df[['year', 'month', 'day_of_week', 'longitude', 'latitude', 'attacktype1']].corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix', fontsize=16)
plt.show()
