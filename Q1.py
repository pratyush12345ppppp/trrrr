# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load the Titanic dataset
titanic_data = pd.read_csv('titanic.csv')

# Preprocess the data
# Handle missing values
titanic_data['Age'].fillna(titanic_data['Age'].mean(), inplace=True)
titanic_data['Embarked'].fillna('S', inplace=True)

# Convert categorical variables to numerical variables
le = LabelEncoder()
titanic_data['Sex'] = le.fit_transform(titanic_data['Sex'])
titanic_data['Embarked'] = le.fit_transform(titanic_data['Embarked'])

# Drop unnecessary columns
titanic_data.drop(['Name', 'Ticket', 'Cabin'], axis=1, inplace=True)

# Perform exploratory data analysis
print(titanic_data.head())
print(titanic_data.info())
print(titanic_data.describe())

# Visualize the data
plt.figure(figsize=(10, 6))
plt.subplot(1, 2, 1)
titanic_data['Survived'].value_counts().plot(kind='bar')
plt.title('Survival Count')

plt.subplot(1, 2, 2)
titanic_data['Pclass'].value_counts().plot(kind='bar')
plt.title('Passenger Class Count')
plt.show()

# Split the data into features (X) and target (y)
X = titanic_data.drop(['Survived'], axis=1)
y = titanic_data['Survived']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build the decision tree classifier model
dtc = DecisionTreeClassifier(random_state=42)
dtc.fit(X_train, y_train)

# Evaluate the model
y_pred = dtc.predict(X_test)
print('Accuracy:', accuracy_score(y_test, y_pred))
print('Classification Report:')
print(classification_report(y_test, y_pred))
print('Confusion Matrix:')
print(confusion_matrix(y_test, y_pred))     
