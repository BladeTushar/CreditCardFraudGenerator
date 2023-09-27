import numpy as np
import pandas as pd

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

"""# IMPORT LIBRARIES"""

import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import gridspec

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (classification_report, accuracy_score,
                            precision_score, recall_score,
                            f1_score, matthews_corrcoef,
                            confusion_matrix)

"""# LOAD AND PREPARE DATA"""

credit = pd.read_csv('/kaggle/input/creditcardfraud/creditcard.csv')
credit.head()

# Number of rows and cols
credit.shape

# Descriptive statistical measures of the dataset
credit.describe().T

"""# EXPLORATORY DATA ANALYSIS"""

# Determine number of fraud cases in the dataset
fraud = credit[credit['Class'] == 1]
valid = credit[credit['Class'] == 0]
fraction = len(fraud)/float(len(valid))

print(fraction)
print("Fraud Cases: {}".format(len(credit[credit['Class'] == 1])))
print("Valid Cases: {}".format(len(credit[credit['Class'] == 0])))

"""* Only **0.17%** fraudulent transaction out all the transactions. The data is highly **Unbalanced**."""

# Print the amount of details for Fraudulent Transaction
print("Amount of details for the Fraudulent Transaction")
fraud.Amount.describe()

# Print the amount of details for Normal Transaction
print("Amount of details for Normal Transaction")
valid.Amount.describe()

# Plotting the Correlation Matrix
corrmat = credit.corr()
fig = plt.figure(figsize=(12, 9))
sns.heatmap(corrmat, vmax=.8, square=True)
plt.show()

"""# MODEL DEVELOPMENT"""

# dividing X and Y from the dataset
X = credit.drop(['Class'], axis=1 )
Y = credit['Class']

print(X.shape)
print(Y.shape)

X_credit = X.values
Y_credit = Y.values

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Build RandomForestClassifier
rfc = RandomForestClassifier()
rfc.fit(X_train, y_train)

y_pred = rfc.predict(X_test)

"""# MODEL EVALUATION"""

# Evaluating the classifier
n_outliers = len(fraud)
n_errors = (y_pred != y_test).sum()
print("The model used is RandomForestClassifier")

acc = accuracy_score(y_test, y_pred)
print(f"The accuracy is {acc}")

prec = precision_score(y_test, y_pred)
print(f"The precision score is {prec}")

rec = recall_score(y_test, y_pred)
print(f"The recall score is {rec}")

f1 = f1_score(y_test, y_pred)
print(f"The f1 score is {f1}")

MCC = matthews_corrcoef(y_test, y_pred)
print(f"The Matthews correlation coeficient is {MCC}")

# Vizualize the confusion matrix
LABELS = ['Normal', 'Fraud']
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(12,12))
sns.heatmap(conf_matrix, xticklabels = LABELS, yticklabels = LABELS, annot=True, fmt='d')
plt.title('Confusion Matrix')
plt.ylabel('True class')
plt.xlabel('Predicted class')
plt.show()