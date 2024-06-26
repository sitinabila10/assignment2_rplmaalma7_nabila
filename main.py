import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

data = pd.read_csv('ai4i2020.csv')

data = data.drop(["UDI","Product ID","TWF","HDF","PWF","OSF","RNF"], axis=1)

real_type = data['Type']

encoder = LabelEncoder()
data['Type'] = encoder.fit_transform(data['Type'])

X = data.drop('Machine failure', axis=1)  
y = data['Machine failure']  

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.45, random_state=42)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
result = pd.DataFrame(X_test)
result['Type'] = real_type
result['Machine failure'] = y_pred

print("Prediction Result:")
print(result.head())

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")

conf_matrix = confusion_matrix(y_test, y_pred)
print(f"Confusion Matrix:\n{conf_matrix}")

class_report = classification_report(y_test, y_pred)
print(f"Classification Report:\n{class_report}")

plt.scatter(range(len(y_pred)), y_pred)
plt.show()