from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import pandas as pd

#Load dataset
iris = load_iris()
X = iris.data
y = iris.target

#Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

#Train model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

#Predict
y_pred = model.predict(X_test)

#Evaluate
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred, target_names=iris.target_names))