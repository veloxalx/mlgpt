import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.impute import SimpleImputer

url = 'https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv'
df = pd.read_csv(url)

df = df[['Survived', 'Pclass', 'Sex', 'Age', 'Fare']]
df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
imp = SimpleImputer()
df[['Age', 'Fare']] = imp.fit_transform(df[['Age', 'Fare']])

X = df.drop('Survived', axis=1)
y = df['Survived']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

clf = RandomForestClassifier()
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred))