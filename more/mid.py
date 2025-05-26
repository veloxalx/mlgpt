import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

#Load data (e.g., from Kaggle's Fake News dataset)
df = pd.read_csv("fake_or_real_news.csv") # Must contain 'text' and 'label' columns
X = df['text']
y = df['label'] # Labels: 'FAKE' or 'REAL'

#Preprocessing with TF-IDF
vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
X_vec = vectorizer.fit_transform(X)

#Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_vec, y, test_size=0.2, random_state=42)

#Train model
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

#Predict and evaluate
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))