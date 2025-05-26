import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import nltk
nltk.download('movie_reviews')
from nltk.corpus import movie_reviews
import random

# Load movie review data
docs = [(list(movie_reviews.words(fileid)), category)
        for category in movie_reviews.categories()
        for fileid in movie_reviews.fileids(category)]
random.shuffle(docs)

# Split into text and labels
texts = [' '.join(doc) for doc, label in docs]
labels = [label for doc, label in docs]

# Vectorize
vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
X = vectorizer.fit_transform(texts)
y = labels

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Model
clf = LogisticRegression()
clf.fit(X_train, y_train)
predictions = clf.predict(X_test)

# Report
print(classification_report(y_test, predictions))
