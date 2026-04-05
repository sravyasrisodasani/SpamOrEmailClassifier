import numpy as np
import pandas as pd
import nltk
import string
import re
import pickle

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, confusion_matrix
from sklearn.naive_bayes import MultinomialNB

# ---------- DOWNLOAD ----------
nltk.download('stopwords')
nltk.download('punkt')

# ---------- LOAD DATA ----------
df = pd.read_csv('spam.csv', encoding='latin-1')

# ---------- CLEANING ----------
df.drop(columns=['Unnamed: 2','Unnamed: 3','Unnamed: 4'], inplace=True)
df.rename(columns={'v1':'target','v2':'text'}, inplace=True)

df['target'] = df['target'].map({'ham':0, 'spam':1})

# remove duplicates
df.drop_duplicates(inplace=True)

# ---------- PREPROCESS ----------
ps = PorterStemmer()
stop_words = set(stopwords.words('english'))
def transform_text(text):
    text = text.lower()

    words = text.split()

    filtered = []
    for word in words:
        word = re.sub(r'[^a-zA-Z0-9]', '', word)

        if word and word not in stop_words:
            filtered.append(ps.stem(word))

    return " ".join(filtered)
df['transformed_text'] = df['text'].apply(transform_text)

# ---------- FEATURE ----------
tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1,2))
X = tfidf.fit_transform(df['transformed_text']).toarray()

y = df['target']

# ---------- SPLIT ----------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=2
)

# ---------- MODEL (MNB) ----------
model = MultinomialNB()
model.fit(X_train, y_train)

# ---------- EVALUATION ----------
y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# ---------- SAVE ----------
pickle.dump(tfidf, open('vectorizer.pkl','wb'))
pickle.dump(model, open('model.pkl','wb'))

print("✅ Model saved successfully!")