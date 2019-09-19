import numpy as np
import re
import nltk
import pickle
import pandas as pd
from nltk.stem import WordNetLemmatizer
import sys
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


train_df = pd.read_csv('train_df.csv')
test_df = pd.read_csv('test_df.csv')

X_train = train_df['comment']
y_train = train_df['classification']

X_test = test_df['comment']
y_test = test_df['classification']

def clean(data):
    lem = WordNetLemmatizer()
    documents = []
    for sen in range(0, len(data)):
        document = re.sub(r'\W', ' ', str(data[sen]))
        document = re.sub(r'\s+[a-zA-Z]\s+', ' ', document)
        document = re.sub(r'\^[a-zA-Z]\s+', ' ', document)
        document = re.sub(r'\s+', ' ', document, flags=re.I)
        document = re.sub(r'^b\s+', '', document)
        document = document.lower()
        document = document.split()
        document = [lem.lemmatize(word) for word in document]
        document = ' '.join(document)
        documents.append(document)
    return documents
def vectorize(data):
    tfidfconverter = TfidfVectorizer(max_features=1500, min_df=5, max_df=0.7, stop_words=stopwords.words('english'))
    data = tfidfconverter.fit_transform(data).toarray()
    return data

X_train = clean(X_train)
X_train = vectorize(X_train)

X_test = clean(X_test)
X_test = vectorize(X_test)

classifier = RandomForestClassifier(n_estimators=1000, random_state=0)
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))
print(accuracy_score(y_test, y_pred))

with open('text_classifier', 'wb') as picklefile:
    pickle.dump(classifier,picklefile)
