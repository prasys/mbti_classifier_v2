import pandas as pd
from io import StringIO
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import chi2
from numpy import *
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from nltk import tokenize
import pprint
import matplotlib.pyplot as plt

df = pd.read_csv('balancedDataset.csv')

df = df[pd.notnull(df['comment'])]

col = ['type', 'comment']
df = df[col]

df.columns = ['type', 'comment']

df['category_id'] = df['type'].factorize()[0]
category_id_df = df[['type', 'category_id']].drop_duplicates().sort_values('category_id')
category_to_id = dict(category_id_df.values)
id_to_category = dict(category_id_df[['category_id', 'type']].values)

tfidf = TfidfVectorizer(sublinear_tf=True, min_df=5, norm=None, encoding='latin-1', ngram_range=(1,2), stop_words='english')

# vectorizes each comment in the dataset
features = tfidf.fit_transform(df['comment']).toarray()
labels = df.category_id

X_train, X_test, y_train, y_test = train_test_split(df['comment'], df['type'], random_state = 0)
count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(X_train)
tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)

# trains the model
clf = MultinomialNB().fit(X_train_tfidf, y_train)

# use this to predict text:
my_input = "I def recommend flying, tickets are cheap on spirit, even round trip. I can help pay too. Weed isn’t legal in NY i don’t think, but I still have a good amount of edibles left"
# Seperates my_input into a list of sentances and types each one with the model.
# Then we list out the models predictions for each sentence.
paragraph = tokenize.sent_tokenize(my_input)
types = []
for sentance in paragraph:
    type = array2string(clf.predict(count_vect.transform([sentance])))
    type = type.replace("[","")
    type = type.replace("]","")
    type = type.replace("'","")
    type = type.replace("'","")
    types.append(type)

def softmax(list):
    typeCount = {
        'ENTJ' : 0,
        'ENTP' : 0,
        'ENFJ' : 0,
        'ENFP' : 0,
        'ESTJ' : 0,
        'ESTP' : 0,
        'ESFJ' : 0,
        'ESFP' : 0,
        'INTJ' : 0,
        'INTP' : 0,
        'INFJ' : 0,
        'INFP' : 0,
        'ISTJ' : 0,
        'ISTP' : 0,
        'ISFJ' : 0,
        'ISFP' : 0,
    }
    for item in list:
        typeCount[item] += 1
    sum = 0
    for i in typeCount:
        sum += typeCount[i]
    for i in typeCount:
        typeCount[i] = str(round(typeCount[i]/sum,2)*100) + "%"

    plt.bar(*zip(*typeCount.items()))
    plt.show()
softmax(types)
