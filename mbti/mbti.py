import pandas as pd
from io import StringIO
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import chi2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB

df = pd.read_csv('mbti_1.csv')

df = df[pd.notnull(df['comment'])]

col = ['type', 'comment']
df = df[col]

df.columns = ['type', 'comment']

df['category_id'] = df['type'].factorize()[0]
category_id_df = df[['type', 'category_id']].drop_duplicates().sort_values('category_id')
category_to_id = dict(category_id_df.values)
id_to_category = dict(category_id_df[['category_id', 'type']].values)

# imbalanced dataset
# fig = plt.figure(figsize=(8,6))
# df.groupby('type')['comment'].count().plot.bar(ylim=0)
# plt.show()

tfidf = TfidfVectorizer(sublinear_tf=True, min_df=5, norm=None, encoding='latin-1', ngram_range=(1,2), stop_words='english')

features = tfidf.fit_transform(df['comment']).toarray()
labels = df.category_id

X_train, X_test, y_train, y_test = train_test_split(df['comment'], df['type'], random_state = 0)
count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(X_train)
tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)

clf = MultinomialNB().fit(X_train_tfidf, y_train)

# use this to predict text:
# print(clf.predict(count_vect.transform(["PLACEHOLDER TEXT"])))
