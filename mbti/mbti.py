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

df = df[pd.notnull(df['posts'])]

col = ['type', 'posts']
df = df[col]

df.columns = ['type', 'posts']

df['category_id'] = df['type'].factorize()[0]
category_id_df = df[['type', 'category_id']].drop_duplicates().sort_values('category_id')
category_to_id = dict(category_id_df.values)
id_to_category = dict(category_id_df[['category_id', 'type']].values)

# fig = plt.figure(figsize=(8,6))
# df.groupby('type')['posts'].count().plot.bar(ylim=0)
# plt.show()

tfidf = TfidfVectorizer(sublinear_tf=True, min_df=5, norm=None, encoding='latin-1', ngram_range=(1,2), stop_words='english')

features = tfidf.fit_transform(df['posts']).toarray()
labels = df.category_id
# print(features.shape)

# N = 2
# for type, category_id in sorted(category_to_id.items()):
#     features_chi2 = chi2(features, labels == category_id)
#     indices = np.argsort(features_chi2[0])
#     feature_names = np.array(tfidf.get_feature_names())[indices]
#     unigrams = [v for v in feature_names if len(v.split(' ')) == 1]
#     bigrams = [v for v in feature_names if len(v.split(' ')) == 2]
    # print("# '{}':".format(type))
    # print("  . Most correlated unigrams:\n       . {}".format('\n       . '.join(unigrams[-N:])))
    # print("  . Most correlated bigrams:\n       . {}".format('\n       . '.join(bigrams[-N:])))

X_train, X_test, y_train, y_test = train_test_split(df['posts'], df['type'], random_state = 0)
count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(X_train)
tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)

clf = MultinomialNB().fit(X_train_tfidf, y_train)

print(clf.predict(count_vect.transform(["Okay, Makes sense. This is all cool and interesting! If you would like any assistance that you believe Optimize can do, let us know!"])))
