import pandas as pd
from io import StringIO
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import chi2
from numpy import *
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from nltk import tokenize

df = pd.read_csv('dataset.csv')

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
my_input = "I started drinking a bottle of wine close to bedtime. I just wasnt feeling anything though, so I chugged the rest. I went to sleep feeling not too messed up. Some time during the night I woke up have to pee. Instead of stumbling to the toilet, I went to the laundry room and peed in the washing machine. It must have been hard since it's a top-loader. The next day I went to do laundry and the smell was sooo bad! It took me a minute to realize I peed in the washer. I thought that was a dream! Nope. It was real. I did a wash cycle with no clothes in there but a lot of bleach. It smelled fine after that. I'm not really sure why I peed in there. Did I think it would be funny? Did I confuse a white washing machine with a white toilet?? Who knows."

# lines 50-59 seperates my_input into a list of sentances and types each one with the model.
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

print(types)

### optional graph of the datasett
# imbalanced dataset
# fig = plt.figure(figsize=(8,6))
# df.groupby('type')['comment'].count().plot.bar(ylim=0)
# plt.show()
