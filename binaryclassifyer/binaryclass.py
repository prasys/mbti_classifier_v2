import numpy as np
import pandas as pd

import tensorflow as tf
import tensorflow_hub as hub

from sklearn.preprocessing import MultiLabelBinarizer

train_df = pd.read_csv('train_df.csv')
test_df = pd.read_csv('test_df.csv')

train_comments = train_df['comment']
train_classification = train_df['classification']

test_comments = test_df['comment']
test_classification = test_df['classification']

comment_embeddings = hub.text_embedding_column("comments",module_spec="https://tfhub.dev/google/universal-sentence-encoder/2")

encoder = MultiLabelBinarizer()
encoder.fit_transform(train_classification)
train_encoded = encoder.transform(train_classification)
test_encoded = encoder.transform(test_classification)
num_classes = len(encoder.classes_)

print(encoder.classes_)
