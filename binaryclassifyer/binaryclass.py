import numpy as np
import pandas as pd

import tensorflow as tf
import tensorflow_hub as hub

from sklearn.preprocessing import MultiLabelBinarizer
from nltk import tokenize

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

multi_label_head = tf.contrib.estimator.multi_label_head(num_classes,loss_reduction=tf.losses.Reduction.SUM_OVER_BATCH_SIZE)
estimator = tf.contrib.estimator.DNNEstimator(head=multi_label_head,hidden_units=[64,10],feature_columns=[comment_embeddings])
features = {"comments": np.array(train_comments)}
labels = np.array(train_encoded)
train_input_fn = tf.estimator.inputs.numpy_input_fn(features,labels,shuffle=True, batch_size=32,num_epochs=20)
estimator.train(input_fn=train_input_fn)
eval_input_fn = tf.estimator.inputs.numpy_input_fn({"comments": np.array(test_comments).astype(np.str)}, test_encoded.astype(np.int32), shuffle=False)

print(estimator.evaluate(input_fn=eval_input_fn))
