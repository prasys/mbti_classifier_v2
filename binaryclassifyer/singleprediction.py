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

with open('text_classifier', 'rb') as training_model:
    model = pickle.load(training_model)

print(model.predict(["I have a question : My problem statement is to classify the conversation of business people into selling,buying,insurance , etc...Can i implement that using your model. If so can you say me the steps"]))
