import json
import pickle
import random
import nltk
import numpy as np

from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Download tokenizer
nltk.download('punkt')
nltk.download('punkt_tab')

stemmer = PorterStemmer()

# Load intents file
with open('intents.json', 'r') as file:
    data = json.load(file)

patterns = []
tags = []

# Process training data
for intent in data['intents']:
    for pattern in intent['patterns']:
        patterns.append(pattern)
        tags.append(intent['tag'])

# Text preprocessing
def preprocess(text):
    tokens = nltk.word_tokenize(text.lower())
    words = [stemmer.stem(word) for word in tokens]
    return " ".join(words)

processed_patterns = [preprocess(pattern) for pattern in patterns]

# Convert text into vectors
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(processed_patterns)

# Train model
model = LogisticRegression()
model.fit(X, tags)

# Save model
pickle.dump(model, open('chatbot_model.pkl', 'wb'))
pickle.dump(vectorizer, open('vectorizer.pkl', 'wb'))

print("Model trained successfully!")