import streamlit as st
import json
import pickle
import random
import nltk

from nltk.stem import PorterStemmer

# Load trained model and vectorizer
model = pickle.load(open('chatbot_model.pkl', 'rb'))
vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))

# Load intents
with open('intents.json', 'r') as file:
    data = json.load(file)

stemmer = PorterStemmer()

# Preprocessing function
def preprocess(text):
    tokens = nltk.word_tokenize(text.lower())
    words = [stemmer.stem(word) for word in tokens]
    return " ".join(words)

# Streamlit UI
st.set_page_config(page_title="AI Chatbot", page_icon="🤖")

st.title("🤖 AI Placement Preparation Chatbot")

st.write("Ask me questions about Python, DBMS, OOPs, and Machine Learning.")

user_input = st.text_input("You:")

if user_input:

    processed_input = preprocess(user_input)

    input_vector = vectorizer.transform([processed_input])

    prediction = model.predict(input_vector)[0]

    response = "Sorry, I don't understand."

    for intent in data['intents']:
        if intent['tag'] == prediction:
            response = random.choice(intent['responses'])

    st.success(response)