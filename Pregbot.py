# -*- coding: utf-8 -*-
"""
Created on Fri Jul  7 10:34:43 2023

@author: USER
"""

import nltk
from nltk.stem import WordNetLemmatizer
import pickle
import numpy as np
from keras.models import load_model
import json
import random
from flask import Flask, render_template, request

app = Flask(__name__)
app.static_folder = 'static'

# Load and preprocess data
nltk.download('punkt')
nltk.download('wordnet')
lemmatizer = WordNetLemmatizer()
model = load_model('model.h5')
intents = json.loads(open('pregdata.json').read())
words = pickle.load(open('texts.pkl', 'rb'))
classes = pickle.load(open('labels.pkl', 'rb'))

def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

def bow(sentence, words, show_details=True):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                bag[i] = 1
                if show_details:
                    print("found in bag: %s" % w)
    return np.array(bag)

def predict_class(sentence, model):
    p = bow(sentence, words, show_details=False)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list

def get_response(intents_json, tag):
    list_of_intents = intents_json['intents']
    for intent in list_of_intents:
        if intent['tag'] == tag:
            return random.choice(intent['responses'])
    return "I'm sorry, I don't have the information you're looking for."

def chatbot_response(msg):
    intents_json = intents
    intents_response = predict_class(msg, model)
    tag = intents_response[0]['intent']
    return get_response(intents_json, tag)

# Route for home page
@app.route("/")
def home():
    return render_template("index.html", chatbot_name="PregBot")

# Route for getting chatbot response
@app.route("/get")
def get_bot_response():
    user_text = request.args.get('msg')
    response = chatbot_response(user_text)
    return response

# Run the app
if __name__ == "__main__":
    app.run()
