# app.py

import streamlit as st
import speech_recognition as sr
import nltk
import random
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Télécharger les ressources nécessaires
nltk.download('punkt')

# ------------------------------
# DATA chatbot
intents = {
    "intents": [
        {"tag": "greeting", "patterns": ["Hi", "Hello", "Hey"], "responses": ["Hello!", "Hi there! How can I help you?"]},
        {"tag": "goodbye", "patterns": ["Bye", "See you later", "Goodbye"], "responses": ["Goodbye!", "See you soon!"]},
        {"tag": "thanks", "patterns": ["Thanks", "Thank you"], "responses": ["You're welcome!", "No problem!"]},
        {"tag": "name", "patterns": ["What's your name?", "Who are you?"], "responses": ["I'm your AI Assistant.", "You can call me Chatbot."]},
    ]
}

# ------------------------------
# Préparation des données

training_sentences = []
training_labels = []
labels = []
responses = []

for intent in intents['intents']:
    for pattern in intent['patterns']:
        training_sentences.append(pattern)
        training_labels.append(intent['tag'])
    responses.append(intent['responses'])
    labels.append(intent['tag'])

lbl_encoder = LabelEncoder()
lbl_encoder.fit(training_labels)
training_labels = lbl_encoder.transform(training_labels)

vocab_size = 1000
embedding_dim = 16
max_len = 20
oov_token = "<OOV>"

tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_token)
tokenizer.fit_on_texts(training_sentences)
sequences = tokenizer.texts_to_sequences(training_sentences)
padded_sequences = pad_sequences(sequences, truncating='post', maxlen=max_len)

# ------------------------------
# Modèle Deep Learning

model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_len),
    tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(len(labels), activation='softmax')
])

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(padded_sequences, np.array(training_labels), epochs=300, verbose=0)

# ------------------------------
# Fonctions principales

def chatbot_response(text):
    input_seq = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(input_seq, truncating='post', maxlen=max_len)
    prediction = model.predict(padded, verbose=0)
    tag = lbl_encoder.inverse_transform([np.argmax(prediction)])
    for intent in intents['intents']:
        if intent['tag'] == tag[0]:
            return random.choice(intent['responses'])

def transcribe_audio_file(uploaded_file):
    recognizer = sr.Recognizer()
    with sr.AudioFile(uploaded_file) as source:
        audio = recognizer.record(source)
    try:
        text = recognizer.recognize_google(audio)
        return text
    except Exception as e:
        return f"Erreur : {str(e)}"

# ------------------------------
# Interface Streamlit

st.title(" Speech-Enabled ChatBot ")

input_mode = st.radio("Choisissez votre mode d'entrée :", ("Texte", "Voix (Upload fichier audio)"))

if input_mode == "Texte":
    user_input = st.text_input("Tapez votre message ici :")
    if st.button("Envoyer"):
        if user_input:
            response = chatbot_response(user_input)
            st.text_area("Réponse du chatbot :", value=response, height=150)
elif input_mode == "Voix (Upload fichier audio)":
    uploaded_file = st.file_uploader("Uploadez votre fichier audio (.wav ou .flac)", type=["wav", "flac"])
    if uploaded_file is not None:
        st.audio(uploaded_file, format="audio/wav")
        transcript = transcribe_audio_file(uploaded_file)
        st.success(f"Texte reconnu : {transcript}")
        if transcript:
            response = chatbot_response(transcript)
            st.text_area("Réponse du chatbot :", value=response, height=150)
