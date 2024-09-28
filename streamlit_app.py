import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.layers import Embedding, SimpleRNN, Dense
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
import streamlit as st
import re

# Streamlit UI
st.title("IMDB Movie Reviews Sentiment Analysis")
st.write("Enter a movie review to classify as positive or negative.")

# Load the pre-trained model
model = load_model("rnn.h5")

# Preprocess the review input
def conv(sample):
    # Get IMDb word index
    word_index = imdb.get_word_index()

    # Clean the review
    sample = sample.lower()
    sample = re.sub(r"[^a-zA-Z0-9\s]", "", sample)  # Remove special characters
    sample = sample.split()

    # Convert words to indices based on IMDb word index
    sample = [word_index.get(word, 2) for word in sample]  # 2 is the index for unknown words
    
    # Pad the sequence to a maximum length of 500
    sample = pad_sequences([sample], maxlen=500, padding='pre')
    return sample

# Function to predict sentiment
def pred_sentiment(review):
    processed_review = conv(review)
    pred = model.predict(processed_review)
    
    # Adjusted threshold for classification
    if pred[0][0] > 0.65:
        return 'Positive', pred[0][0]
    elif pred[0][0] < 0.45:
        return 'Negative', pred[0][0]
    else:
        return 'Neutral', pred[0][0]  # Neutral sentiment between 0.45 and 0.65

# Streamlit input handling
user_input = st.text_area('Movie Review (Limit 500 words)')
if len(user_input.split()) > 500:
    st.error("Your review exceeds 500 words. Please shorten it.")
elif st.button("Classify"):
    sentiment, score = pred_sentiment(user_input)
    st.write(f'Review: {user_input}')
    st.write(f'Sentiment: {sentiment}')
    st.write(f'Prediction Score: {score}')
else:
    st.write("Please write a review.")
