import os
import pickle

import streamlit as st
from app_utils import download_weights

weights_file = "movie_sentiment.pkl"
url = "https://github.com/akshaydevml/lazy-film-reviews/releases/download"\
        "/v0.0/movie_sentiment.pkl"

if not os.path.exists(weights_file):
    download_weights(url, ".")


@st.cache()
def load_sentiment_model():
    try:
        return pickle.load(open(weights_file, 'rb'))
    except Exception as e:
        print("Couldn't read model weights:", e)
        print("Downloading weights again.")
        download_weights(url, ".")
        return pickle.load(open(weights_file, 'rb'))


def sentiment_model(sample):
    sentiment_analysis = load_sentiment_model()
    prediction = sentiment_analysis.predict([sample])
    prediction = str(prediction[0])

    return prediction
