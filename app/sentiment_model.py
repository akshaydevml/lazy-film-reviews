import os
import pickle

from app_utils import download_weights

weights_file = "movie_sentiment.pkl"
url = "https://github.com/akshaydevml/lazy-film-reviews/releases/download"\
        "/v0.0/movie_sentiment.pkl"

if not os.path.exists(weights_file):
    download_weights(url, ".")

try:
    sentiment_analysis = pickle.load(open(weights_file, 'rb'))
except Exception as e:
    print("Couldn't read model weights:", e)
    print("Downloading weights again.")
    download_weights(url, ".")
    sentiment_analysis = pickle.load(open(weights_file, 'rb'))


def sentiment_model(sample):
    prediction = sentiment_analysis.predict([sample])
    prediction = str(prediction[0])

    return prediction
