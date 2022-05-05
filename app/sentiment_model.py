import os
import urllib

import joblib
from tqdm import tqdm


def download_weights(url, out_path):
    response = getattr(urllib, 'request', urllib).urlopen(url)
    filename = url.split('/')[-1]
    with tqdm.wrapattr(open(os.path.join(out_path, filename), "wb"),
                       "write",
                       miniters=1,
                       desc=filename,
                       position=0,
                       leave=True,
                       total=getattr(response, 'length', None)) as fout:
        for chunk in response:
            fout.write(chunk)


def sentiment_model(sample):
    weights_file = "movie_sentiment.joblib"
    url = "https://api.wandb.ai/files/akshaydevwandb/uncategorized/"\
        "2hldjwkn/movie_sentiment.joblib"

    if not os.path.exists(weights_file):
        download_weights(url, ".")

    sentiment_analysis = joblib.load(open(weights_file, 'rb'))
    prediction = sentiment_analysis.predict([sample])
    prediction = str(prediction[0])

    return prediction
