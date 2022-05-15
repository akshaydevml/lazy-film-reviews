import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def load_sentiment_model():
    tokenizer = AutoTokenizer.from_pretrained(
        "Sreevishnu/funnel-transformer-small-imdb", use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        "Sreevishnu/funnel-transformer-small-imdb",
        num_labels=2,
        max_position_embeddings=1024)

    return model, tokenizer


model, tokenizer = load_sentiment_model()


def get_prediction(review, tokenizer, model):
    inputs = tokenizer(review, return_tensors='pt')
    with torch.no_grad():
        predictions = model(**inputs)[0].numpy()
    top_prediction = predictions.argmax().item()
    return "positive" if top_prediction == 1 else "negative"


def sentiment_model(sample):
    prediction = get_prediction(sample, tokenizer, model)

    return prediction
