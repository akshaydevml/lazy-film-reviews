import spacy
import streamlit as st

from sentiment_model import sentiment_model
from summarization_model import abstractive_summarization, wordcloud_gen

nlp = spacy.load('en_core_web_md')

st.header('Summarization and Sentiment Analysis for Film Reviews')
st.subheader('Abstractive Summarization for Film Reviews')
text = st.text_area(label='', height=200)
if text:
    abstractive_summarization(text)
    sentiment_model(text)
    text = nlp(text)
    descriptive_list = []
    for token in text:
        if token.pos_ == 'ADJ' or 'ADV':
            descriptive_list.append(str(token))
    st.subheader('WordCloud of Descriptive Terms')
    wordcloud_gen(" ".join(descriptive_list))
