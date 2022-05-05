import spacy
import streamlit as st
from cleantext import clean

from sentiment_model import sentiment_model
from summarization_model import abstractive_summarization, wordcloud_gen

nlp = spacy.load('en_core_web_md', disable=['parser', 'ner'])

st.header('Summarization and Sentiment Analysis for Film Reviews')
st.subheader('Abstractive Summarization for Film Reviews')
text = st.text_area(label='', height=200)
if text:
    if len(text.split()) < 150:
        st.error('Your input needs to be at least 150 words long')
    else:
        abstractive_summarization(clean(text))
        sentiment_model(text)
        text = nlp(text)
        descriptive_list = []
        for token in text:
            if token.pos_ == 'ADJ' or 'ADV':
                descriptive_list.append(str(token))
        st.subheader('WordCloud of Descriptive Terms')
        text = clean(text)
        wordcloud_gen(" ".join(descriptive_list))
