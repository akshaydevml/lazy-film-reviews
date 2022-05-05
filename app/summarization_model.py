import matplotlib.pyplot as plt
import streamlit as st
from rouge import Rouge
from transformers import pipeline
from wordcloud import STOPWORDS, WordCloud


def summary_cleaner(summary):
    summary = summary.split('. ')
    if summary[-1][-1] != '.':
        summary.pop(-1)
    summary = ". ".join(summary) + '.'
    return summary


def abstractive_summarization(sample):
    abstractive_summarizer = pipeline('summarization')
    abstract_summary = abstractive_summarizer(
        sample,
        min_length=100,
        max_length=120,
    )
    summary = abstract_summary[0].get('summary_text')
    summary = summary_cleaner(summary)
    rouge = Rouge()
    rouge_score = rouge.get_scores(summary, sample)
    rouge_score_precision = rouge_score[0].get('rouge-2').get('p')
    rouge_score_precision = round(rouge_score_precision, 2)
    rouge_score_f1 = rouge_score[0].get('rouge-2').get('f')
    rouge_score_f1 = round(rouge_score_f1, 2)
    st.markdown("#### Here's your Abstract Summary")
    st.write(summary)
    st.write(f"The Rouge Precision Score is: {rouge_score_precision}")
    st.write(f"The Rouge F1 score is: {rouge_score_f1}")
    st.write(f"Your text's length is: {len(sample)}")
    st.write(f"Summarized text length is: {len(summary)}")


def wordcloud_gen(sample):
    stopwords = set(STOPWORDS)
    wordcloud = WordCloud(width=800,
                          height=300,
                          background_color='white',
                          stopwords=stopwords,
                          min_font_size=8).generate(sample)
    fig = plt.figure(figsize=(4, 4), facecolor=None)
    plt.axis("off")
    plt.tight_layout(pad=0)
    plt.imshow(wordcloud, interpolation='bilinear')
    st.pyplot(fig)
