import spacy
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
    abstract_summary = abstractive_summarizer(sample,
                                              min_length=100,
                                              max_length=120,
                                              num_return_sequences=4)
    summary = abstract_summary[0].get('summary_text')
    summary = summary_cleaner(summary)
    rouge = Rouge()
    rouge_score = rouge.get_scores(summary, sample)
    rouge_precision = rouge_score[0].get('rouge-2').get('p')
    rouge_precision = round(rouge_precision, 2)
    rouge_f1 = rouge_score[0].get('rouge-2').get('f')
    rouge_f1 = round(rouge_f1, 2)

    return summary, rouge_precision, rouge_f1


def wordcloud_gen(sample, model='en_core_web_md'):
    nlp = spacy.load(model)
    text = nlp(sample)
    descriptive_list = []

    for token in text:
        if token.pos_ == 'ADJ' or 'ADV':
            descriptive_list.append(str(token))

    sample = " ".join(descriptive_list)
    stopwords = set(STOPWORDS)
    wordcloud = WordCloud(width=600,
                          height=400,
                          background_color="#3a4064",
                          colormap="Pastel1",
                          stopwords=stopwords,
                          min_font_size=8).generate(sample)

    return wordcloud
