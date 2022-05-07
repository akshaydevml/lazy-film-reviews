import spacy
from cleantext import clean
from rouge import Rouge
from spacy.attrs import IS_ALPHA
from transformers import pipeline
from wordcloud import STOPWORDS, WordCloud

model = 'en_core_web_md'
nlp = spacy.load(model, disable=['parser', 'ner'])

abstractive_summarizer = pipeline('summarization',
                                  model="sshleifer/distilbart-cnn-6-6")


def summary_cleaner(summary):
    summary = clean(summary, lower=False)
    summary = summary.split('. ')
    if summary[-1][-1] != '.':
        summary.pop(-1)
        summary = ". ".join(summary) + '.'
        return summary
    else:
        return ". ".join(summary)


def abstractive_summarization(sample):
    abstract_summary = abstractive_summarizer(
        sample,
        min_length=100,
        max_length=120,
    )
    summary = abstract_summary[0].get('summary_text')
    summary = summary_cleaner(summary)
    rouge = Rouge()
    rouge_score = rouge.get_scores(summary, sample)
    rouge_precision = rouge_score[0].get('rouge-2').get('p')
    rouge_precision = round(rouge_precision, 2)
    rouge_f1 = rouge_score[0].get('rouge-2').get('f')
    rouge_f1 = round(rouge_f1, 2)

    return summary, rouge_precision, rouge_f1


def count_alpha_tokens(sample, summary):
    sample = clean(sample)
    sample = nlp(sample)
    input_count = sample.count_by(IS_ALPHA)[1]

    summary = clean(summary)
    summary = nlp(summary)
    summary_count = summary.count_by(IS_ALPHA)[1]

    return input_count, summary_count


def wordcloud_gen(sample):
    sample = clean(sample, lower=False)
    sample = nlp(sample)
    descriptive_list = []

    for token in sample:
        if token.pos_ == 'ADJ' or token.pos_ == 'ADV':
            descriptive_list.append(str(token))

    sample = " ".join(descriptive_list)
    stopwords = set(STOPWORDS)
    wordcloud = WordCloud(width=540,
                          height=360,
                          background_color="#3a4064",
                          colormap="Pastel1",
                          stopwords=stopwords,
                          min_font_size=9).generate(sample)

    return wordcloud
