from typing import Any, Dict

import matplotlib.pyplot as plt
import streamlit as st
import toml
from matplotlib.backends.backend_agg import RendererAgg
from sentiment_model import sentiment_model
from summarization_model import (abstractive_summarization, count_alpha_tokens,
                                 wordcloud_gen)


@st.cache(allow_output_mutation=True, ttl=300)
def load_config(config_file: str) -> Dict[Any, Any]:
    config = toml.load(config_file)
    return dict(config)


config = load_config("config/app_config.toml")

_lock = RendererAgg.lock


def sidebar():
    st.sidebar.image("assets/popcorn.png", use_column_width=True)
    st.sidebar.text("")
    st.sidebar.markdown(
        "<h5 style='text-align: center; font-weight: 300; "
        "color: #8be9fd;'><i>pour l'amour du cinéma</i></h5>",
        unsafe_allow_html=True)

    st.sidebar.text("")
    st.sidebar.write("")
    st.sidebar.markdown(
        "<h5 style='text-align: center;'>"
        "<a href='https://github.com/akshaydevml/lazy-film-reviews'>"
        "<img src='https://img.shields.io/badge/view%20in%20"
        "github-%23121011.svg?style=for-the-badge&logo=gitshub&"
        "logoColor=white'></a></h5>",
        unsafe_allow_html=True)

    st.sidebar.write("")
    st.sidebar.markdown(
        "<h5 style='text-align: center; color: gray;'>"
        "Contributors</h5>",
        unsafe_allow_html=True)

    st.sidebar.markdown(
        "<h5 style='text-align: center;'>"
        "<a href='https://www.linkedin.com/in/akshay-dev-karama'>"
        "<img src='https://img.shields.io/badge/akshay--dev--karama-0077B5?"
        "style=social&logo=linkedin&logoColor=blue' class='img-fluid'"
        " height=17></a>&emsp;<a href='https://www.linkedin.com/in/"
        "sreevishnu-damodaran'><img src='https://img.shields.io/badge/"
        "sreevishnu--damodaran-0077B5?style=social&logo=linkedin&"
        "logoColor=blue' class='img-fluid' height=17></a></h5>",
        unsafe_allow_html=True)


def run_models(text):
    if len(text.split(" ")) < 150:
        st.error('Your input needs to be at least 150 words long')
    else:
        with st.spinner('Running Sentiment Model...'):
            # Sentiment analysis task
            st.session_state.sentiment = sentiment_model(text)

        with st.spinner('Running Summarization Model...'):
            # Summarization task
            (st.session_state.summary, st.session_state.rouge_precision,
             st.session_state.rouge_f1) = abstractive_summarization(text)

        with st.spinner('Generating Word Cloud...'):
            # Wordcloud generation task
            st.session_state.wordcloud = wordcloud_gen(text)

        with st.spinner('Getting Word Counts...'):
            (st.session_state.input_count,
             st.session_state.summary_count) = count_alpha_tokens(
                 text, st.session_state.summary)


def body():
    st.title('Lazy Film Reviews')
    st.caption(config["intro"]["caption"])

    with st.expander("What is this app?", expanded=False):
        st.markdown(config["intro"]["description"])
        st.write("")
    st.write("")

    r1_c1, _, r1_c2 = st.columns([3, 0.1, 2.2])
    r2_c1, r2_c2, r2_c3, r2_c4 = st.columns(4)

    text = r1_c1.text_area(label="Enter any film review of choice below:",
                           value=config["defaults"]["review"],
                           max_chars=4000,
                           height=700,
                           help="Press 'Generate' after entering a review")
    r1_c1.write("")

    st.empty()

    if 'summary' not in st.session_state:
        st.session_state.sentiment = "positive"
        st.session_state.summary = config['defaults']['summary']
        st.session_state.rouge_precision = config['defaults'][
            'rouge_precision']
        st.session_state.rouge_f1 = config['defaults']['rouge_f1']
        st.session_state.wordcloud = wordcloud_gen(
            config["defaults"]["review"])
        st.session_state.input_count = config['defaults']['input_count']
        st.session_state.summary_count = config['defaults']['summary_count']

    r1_c1.button(label="Generate", on_click=run_models, args=(text, ))

    if str(st.session_state.sentiment) == "positive":
        emoji = " 👍🏻"
    else:
        emoji = " 👎🏻"

    r1_c2.metric(label="Overall Sentiment 🧐",
                 value=str(st.session_state.sentiment).capitalize() + emoji)
    r1_c2.write("")

    r1_c2.markdown("##### Text Summarization Results")
    r1_c2.text_area(label="Here's your summary",
                    value=st.session_state.summary,
                    height=200,
                    disabled=True)
    r1_c2.write("")

    r1_c2.markdown('##### Descriptive Terms')

    with _lock:
        fig = plt.figure(figsize=(4, 4),
                         facecolor="#3a4064",
                         edgecolor="#3a4064")
        plt.axis("off")
        plt.tight_layout(pad=0)
        plt.imshow(st.session_state.wordcloud, interpolation='bilinear')
        r1_c2.pyplot(fig, clear_figure=True)

    r1_c2.write("")

    r2_c1.metric("Rouge F1 Score", st.session_state.rouge_f1)
    r2_c2.metric("Rouge Precision Score", st.session_state.rouge_precision)
    r2_c3.metric("Input Word Count", st.session_state.input_count)
    r2_c4.metric("Summary Word Count", st.session_state.summary_count)


def main():
    sidebar()
    body()


if __name__ == '__main__':
    main()
