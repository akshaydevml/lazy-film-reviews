# Lazy Film Reviews 

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![python lint](https://img.shields.io/badge/PyLint-passing-brightgreen)](.github/workflows/pylint.yml)
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io/akshaydevml/lazy-film-reviews/main/app/streamlit_app.py)

Lazy Film Reviews is a Streamlit app for summarizing and sentiment scoring a review, the idea is to input a film review and get a concise summary and a sentiment indicating whether the review was positive or negative. We also display a WordCloud containing descriptive terms used in the review and the ROUGE metric to gauge the effectiveness of the summary.

We trained a custom sci-kit learn LinearSVC model on the classic IMDB dataset for sentiment scoring. More advanced DL models like BERT/Bi-Directional LSTM with attention were tried, but they failed to yield satisficing results. 

For summarization, a pretrained hugging face model, 'distilbart-cnn-6-6' (by @sshleifer) is used. The model was trained on the CNN/DailyMail dataset. 

<br>

## License

This project is open-source and licensed under the MIT license.
