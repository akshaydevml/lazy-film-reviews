# Lazy Film Reviews 

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![python lint](https://img.shields.io/badge/PyLint-passing-brightgreen)](.github/workflows/pylint.yml)
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io/akshaydevml/lazy-film-reviews/main/app/streamlit_app.py)

Lazy Film Reviews is a Streamlit app for summarizing and sentiment scoring movie reviews. Input any film review of choice and click 'Generate' to get a concise summary and an overall sentiment indicating whether the review was positive or negative. We also display a Wordcloud containing descriptive terms in the input and the ROUGE metric to gauge the effectiveness of the summary.

We trained a custom scikit-learn LinearSVC model on the classic IMDB dataset for sentiment scoring. More advanced DL models like BERT/Bi-Directional LSTM with attention were tried, but they failed to yield satisficing results.

For text summarization, we use a pretrained Hugging Face model, 'distilbart-cnn-6-6' (by [sshleifer](https://github.com/sshleifer)), trained on the CNN/DailyMail dataset.

<br>

## License

This project is open-source and licensed under the MIT license.
