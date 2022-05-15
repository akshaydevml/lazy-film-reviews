# Lazy Film Reviews 

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![python lint](https://img.shields.io/badge/PyLint-passing-brightgreen)](.github/workflows/pylint.yml)
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://lazy-film-reviews-7gif2bz4sa-ew.a.run.app/)

Lazy Film Reviews is a Streamlit app for summarizing and sentiment scoring movie reviews. Input any film review of choice and click 'Generate' to get a concise summary and an overall sentiment indicating whether the review was positive or negative. We also display a Wordcloud containing descriptive terms in the input and the ROUGE metric to gauge the effectiveness of the summary.

We fine-tuned the `funnel-transformer-small` using Hugging Face to create a [custom model](https://huggingface.co/Sreevishnu/funnel-transformer-small-imdb) for the sentiment analysis task.

For text summarization, we use the pretrained `distilbart-cnn-6-6` (by [sshleifer](https://github.com/sshleifer)) from Hugging Face hub, trained on the CNN/DailyMail dataset.

<br>

## License

This project is open-source and licensed under the MIT license.
