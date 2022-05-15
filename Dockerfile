FROM python:3.8.13

EXPOSE 8080

RUN pip install --upgrade pip

COPY requirements.txt .

RUN pip install --no-cache-dir -r  requirements.txt 

COPY .streamlit  ./.streamlit/
COPY app  ./app/
COPY assets  ./assets/
COPY config  ./config/

WORKDIR .

ENTRYPOINT ["streamlit", "run", "./app/streamlit_app.py", "--server.port=8080", "--server.address=0.0.0.0"]