# 1 
FROM python:3.7

# 2
COPY src/ /app
WORKDIR /app

# 3
RUN pip install Werkzeug==0.16.1
RUN pip install Flask-Gunicorn
RUN pip install flask-restplus
RUN pip install statsmodels
RUN pip install scikit-learn==0.21
RUN pip install nltk
RUN pip install gensim
# 4
ENV PORT 8080

# 5
CMD exec gunicorn --bind :$PORT --workers 1 --threads 8 --timeout 1000 --log-level=DEBUG app:app


