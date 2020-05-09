# 1 
FROM python:3.7

# 2
RUN pip install Werkzeug==0.16.1
RUN pip install Flask gunicorn
RUN pip install flask-restplus
RUN pip install statsmodels

# 3
COPY src/ /app
WORKDIR /app

# 4
ENV PORT 8080

# 5
CMD exec gunicorn --bind :$PORT --workers 1 --threads 8 app:app

