# 1 
FROM python:3.7

# 2
RUN pip install -r requirements.txt

# 3
COPY src/ /app
WORKDIR /app

# 4
ENV PORT 8080

# 5
ENTRYPOINT ["python"]
CMD ["main.py"]
