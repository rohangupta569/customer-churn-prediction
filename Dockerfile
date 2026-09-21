FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
COPY app.py ./
RUN pip install --no-cache-dir -e '.[app]'
RUN churn generate --rows 3000 --output data/customer_churn.csv && \
    churn train --data data/customer_churn.csv --output-dir artifacts && \
    rm -rf data

EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0"]
