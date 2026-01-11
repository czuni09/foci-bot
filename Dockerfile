FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY streamlit_app.py .
COPY start.sh .
RUN chmod +x start.sh

EXPOSE 8501

CMD ["bash", "start.sh"]
