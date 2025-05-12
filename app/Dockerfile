FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
#COPY .streamlit /app/.streamlit
ENV HOME=/app

EXPOSE 8502

#CMD ["streamlit", "run", "/app/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
#CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0", "--config", "/app/.streamlit/config.toml"]

CMD ["streamlit", "run", "app.py", "--server.port=8502", "--server.address=0.0.0.0"]
