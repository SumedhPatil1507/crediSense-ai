FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt requirements-api.txt ./
RUN pip install --no-cache-dir -r requirements.txt -r requirements-api.txt

# Copy project
COPY . .

# Expose ports
EXPOSE 8000 8501

# Default: run Streamlit (override with CMD for API)
CMD ["streamlit", "run", "app/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
