FROM python:3.11-slim

WORKDIR /app

# Install system build dependencies (e.g. for scipy if it falls back to source)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gfortran \
    pkg-config \
    libopenblas-dev \
    && rm -rf /var/lib/apt/lists/*

# Install dependencies
COPY requirements.txt requirements-api.txt ./
RUN pip install --no-cache-dir --upgrade pip setuptools wheel
RUN pip install --no-cache-dir -r requirements.txt -r requirements-api.txt

# Copy project
COPY . .

# Expose ports
EXPOSE 8000 8501

# Default: run Streamlit (override with CMD for API)
CMD ["streamlit", "run", "app/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
