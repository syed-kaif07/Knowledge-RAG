#base image
FROM python:3.12-slim

# Prevent Python from writing .pyc files and enable immediate stdout/stderr logging
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

#working dir
WORKDIR /app

# Install system dependencies (needed for compiling C-extensions or packages like chromadb/sqlite)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Step 1: Copy and install dependencies first (leverages Docker layer cache)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Step 2: Copy the rest of the application code
COPY . .

#port
EXPOSE 8501

# Streamlit healthcheck (optional, recommended for production/containers)
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1
# Start the Streamlit application
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
