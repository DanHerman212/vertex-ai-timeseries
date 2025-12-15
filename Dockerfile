FROM python:3.11-slim

WORKDIR /app

# Install system dependencies if needed
RUN apt-get update && apt-get install -y build-essential

# Install python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy training code
COPY src/ .

# Set entrypoint (optional, can be overridden by pipeline)
# ENTRYPOINT ["python", "train_model.py"]
