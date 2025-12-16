FROM us-docker.pkg.dev/vertex-ai/training/tf-gpu.2-17.py310:latest

WORKDIR /app

# Install system dependencies if needed
RUN apt-get update && apt-get install -y build-essential git

# Install uv for faster dependency resolution
RUN pip install uv

# Install python dependencies
COPY requirements.txt .
RUN uv pip install --system --no-cache -r requirements.txt

# Copy training code and data
COPY src/ src/
COPY weather_data.csv .

# Set entrypoint (optional, can be overridden by pipeline)
# ENTRYPOINT ["python", "train_model.py"]
