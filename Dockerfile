FROM tensorflow/tensorflow:2.15.0-gpu

WORKDIR /app

# Install system dependencies if needed
RUN apt-get update && apt-get install -y build-essential git

# Install uv for faster dependency resolution
RUN pip install uv

# Install torch with CUDA support (compatible with CUDA 12.x from TF image)
RUN uv pip install --system torch --index-url https://download.pytorch.org/whl/cu121

# Install python dependencies
COPY requirements.txt .
RUN uv pip install --system --no-cache -r requirements.txt

# Copy training code and data
COPY src/ src/
COPY weather_data.csv .

# Set entrypoint (optional, can be overridden by pipeline)
# ENTRYPOINT ["python", "train_model.py"]
