FROM tensorflow/tensorflow:2.15.0-gpu

WORKDIR /app

# Install system dependencies if needed
RUN apt-get update && apt-get install -y build-essential git

# Install torch with CUDA support (compatible with CUDA 12.x from TF image)
RUN pip install torch --index-url https://download.pytorch.org/whl/cu121

# Install python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy training code and data
COPY src/ src/
COPY weather_data.csv .

# Set entrypoint (optional, can be overridden by pipeline)
# ENTRYPOINT ["python", "train_model.py"]
