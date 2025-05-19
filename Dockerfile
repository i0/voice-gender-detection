FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libsndfile1 \
    ffmpeg \
    libasound2-dev \
    libportaudio2 \
    libportaudiocpp0 \
    portaudio19-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker cache
COPY requirements.txt .
# Ensure NumPy < 2.0 is installed first to prevent compatibility issues
RUN pip install --no-cache-dir numpy==1.24.3
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir pydub

# Copy the application code
COPY . .

# Create cache directory 
RUN mkdir -p /app/cache
RUN mkdir -p /app/static

# Create a sample audio file for testing
RUN echo "Creating sample test audio file"
RUN ffmpeg -f lavfi -i "sine=frequency=1000:duration=5" -ar 16000 -ac 1 /app/cache/test.wav

# Expose the API port
EXPOSE 8000

# Run the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]