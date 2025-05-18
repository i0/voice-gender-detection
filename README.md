# Gender Recognition API

This project provides a service for recognizing gender from audio files using a pre-trained model based on wav2vec2. It includes both an API with interactive documentation and a web-based UI.

## Features

- 🎙️ Gender prediction from audio files (.wav, .mp3, .ogg)
- 🌐 FastAPI backend with automatic API documentation
- 👥 Simple web interface for easy usage
- 🐳 Docker support for easy deployment
- 📊 Detailed logging with emojis

## Getting Started

### Prerequisites

- Docker and Docker Compose (recommended)
- Python 3.9+ (for local development)

### Running with Docker (Recommended)

1. Clone this repository
2. Start the service:

```bash
docker-compose up
```

3. Access the services:
   - API and Documentation: http://localhost:8000/docs
   - Web Interface: http://localhost:8000/ui

### Running Locally

1. Clone this repository
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run the API server:

```bash
python api.py
```

4. Access the services:
   - API and Documentation: http://localhost:8000/docs
   - Web Interface: http://localhost:8000/ui

## API Usage

### API Endpoints

- `GET /`: Home page with links to documentation and UI
- `GET /ui`: Web-based user interface
- `GET /docs`: Interactive API documentation
- `POST /predict`: Predict gender from an audio file
- `GET /health`: Health check endpoint

### Example Requests

#### Using curl

```bash
# Predict gender using a WAV file
curl -X POST "http://localhost:8000/predict" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@/path/to/audio.wav"

# Check API health
curl -X GET "http://localhost:8000/health"
```

#### Using Python

```python
import requests

url = "http://localhost:8000/predict"
files = {"file": open("sample.wav", "rb")}

response = requests.post(url, files=files)
result = response.json()
print(result)  # e.g. {'female': 0.998633, 'male': 0.001367}
```

### Example Response

```json
{
  "female": 0.998633086681366,
  "male": 0.0013668379979208112
}
```

## Model Information

This project uses the [alefiury/wav2vec2-large-xlsr-53-gender-recognition-librispeech](https://huggingface.co/alefiury/wav2vec2-large-xlsr-53-gender-recognition-librispeech) model from Hugging Face for gender recognition. This model is a fine-tuned version of Facebook's wav2vec2-xls-r-300m specifically trained for gender recognition.

### Model Performance

- Achieves an F1 score of 0.9993 and a loss of 0.0061 on the Librispeech-clean-100 dataset
- Trained on the Librispeech-clean-100 dataset with a 70/10/20 train/validation/test split
- Handles audio recordings up to 5 seconds long
- Processes 16kHz mono audio files

## Development

The code structure includes:

- `gender_predictor.py`: Core gender prediction functionality
- `api.py`: FastAPI server implementation
- `Dockerfile` & `docker-compose.yml`: Docker configuration
- `static/`: Directory for static assets

## Credits

- Model: [alefiury/wav2vec2-large-xlsr-53-gender-recognition-librispeech](https://huggingface.co/alefiury/wav2vec2-large-xlsr-53-gender-recognition-librispeech) by Alexander Leandro Figueiredo
- Base architecture: [facebook/wav2vec2-large-xlsr-53](https://huggingface.co/facebook/wav2vec2-large-xlsr-53) by Facebook AI Research

## License

This project is provided as-is without any warranty. Use at your own risk.
