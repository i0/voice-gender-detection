# Gender Recognition API

This project provides a service for recognizing gender from audio files using a pre-trained model based on wav2vec2. It includes both an API with interactive documentation and a web-based UI.

## Features

- 🎙️ Gender prediction from audio files (.wav, .mp3, .ogg)
- 🌐 FastAPI backend with automatic API documentation
- 👥 Simple web interface for easy usage
- 🐳 Docker support for easy deployment
- ⏱️ Detailed timing metrics for performance analysis
- 📊 Detailed logging with emojis

## Ethical Considerations

**Important Disclaimer**: This tool is provided for educational and research purposes only.

- **Bias Awareness**: The underlying model was trained on the LibriSpeech dataset, which may not represent the full diversity of human voices and may contain inherent biases.
- **Binary Classification**: This model only classifies audio as "male" or "female" based on acoustic patterns learned from the training data. It does not account for non-binary gender identities.
- **Consent**: Always obtain consent before analyzing someone's voice data.
- **Privacy**: Process audio data responsibly and in accordance with relevant privacy laws and regulations.
- **Responsible Use**: Do not use this tool for discrimination, surveillance, or any applications that may infringe on human rights.

Gender prediction systems should be approached as probabilistic tools rather than definitive classifiers of a person's gender identity.

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

### Running Locally (Faster Performance)

Running the application locally typically provides better performance than Docker, especially for inference.

1. Clone this repository
2. Create a virtual environment (recommended):

```bash
# Create virtual environment
python -m venv venv

# Activate on Windows
venv\Scripts\activate

# Activate on macOS/Linux
source venv/bin/activate
```

3. Install dependencies:

```bash
# Install numpy first to avoid compatibility issues
pip install numpy==1.24.3
pip install -r requirements.txt
```

4. Run the API server:

```bash
python main.py
```

5. Access the services:
   - API and Documentation: http://localhost:8000/docs
   - Web Interface: http://localhost:8000/ui

## API Usage

### API Endpoints

- `GET /`: Home page with links to documentation and UI
- `GET /ui`: Web-based user interface
- `GET /docs`: Interactive API documentation
- `POST /predict`: Predict gender from an audio file

### Example Requests

#### Using curl

```bash
# Predict gender using a WAV file
curl -X POST "http://localhost:8000/predict" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@/path/to/audio.wav"

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
- `main.py`: FastAPI server implementation (formerly api.py)
- `Dockerfile` & `docker-compose.yml`: Docker configuration
- `static/`: Directory for static assets
- `cache/`: Directory for cached model files

## Credits

- Model: [alefiury/wav2vec2-large-xlsr-53-gender-recognition-librispeech](https://huggingface.co/alefiury/wav2vec2-large-xlsr-53-gender-recognition-librispeech) by Alexander Leandro Figueiredo
- Base architecture: [facebook/wav2vec2-large-xlsr-53](https://huggingface.co/facebook/wav2vec2-large-xlsr-53) by Facebook AI Research

## License

This project is licensed under the MIT License - see the LICENSE file for details.

The underlying model [alefiury/wav2vec2-large-xlsr-53-gender-recognition-librispeech](https://huggingface.co/alefiury/wav2vec2-large-xlsr-53-gender-recognition-librispeech) is subject to its own license terms as specified by its creators.