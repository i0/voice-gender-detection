import os
import json
import logging
import torch
import uvicorn
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import List, Dict, Union
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydub import AudioSegment
from gender_predictor import get_gender_and_score

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger("gender-api")

# Initialize FastAPI
app = FastAPI(
    title="Gender Recognition API",
    description="API for gender recognition based on audio files. This tool is provided for educational and research purposes only.",
    version="1.0.0",
    license_info={
        "name": "MIT License",
        "url": "https://opensource.org/licenses/MIT",
    },
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Model configuration
MODEL_PATH = "alefiury/wav2vec2-large-xlsr-53-gender-recognition-librispeech"
LABEL2ID = {"female": 0, "male": 1}
ID2LABEL = {0: "female", 1: "male"}
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Create static directory if it doesn't exist
STATIC_DIR = Path("static")
STATIC_DIR.mkdir(exist_ok=True)

# Mount static files directory
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

@app.get("/", response_class=HTMLResponse)
async def root():
    """Root endpoint that redirects to the documentation"""
    return """
    <html>
        <head>
            <title>Gender Recognition API</title>
            <style>
                body { font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }
                h1 { color: #333; }
                .links { margin-top: 20px; }
                .links a { display: block; margin-bottom: 10px; }
                .disclaimer { margin-top: 20px; padding: 15px; border: 1px solid #f8d7da; background-color: #f8d7da; color: #721c24; border-radius: 5px; }
            </style>
        </head>
        <body>
            <h1>Gender Recognition API</h1>
            <p>Welcome to the Gender Recognition API. This service predicts gender based on audio files.</p>
            
            <div class="links">
                <a href="/docs">📚 API Documentation</a>
                <a href="/ui">🖥️ Web Interface</a>
            </div>
        </body>
    </html>
    """

@app.get("/ui", response_class=HTMLResponse)
async def ui():
    """User interface for gender prediction"""
    return """
    <html>
        <head>
            <title>Gender Recognition UI</title>
            <style>
                body { font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }
                h1 { color: #333; }
                .form-container { margin: 20px 0; padding: 20px; border: 1px solid #ddd; border-radius: 5px; }
                button { background-color: #4CAF50; color: white; padding: 10px 15px; border: none; cursor: pointer; margin-top: 10px; }
                #result { margin-top: 20px; padding: 15px; border: 1px solid #ddd; border-radius: 5px; display: none; }
                .loading { display: none; margin-top: 20px; }
                .female { color: #FF69B4; }
                .male { color: #0000FF; }
                .disclaimer { margin-top: 20px; padding: 15px; border: 1px solid #f8d7da; background-color: #f8d7da; color: #721c24; border-radius: 5px; }
            </style>
        </head>
        <body>
            <h1>Gender Recognition UI</h1>
            
            <div class="form-container">
                <h2>Upload Audio</h2>
                <p>Upload an audio file (.wav, .mp3, .ogg) to predict gender:</p>
                <input type="file" id="audioFile" accept=".wav, .mp3, .ogg">
                <button onclick="predictGender()">Predict Gender</button>
                <div class="loading" id="loading">Processing... This may take a few seconds</div>
            </div>
            <div id="result"></div>
            
            <script>
                async function predictGender() {
                    const fileInput = document.getElementById('audioFile');
                    const loading = document.getElementById('loading');
                    const result = document.getElementById('result');
                    
                    if (!fileInput.files.length) {
                        alert('Please select an audio file');
                        return;
                    }
                    
                    const file = fileInput.files[0];
                    const formData = new FormData();
                    formData.append('file', file);
                    
                    loading.style.display = 'block';
                    result.style.display = 'none';
                    
                    try {
                        const response = await fetch('/predict', {
                            method: 'POST',
                            body: formData
                        });
                        
                        if (!response.ok) {
                            throw new Error('Server error');
                        }
                        
                        const data = await response.json();
                        let resultHtml = '<h2>Prediction Result</h2>';
                        
                        const femaleScore = data.female * 100;
                        const maleScore = data.male * 100;
                        
                        resultHtml += `<p><span class="female">Female: ${femaleScore.toFixed(2)}%</span></p>`;
                        resultHtml += `<p><span class="male">Male: ${maleScore.toFixed(2)}%</span></p>`;
                        
                        const predictedGender = femaleScore > maleScore ? 'Female' : 'Male';
                        const confidence = Math.max(femaleScore, maleScore);
                        
                        resultHtml += `<p><strong>Predicted Gender:</strong> ${predictedGender} (${confidence.toFixed(2)}% confidence)</p>`;
                        
                        result.innerHTML = resultHtml;
                        result.style.display = 'block';
                    } catch (error) {
                        result.innerHTML = '<h2>Error</h2><p>An error occurred during prediction. Please try again.</p>';
                        result.style.display = 'block';
                        console.error(error);
                    } finally {
                        loading.style.display = 'none';
                    }
                }
            </script>
        </body>
    </html>
    """

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Predict gender from an audio file
    
    This endpoint analyzes an audio file and returns probabilities for male/female classification.
    Note that this is for educational purposes only and should be used responsibly.
    """
    # Validate audio format
    valid_formats = [".wav", ".mp3", ".ogg"]
    ext = os.path.splitext(file.filename)[1].lower()
    
    if ext not in valid_formats:
        raise HTTPException(status_code=400, detail="Unsupported file format. Please upload .wav, .mp3, or .ogg files.")
    
    # Save the uploaded file
    with NamedTemporaryFile(delete=False, suffix=ext) as temp_file:
        temp_file.write(await file.read())
        temp_path = temp_file.name
    
    try:
        # Convert to WAV if needed
        wav_path = temp_path
        if ext != ".wav":
            logger.info(f"🔄 Converting {ext} to WAV format")
            audio = AudioSegment.from_file(temp_path)
            wav_path = temp_path.replace(ext, ".wav")
            audio.export(wav_path, format="wav")
        
        # Process the audio file
        logger.info(f"🎵 Processing audio file: {file.filename}")
        result = get_gender_and_score(
            model_name_or_path=MODEL_PATH,
            audio_paths=[wav_path],
            label2id=LABEL2ID,
            id2label=ID2LABEL,
            device=DEVICE
        )
        
        # Log the prediction
        female_score = result["female"]
        male_score = result["male"]
        predicted_gender = "female" if female_score > male_score else "male"
        confidence = max(female_score, male_score) * 100
        
        if predicted_gender == "female":
            logger.info(f"👩 Prediction: Female (confidence: {confidence:.2f}%)")
        else:
            logger.info(f"👨 Prediction: Male (confidence: {confidence:.2f}%)")
        
        return result
    except Exception as e:
        logger.error(f"❌ Error processing audio: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Clean up the temporary files
        if os.path.exists(temp_path):
            os.unlink(temp_path)
        if ext != ".wav" and os.path.exists(wav_path):
            os.unlink(wav_path)

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "model": MODEL_PATH}

if __name__ == "__main__":
    logger.info(f"🚀 Starting Gender Recognition API on device: {DEVICE}")
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)