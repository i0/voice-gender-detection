import os
import json
import logging
import torch
import uvicorn
import asyncio
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import List, Dict, Union
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse
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

@app.on_event("startup")
async def startup_event():
    """Log startup information when the application starts"""
    port = 8000
    logger.info(f"🚀 Starting Gender Recognition API on device: {DEVICE}")
    logger.info(f"🌐 Server is running at: http://localhost:{port}")
    logger.info(f"✨ Web UI: 🎙️ http://localhost:{port}/ui")
    logger.info(f"📚 API Docs: 📋 http://localhost:{port}/docs")

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
                .spinner { display: inline-block; width: 20px; height: 20px; border: 3px solid rgba(0, 0, 0, 0.1); border-radius: 50%; border-top-color: #4CAF50; animation: spin 1s ease-in-out infinite; margin-right: 10px; vertical-align: middle; }
                @keyframes spin { to { transform: rotate(360deg); } }
                .progress-container { display: none; margin-top: 20px; }
                .progress-bar { width: 100%; height: 20px; background-color: #e0e0e0; border-radius: 5px; overflow: hidden; }
                .progress-fill { height: 100%; background-color: #4CAF50; width: 0%; transition: width 0.3s ease; }
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
                <div class="loading" id="loading"><div class="spinner"></div> Processing... This may take a few seconds</div>
                <div class="progress-container" id="progress-container">
                    <div class="progress-bar">
                        <div class="progress-fill" id="progress-fill"></div>
                    </div>
                    <div id="progress-text">0% - Starting...</div>
                </div>
            </div>
            <div id="result"></div>
            
            <script>
                async function predictGender() {
                    const fileInput = document.getElementById('audioFile');
                    const loading = document.getElementById('loading');
                    const result = document.getElementById('result');
                    const progressContainer = document.getElementById('progress-container');
                    const progressFill = document.getElementById('progress-fill');
                    const progressText = document.getElementById('progress-text');
                    
                    if (!fileInput.files.length) {
                        alert('Please select an audio file');
                        return;
                    }
                    
                    const file = fileInput.files[0];
                    const formData = new FormData();
                    formData.append('file', file);
                    
                    // Hide previous results and show progress indicators
                    loading.style.display = 'none';
                    progressContainer.style.display = 'block';
                    result.style.display = 'none';
                    progressFill.style.width = '0%';
                    progressText.textContent = '0% - Starting...';
                    
                    try {
                        // Use the predict endpoint
                        const response = await fetch('/predict', {
                            method: 'POST',
                            body: formData
                        });
                        
                        if (!response.ok) {
                            throw new Error('Server error');
                        }
                        
                        // Process the streaming response
                        const reader = response.body.getReader();
                        const decoder = new TextDecoder();
                        
                        while (true) {
                            const { value, done } = await reader.read();
                            
                            if (done) {
                                break;
                            }
                            
                            // Process each line in the streamed response
                            const text = decoder.decode(value);
                            const lines = text.split('\\n').filter(line => line.trim());
                            
                            for (const line of lines) {
                                try {
                                    const data = JSON.parse(line);
                                    
                                    if (data.status === 'processing') {
                                        // Update progress bar
                                        progressFill.style.width = `${data.progress}%`;
                                        progressText.textContent = `${data.progress}% - ${data.message}`;
                                    } 
                                    else if (data.status === 'complete') {
                                        // Process the final result
                                        let resultHtml = '<h2>Prediction Result</h2>';
                                        
                                        const femaleScore = data.result.female * 100;
                                        const maleScore = data.result.male * 100;
                                        
                                        resultHtml += `<p><span class="female">Female: ${femaleScore.toFixed(6)}%</span></p>`;
                                        resultHtml += `<p><span class="male">Male: ${maleScore.toFixed(6)}%</span></p>`;
                                        
                                        const predictedGender = data.prediction.gender === 'female' ? 'Female' : 'Male';
                                        const icon = data.prediction.icon;
                                        
                                        resultHtml += `<p><strong>Predicted Gender:</strong> ${icon} ${predictedGender} (${data.prediction.confidence.toFixed(6)}% confidence)</p>`;
                                        
                                        result.innerHTML = resultHtml;
                                        result.style.display = 'block';
                                        
                                        // Set progress to 100% completed
                                        progressFill.style.width = '100%';
                                        progressText.textContent = '100% - Completed!';
                                    }
                                    else if (data.status === 'error') {
                                        throw new Error(data.message);
                                    }
                                } catch (parseError) {
                                    console.error('Error parsing stream data:', parseError);
                                }
                            }
                        }
                    } catch (error) {
                        result.innerHTML = '<h2>Error</h2><p>An error occurred during prediction. Please try again.</p>';
                        result.style.display = 'block';
                        console.error(error);
                    }
                }
            </script>
        </body>
    </html>
    """

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Stream gender prediction progress from an audio file
    
    This endpoint analyzes an audio file and streams the progress and results.
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
    
    async def progress_generator():
        try:
            # Initial progress update
            yield json.dumps({"status": "processing", "progress": 0, "message": "Starting processing..."}) + "\n"
            await asyncio.sleep(0.1)  # Small delay to ensure frontend receives this message
            
            # Create a shared object for progress updates that's thread-safe
            progress_updates = {"current": None}
            
            def progress_callback(progress, message):
                # Just update the shared progress data - no asyncio calls from the worker thread
                progress_updates["current"] = {"progress": progress, "message": message}
                logger.info(f"Progress update: {progress}% - {message}")
                
            # Convert to WAV if needed
            wav_path = temp_path
            if ext != ".wav":
                yield json.dumps({"status": "processing", "progress": 5, "message": f"Converting {ext} to WAV format..."}) + "\n"
                audio = AudioSegment.from_file(temp_path)
                wav_path = temp_path.replace(ext, ".wav")
                audio.export(wav_path, format="wav")
                yield json.dumps({"status": "processing", "progress": 10, "message": "Audio conversion complete"}) + "\n"
            
            # Process the audio file
            logger.info(f"🎵 Processing audio file: {file.filename}")
            yield json.dumps({"status": "processing", "progress": 15, "message": "Checking model files..."}) + "\n"
            
            # The model download callback will provide updates from 0-10% progress
            # We want this shown as 15-25% in the UI, so we'll offset accordingly
            def model_download_progress_callback(progress, message):
                # Map model_progress (0-10%) to UI progress (15-25%)
                ui_progress = 15 + progress
                # Update the progress as usual
                progress_updates["current"] = {"progress": ui_progress, "message": message}
                logger.info(f"Download progress: {progress}% - {message}")
                
            # Don't add artificial progress here anymore, as the real download progress
            # will be reported through the callback
            
            # Start model processing in a separate task
            model_task = asyncio.create_task(run_model_processing(wav_path, model_download_progress_callback))
            
            # While the model is processing, poll for progress updates
            last_progress = None
            
            while not model_task.done():
                # Check if we have a new progress update
                current_progress = progress_updates["current"]
                
                if current_progress is not None and current_progress != last_progress:
                    # We have a new progress update, send it to the client
                    yield json.dumps({
                        "status": "processing", 
                        "progress": current_progress["progress"], 
                        "message": current_progress["message"]
                    }) + "\n"
                    
                    # Update last progress so we don't send duplicates
                    last_progress = current_progress
                
                # Wait a bit before checking again
                await asyncio.sleep(0.2)
            
            # Get the result from the completed model task
            result = await model_task
            
            # Send a 100% complete message
            yield json.dumps({"status": "processing", "progress": 99, "message": "Generating results..."}) + "\n"
            
            # After model processing is done, yield the final update
            # Log the prediction
            female_score = result["female"]
            male_score = result["male"]
            predicted_gender = "female" if female_score > male_score else "male"
            confidence = max(female_score, male_score) * 100
            
            if predicted_gender == "female":
                logger.info(f"👩 Prediction: Female (confidence: {confidence:.2f}%)")
                gender_icon = "👩"
            else:
                logger.info(f"👨 Prediction: Male (confidence: {confidence:.2f}%)")
                gender_icon = "👨"
                
            # Final result with predictions
            yield json.dumps({
                "status": "complete", 
                "result": result,
                "prediction": {
                    "gender": predicted_gender,
                    "confidence": confidence,
                    "icon": gender_icon
                }
            }) + "\n"
            
        except Exception as e:
            logger.error(f"❌ Error processing audio: {str(e)}")
            yield json.dumps({"status": "error", "message": str(e)}) + "\n"
        finally:
            # Clean up the temporary files
            if os.path.exists(temp_path):
                os.unlink(temp_path)
            if ext != ".wav" and os.path.exists(wav_path):
                os.unlink(wav_path)
    
    return StreamingResponse(progress_generator(), media_type="application/x-ndjson")
    

async def run_model_processing(wav_path, download_progress_callback):
    """Run model processing in an async-friendly way"""
    # Python 3.8 doesn't have asyncio.to_thread, so we use loop.run_in_executor instead
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        None,  # Use default executor
        lambda: get_gender_and_score(
            model_name_or_path=MODEL_PATH,
            audio_paths=[wav_path],
            label2id=LABEL2ID,
            id2label=ID2LABEL,
            device=DEVICE,
            progress_callback=download_progress_callback
        )
    )


if __name__ == "__main__":
    port = 8000
    # Logs are already printed in the app initialization
    uvicorn.run("api:app", host="0.0.0.0", port=port, reload=True)