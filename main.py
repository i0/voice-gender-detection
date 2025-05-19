import os
import json
import logging
import torch
import uvicorn
import asyncio
import time
from pathlib import Path
from tempfile import NamedTemporaryFile
from contextlib import asynccontextmanager
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydub import AudioSegment
from gender_predictor import get_gender_and_score
from typing import Callable, Dict, Optional

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger("gender-api")

# Progress milestones (percentage complete)
class ProgressPhase:
    INITIAL = 0
    AUDIO_CONVERSION_START = 5
    AUDIO_CONVERSION_COMPLETE = 15
    MODEL_LOADING_START = 15
    # Increase the MODEL_LOADING_COMPLETE value to ensure better progress display
    MODEL_LOADING_COMPLETE = 60  # Was 40, increased to show more progress during download
    INFERENCE_START = 60         # Updated to match MODEL_LOADING_COMPLETE
    INFERENCE_COMPLETE = 95
    FINALIZING = 95
    COMPLETE = 100

class ProgressManager:
    """
    A central class to manage progress tracking throughout the processing pipeline.
    Ensures that progress only moves forward and provides a unified interface
    for progress updates.
    """
    def __init__(self, on_progress: Optional[Callable[[int, str], None]] = None):
        self.current_progress = ProgressPhase.INITIAL
        self.current_message = "Initializing..."
        self.on_progress = on_progress
        self.logger = logging.getLogger("progress-manager")
    
    def update(self, progress: int, message: str) -> None:
        """
        Update progress to a new value, ensuring it never moves backward.
        
        Args:
            progress: The new progress value (0-100)
            message: A descriptive message about the current progress
        """
        # Ensure progress is within bounds
        progress = max(0, min(100, progress))
        
        # Check if this is a download message which should override UI
        is_download_message = "DOWNLOADING" in message and "%" in message
        
        # For download messages, always update regardless of progress direction
        if is_download_message or progress >= self.current_progress:
            old_progress = self.current_progress
            
            # If it's a download message, we'll keep using the same progress value
            # but update the message, so the UI doesn't jump back
            if is_download_message and progress < self.current_progress:
                # Keep the progress at the same level, just update the message
                self.logger.info(f"Override with download message: {message} (keeping progress at {self.current_progress}%)")
                self.current_message = message
            else:
                # Normal forward progress update
                self.current_progress = progress
                self.current_message = message
                self.logger.info(f"Progress update: {old_progress}% → {progress}%: {message}")
            
            # Notify listener if available
            if self.on_progress:
                # Always send current_progress (not progress) to avoid backward jumps
                self.on_progress(self.current_progress, self.current_message)
        else:
            # Log attempt to move backward (but don't update)
            self.logger.warning(
                f"Ignoring backward progress update: {self.current_progress}% → {progress}%: {message}"
            )
    
    def get_state(self) -> Dict[str, any]:
        """Get current progress state as a dictionary"""
        return {
            "progress": self.current_progress,
            "message": self.current_message
        }
    
    def create_callback(self) -> Callable[[int, str], None]:
        """Create a callback function that updates progress via this manager"""
        return lambda progress, message: self.update(progress, message)
    
    def phase_callback(self, phase_start: int, phase_end: int) -> Callable[[float, str], None]:
        """
        Create a callback for a specific processing phase.
        
        Args:
            phase_start: The starting percentage of this phase
            phase_end: The ending percentage of this phase
            
        Returns:
            A callback function that maps phase progress (0-1) to overall progress
        """
        def callback(phase_progress: float, message: str) -> None:
            # Map phase progress (0-1) to global progress
            phase_progress = max(0, min(1, phase_progress))  # Clamp to 0-1
            global_progress = int(phase_start + phase_progress * (phase_end - phase_start))
            self.update(global_progress, message)
        return callback
    
    def audio_conversion_callback(self) -> Callable[[float, str], None]:
        """Create a callback for audio conversion progress (5-15%)"""
        return self.phase_callback(ProgressPhase.AUDIO_CONVERSION_START, ProgressPhase.AUDIO_CONVERSION_COMPLETE)
    
    def model_loading_callback(self) -> Callable[[float, str], None]:
        """Create a callback for model loading progress (15-40%)"""
        return self.phase_callback(ProgressPhase.MODEL_LOADING_START, ProgressPhase.MODEL_LOADING_COMPLETE)
    
    def inference_callback(self) -> Callable[[float, str], None]:
        """Create a callback for model inference progress (40-95%)"""
        return self.phase_callback(ProgressPhase.INFERENCE_START, ProgressPhase.INFERENCE_COMPLETE)

# Model configuration
MODEL_PATH = "alefiury/wav2vec2-large-xlsr-53-gender-recognition-librispeech"
LABEL2ID = {"female": 0, "male": 1}
ID2LABEL = {0: "female", 1: "male"}
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Create static directory if it doesn't exist
STATIC_DIR = Path("static")
STATIC_DIR.mkdir(exist_ok=True)

# Define lifespan context manager (replaces on_event("startup"))
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: log application information
    logger.info(f"🚀 Starting Gender Recognition API on device: {DEVICE}")
    logger.info(f"🌐 Server is running at: http://localhost:8000")
    logger.info(f"✨ Web UI: 🎙️ http://localhost:8000/ui")
    logger.info(f"📚 API Docs: 📋 http://localhost:8000/docs")
    
    yield  # This is where FastAPI serves the application
    
    # Shutdown: clean up resources if needed
    logger.info("🛑 Shutting down Gender Recognition API")

# Initialize FastAPI
app = FastAPI(
    title="Gender Recognition API",
    description="API for gender recognition based on audio files. This tool is provided for educational and research purposes only.",
    version="1.0.0",
    license_info={
        "name": "MIT License",
        "url": "https://opensource.org/licenses/MIT",
    },
    lifespan=lifespan,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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
                .timing-container { margin-top: 20px; padding: 10px; border: 1px solid #ddd; border-radius: 5px; background-color: #f9f9f9; }
                .timing-container h3 { margin-top: 0; color: #333; }
                .timing-table { width: 100%; border-collapse: collapse; }
                .timing-table th, .timing-table td { padding: 8px; text-align: left; border-bottom: 1px solid #ddd; }
                .timing-table th { background-color: #f2f2f2; }
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
                                        
                                        // Add timing information
                                        if (data.timing) {
                                            resultHtml += `
                                            <div class="timing-container">
                                                <h3>⏱️ Processing Times</h3>
                                                <table class="timing-table">
                                                    <tr>
                                                        <th>Process</th>
                                                        <th>Time (seconds)</th>
                                                    </tr>
                                                    <tr>
                                                        <td>Total Processing</td>
                                                        <td>${data.timing.total}s</td>
                                                    </tr>
                                                    <tr>
                                                        <td>Audio Conversion</td>
                                                        <td>${data.timing.conversion}s</td>
                                                    </tr>
                                                    <tr>
                                                        <td>Model Loading</td>
                                                        <td>${data.timing.model_loading}s</td>
                                                    </tr>
                                                    <tr>
                                                        <td>Inference</td>
                                                        <td>${data.timing.inference}s</td>
                                                    </tr>
                                                </table>
                                            </div>`;
                                        }
                                        
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
            # Start timing
            start_time = time.time()
            processing_times = {
                "total": 0,
                "conversion": 0,
                "model_loading": 0,
                "inference": 0
            }
            
            # Initialize thread-safe progress tracking system
            # We'll use a synchronized dictionary to store the latest progress to allow
            # worker threads to safely report progress to the main thread
            import threading
            progress_lock = threading.Lock()
            latest_progress = {"value": None}
            
            # Function to update progress from any thread
            def thread_safe_progress_update(progress, message):
                with progress_lock:
                    latest_progress["value"] = {
                        "status": "processing", 
                        "progress": progress, 
                        "message": message
                    }
            
            # Create a ProgressManager that reports to our thread-safe function
            progress_manager = ProgressManager(
                on_progress=thread_safe_progress_update
            )
            
            # Initial progress update
            progress_manager.update(ProgressPhase.INITIAL, "Starting processing...")
            
            # Send initial progress directly
            yield json.dumps({
                "status": "processing", 
                "progress": ProgressPhase.INITIAL,
                "message": "Starting processing..."
            }) + "\n"
                
            # Convert to WAV if needed
            wav_path = temp_path
            if ext != ".wav":
                conversion_start = time.time()
                progress_manager.update(
                    ProgressPhase.AUDIO_CONVERSION_START,
                    f"Converting {ext} to WAV format..."
                )
                
                # Do the actual conversion
                audio = AudioSegment.from_file(temp_path)
                wav_path = temp_path.replace(ext, ".wav")
                audio.export(wav_path, format="wav")
                
                processing_times["conversion"] = time.time() - conversion_start
                progress_manager.update(
                    ProgressPhase.AUDIO_CONVERSION_COMPLETE,
                    "Audio conversion complete"
                )
            
            # Process the audio file
            logger.info(f"🎵 Processing audio file: {file.filename}")
            progress_manager.update(
                ProgressPhase.MODEL_LOADING_START, 
                "Loading model files..."
            )
            
            # Record model loading start time
            model_loading_start = time.time()
            
            # Create a model download progress callback that maps to our progress phases
            model_callback = progress_manager.model_loading_callback()
            
            # Send an initial estimate based on cache size
            try:
                from gender_predictor import get_cache_size_mb, EXPECTED_CACHE_SIZE_MB
                cache_size_mb = get_cache_size_mb("./cache")
                # Map cache_size to a 0-1 progress within the model loading phase
                # Note: We're using min() to cap it at 1.0 (100%)
                cache_progress = min(1.0, cache_size_mb / EXPECTED_CACHE_SIZE_MB)
                model_callback(
                    cache_progress, 
                    f"Loading model files: {cache_size_mb:.1f}MB/{EXPECTED_CACHE_SIZE_MB}MB"
                )
            except Exception as e:
                logger.error(f"Failed to estimate cache size: {str(e)}")
                # Still provide a reasonable progress update even if estimation fails
                model_callback(0.1, "Loading model files...")
            
            # Start model processing in a separate task
            # Note: This is where our refactored model processing function will be used
            inference_start = None
            
            # Define our inference callback that receives progress from the model
            def inference_progress_callback(progress, message):
                nonlocal inference_start
                
                # Mark the start of inference if this is the first callback
                if inference_start is None and progress >= 0.01:  # Using a small threshold
                    inference_start = time.time()
                    processing_times["model_loading"] = time.time() - model_loading_start
                
                # Map model progress (0-1) to our inference phase (40-95%)
                inference_callback = progress_manager.inference_callback()
                inference_callback(progress, message)
            
            # Start model processing 
            model_task = asyncio.create_task(
                run_model_processing(wav_path, inference_progress_callback)
            )
            
            # Process progress updates while model is running
            last_sent_progress = None
            
            while not model_task.done():
                # Check for new progress updates from the thread-safe structure
                with progress_lock:
                    current_progress = latest_progress["value"]
                
                # Only send if we have a new progress update
                if current_progress is not None and current_progress != last_sent_progress:
                    # Send the progress update to the client
                    yield json.dumps(current_progress) + "\n"
                    last_sent_progress = current_progress
                    
                # Short wait before checking again
                await asyncio.sleep(0.2)
            
            # Get the result from the completed model task
            result = await model_task
            
            # Calculate timing information
            if inference_start:
                processing_times["inference"] = time.time() - inference_start
            processing_times["total"] = time.time() - start_time
            
            # Send the finalizing update
            progress_manager.update(
                ProgressPhase.FINALIZING, 
                "Generating results..."
            )
            
            # Get the final progress update
            with progress_lock:
                final_progress = latest_progress["value"]
            
            # Send the final progress update if we have one
            if final_progress is not None and final_progress != last_sent_progress:
                yield json.dumps(final_progress) + "\n"
            
            # Process the prediction results
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
                
            # Log timing information
            logger.info(f"⏱️ Total processing time: {processing_times['total']:.2f}s")
            logger.info(f"⏱️ Audio conversion: {processing_times['conversion']:.2f}s")
            logger.info(f"⏱️ Model loading: {processing_times['model_loading']:.2f}s")
            logger.info(f"⏱️ Inference: {processing_times['inference']:.2f}s")
            
            # Mark as complete (100%)
            progress_manager.update(ProgressPhase.COMPLETE, "Processing complete")
            
            # Send the completion update
            with progress_lock:
                completion_progress = latest_progress["value"]
                
            if completion_progress is not None:
                yield json.dumps(completion_progress) + "\n"
                
            # Final result with predictions and timing
            yield json.dumps({
                "status": "complete", 
                "result": result,
                "prediction": {
                    "gender": predicted_gender,
                    "confidence": confidence,
                    "icon": gender_icon
                },
                "timing": {
                    "total": round(processing_times["total"], 2),
                    "conversion": round(processing_times["conversion"], 2),
                    "model_loading": round(processing_times["model_loading"], 2),
                    "inference": round(processing_times["inference"], 2)
                }
            }) + "\n"
            
        except Exception as e:
            logger.error(f"❌ Error processing audio: {str(e)}")
            yield json.dumps({"status": "error", "message": str(e)}) + "\n"
        finally:
            # Clean up the temporary files
            if os.path.exists(temp_path):
                os.unlink(temp_path)
            
            # Check if wav_path was initialized and exists (only needed for non-wav files)
            if ext != ".wav" and 'wav_path' in locals() and os.path.exists(wav_path):
                os.unlink(wav_path)
    
    return StreamingResponse(progress_generator(), media_type="application/x-ndjson")
    

async def run_model_processing(wav_path, progress_callback):
    """
    Run model processing in an async-friendly way
    
    Args:
        wav_path: Path to the WAV file to process
        progress_callback: Callback function that accepts (progress, message) 
                           where progress is 0-1
    
    Returns:
        The result of gender prediction
    """
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
            progress_callback=progress_callback
        )
    )

if __name__ == "__main__":
    port = 8000
    # Logs are already printed in the app initialization
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)