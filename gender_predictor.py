import os
import time
import torch
import torchaudio
import tqdm
import subprocess
import logging
import threading
from typing import List, Optional, Union, Dict
from torch.nn import functional as F
from torch.utils.data import DataLoader
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification, Wav2Vec2Processor
from transformers import logging as transformers_logging
from huggingface_hub import HfFolder # Used in get_gender_and_score

# Configure logging
logging.basicConfig(level=logging.INFO)


# Expected model cache size in MB (1.28 GB)
EXPECTED_CACHE_SIZE_MB = 2408.7

# Cache the folder size to avoid excessive disk operations
_cache_size_cache = {}
_cache_size_last_check = 0

# Global state for download progress tracking
_download_phase = {"current": "feature_extractor"}
_stop_monitoring = None
_monitor_thread = None
_last_reported_percentage = 0

def get_cache_size_mb(cache_dir="./cache"):
    """Get the size of the cache directory in megabytes.
    
    Args:
        cache_dir: Path to the cache directory
        
    Returns:
        Size of the cache directory in megabytes or 0 if directory doesn't exist
    """
    global _cache_size_cache, _cache_size_last_check
    
    # Use cached value if checked within the last 0.5 seconds
    current_time = time.time()
    if cache_dir in _cache_size_cache and current_time - _cache_size_last_check < 0.5:
        return _cache_size_cache[cache_dir]
    
    _cache_size_last_check = current_time
    
    if not os.path.exists(cache_dir):
        _cache_size_cache[cache_dir] = 0
        return 0
        
    try:
        # Calculate total size by walking the directory
        total_size = 0
        for dirpath, dirnames, filenames in os.walk(cache_dir):
            for f in filenames:
                try:
                    fp = os.path.join(dirpath, f)
                    if os.path.exists(fp):
                        total_size += os.path.getsize(fp)
                except (PermissionError, FileNotFoundError, OSError) as e:
                    # Skip files that can't be accessed
                    continue
        
        # Convert to MB
        size_mb = total_size / (1024 * 1024)
        _cache_size_cache[cache_dir] = size_mb
        return size_mb
        
    except Exception:
        # Fallback to subprocess
        try:
            if os.name == 'posix':  # Linux/Mac
                output = subprocess.check_output(['du', '-sm', cache_dir]).decode('utf-8')
                size_mb = float(output.split()[0])
                _cache_size_cache[cache_dir] = size_mb
                return size_mb
            else:  # Windows or unknown
                _cache_size_cache[cache_dir] = 0
                return 0
        except Exception:
            _cache_size_cache[cache_dir] = 0
            return 0

class ProgressCallback:
    """
    Progress callback for Hugging Face model downloads that tracks progress
    using cache directory size.
    """
    def __init__(self, progress_callback=None, cache_dir="./cache"):
        """
        Initialize a progress callback that reports download progress.
        
        Args:
            progress_callback: Callback function that accepts (progress, message)
                              where progress is 0-1
            cache_dir: Path to the cache directory
        """
        self.progress_callback = progress_callback
        self.current_file = ""
        self.last_progress = -1  # Start at -1 to ensure first update happens
        self.cache_dir = cache_dir
        self.last_reported_cache_size = 0
        # Threshold for update in MB (smaller for more frequent updates)
        # Using a very small threshold to get more frequent updates
        self.update_threshold_mb = 0.5
    
    def __call__(self, current: int, total: int, file_name: str):
        """
        Progress callback for file downloads.
        
        Args:
            current: Current number of bytes downloaded
            total: Total file size in bytes
            file_name: Name of file being downloaded
        """
        # Update current file name if it's different and not None
        if file_name is not None and file_name != self.current_file:
            self.current_file = file_name
        
        # If file_name is None, use a default value
        display_file = self.current_file if self.current_file else "model files"
        
        # Skip if no callback is provided
        if not self.progress_callback:
            return
            
        # Estimate overall download progress using cache directory size
        try:
            cache_size_mb = get_cache_size_mb(self.cache_dir)
            
            # Only update if cache size has changed significantly
            if abs(cache_size_mb - self.last_reported_cache_size) > self.update_threshold_mb:
                self.last_reported_cache_size = cache_size_mb
                
                # Calculate normalized progress (0-1) based on expected total size
                progress = min(1.0, cache_size_mb / EXPECTED_CACHE_SIZE_MB)
                
                # Calculate a progress percentage for display
                progress_percent = int(progress * 100)
                
                # Only call progress_callback if progress has changed significantly
                if round(progress, 2) != self.last_progress:
                    # Create a detailed message with progress percentage
                    message = (
                        f"Downloading {display_file}: {progress_percent}% complete " 
                        f"({cache_size_mb:.1f}MB/{EXPECTED_CACHE_SIZE_MB}MB)"
                    )
                    
                    self.progress_callback(progress, message)
                    self.last_progress = round(progress, 2)
        except Exception as e:
            # Log error but don't crash if progress reporting fails
            logging.error(f"Error reporting download progress: {str(e)}")
            
            # Still try to provide a reasonable progress update
            if self.last_progress < 0:  # Only if we haven't reported any progress yet
                self.progress_callback(0.1, f"Downloading {display_file}... (progress estimation failed)")
                self.last_progress = 0.1


class CustomDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        dataset: List[str],
        basedir: Optional[str] = None,
        sampling_rate: int = 16000,
        max_audio_len: int = 5,
    ):
        self.dataset = dataset
        self.basedir = basedir
        self.sampling_rate = sampling_rate
        self.max_audio_len = max_audio_len

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        path = self.dataset[idx] if self.basedir is None else os.path.join(self.basedir, self.dataset[idx])
        speech, sr = torchaudio.load(path)
        if speech.shape[0] > 1:
            speech = torch.mean(speech, dim=0, keepdim=True)
        if sr != self.sampling_rate:
            resampler = torchaudio.transforms.Resample(sr, self.sampling_rate)
            speech = resampler(speech)
        # pad/truncate
        max_len = self.max_audio_len * self.sampling_rate
        if speech.shape[1] < max_len:
            pad = torch.zeros(1, max_len - speech.shape[1])
            speech = torch.cat([speech, pad], dim=1)
        else:
            speech = speech[:, :max_len]
        return {"input_values": speech.squeeze().numpy(), "attention_mask": None}


class CollateFunc:
    def __init__(
        self,
        processor: Wav2Vec2Processor,
        padding: Union[bool, str] = True,
        pad_to_multiple_of: Optional[int] = None,
        return_attention_mask: bool = True,
        sampling_rate: int = 16000,
        max_length: Optional[int] = None,
    ):
        self.processor = processor
        self.padding = padding
        self.pad_to_multiple_of = pad_to_multiple_of
        self.return_attention_mask = return_attention_mask
        self.sampling_rate = sampling_rate
        self.max_length = max_length

    def __call__(self, batch):
        inputs = [x["input_values"] for x in batch]
        processed = self.processor(
            inputs,
            sampling_rate=self.sampling_rate,
            return_tensors="pt",
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_attention_mask=self.return_attention_mask,
        )
        return {
            "input_values": processed.input_values,
            "attention_mask": processed.attention_mask if self.return_attention_mask else None
        }


def predict_with_score(
    dataloader: DataLoader,
    model: torch.nn.Module,
    device: torch.device,
    progress_callback=None
) -> List[Dict[str, float]]:
    """
    Run inference on audio files and return probabilities for each class.
    
    Args:
        dataloader: DataLoader with audio inputs
        model: The model to use for inference
        device: The device to run inference on
        progress_callback: Optional callback function that accepts (progress, message)
                          where progress is 0-1 representing inference progress
                          
    Returns:
        List of dictionaries with prediction probabilities for each class
    """
    model.to(device)
    model.eval()
    all_results = []

    with torch.no_grad():
        total_batches = len(dataloader)
        progress_bar = tqdm.tqdm(dataloader, desc="Running inference")
        
        for batch_idx, batch in enumerate(progress_bar):
            inputs = batch['input_values'].to(device)
            masks = batch['attention_mask'].to(device) if batch['attention_mask'] is not None else None
            outputs = model(inputs, attention_mask=masks).logits
            probs = F.softmax(outputs, dim=-1).cpu().numpy()
            
            # Calculate normalized inference progress (0-1)
            if total_batches > 0 and progress_callback:
                # Calculate inference progress from 0 to 1
                inference_progress = (batch_idx + 1) / total_batches
                
                # Create appropriate message based on progress
                if batch_idx == 0:
                    message = "🔍 STARTING ANALYSIS - Processing audio..."
                elif batch_idx == total_batches - 1:  # Last batch
                    message = "🔍 COMPLETING ANALYSIS - Finalizing results..."
                else:
                    percent_done = int(100 * (batch_idx + 1) / total_batches)
                    message = f"🔍 ANALYZING AUDIO: {percent_done}% (Batch {batch_idx + 1}/{total_batches})"
                
                # Report progress to callback
                progress_callback(inference_progress, message)
            
            for p in probs:
                # map each label to its probability
                all_results.append({model.config.id2label[i]: float(p[i]) for i in range(len(p))})

    return all_results


def get_gender_and_score(
    model_name_or_path: str,
    audio_paths: List[str],
    label2id: Dict[str, int],
    id2label: Dict[int, str],
    device: torch.device = torch.device('cpu'),
    progress_callback=None
) -> Union[Dict[str, float], List[Dict[str, float]]]:
    """
    Predict gender from audio files.
    
    Args:
        model_name_or_path: HuggingFace model name or path to local model
        audio_paths: List of paths to audio files
        label2id: Dictionary mapping labels to IDs
        id2label: Dictionary mapping IDs to labels
        device: Device to run inference on (cpu or cuda)
        progress_callback: Optional callback function that accepts (progress, message)
                          where progress is 0-1 (normalized progress)
        
    Returns:
        Dictionary with gender probabilities if one audio file, 
        or list of dictionaries if multiple audio files
    """
    global _download_phase, _stop_monitoring, _monitor_thread, _last_reported_percentage
    cache_dir = "./cache"
    
    # Track progress in two phases: model loading (0-0.7) and inference (0.7-1.0)
    # Giving more weight to model loading since it's a significant part of the process
    model_loading_callback = None
    inference_callback = None
    
    if progress_callback:
        # Create a callback for model loading phase (0-0.7 of total progress)
        def model_loading_callback(phase_progress, message):
            # Map phase progress (0-1) to overall progress (0-0.7)
            overall_progress = phase_progress * 0.7
            progress_callback(overall_progress, message)
        
        # Create a callback for inference phase (0.7-1.0 of total progress)
        def inference_callback(phase_progress, message):
            # Map phase progress (0-1) to overall progress (0.7-1.0)
            overall_progress = 0.7 + phase_progress * 0.3
            progress_callback(overall_progress, message)
        
        # Initial progress report
        initial_cache_size_mb = get_cache_size_mb(cache_dir)
        percentage = int(100 * initial_cache_size_mb / EXPECTED_CACHE_SIZE_MB)
        model_loading_callback(
            0.0, 
            f"⬇️ STARTING MODEL DOWNLOAD: {percentage}% ({initial_cache_size_mb:.1f}MB/{EXPECTED_CACHE_SIZE_MB}MB)"
        )
        
        # Start a background thread to monitor cache size for progress updates
        # Reset global state
        _download_phase["current"] = "feature_extractor"
        _last_reported_percentage = 0
        
        # Create a new stop event if needed
        if _stop_monitoring is None or _monitor_thread is None or not _monitor_thread.is_alive():
            logging.info("MONITOR: Starting new monitoring thread")
            _stop_monitoring = threading.Event()
            
            def monitor_cache_size():
                last_size = 0
                logging.info("MONITOR: Thread started")
                
                # Calculate expected sizes for phases
                feature_size = 0.2 * EXPECTED_CACHE_SIZE_MB  # Feature extractor: ~20% of total
                model_size = 0.8 * EXPECTED_CACHE_SIZE_MB    # Main model: ~80% of total
                
                while not _stop_monitoring.is_set():
                    try:
                        current_size = get_cache_size_mb(cache_dir)
                        # Always log current state for debugging
                        logging.info(f"MONITOR: Cache size={current_size:.1f}MB, Phase={_download_phase['current']}, Last Size={last_size:.1f}MB")
                        
                        # Make threshold smaller to get more updates
                        if abs(current_size - last_size) > 0.1:  # Only update if size changed by more than 0.1MB
                            # Different progress scaling depending on which component we're downloading
                            if _download_phase["current"] == "feature_extractor":
                                # Feature extractor is about 20% of the total download
                                message_prefix = "⬇️ DOWNLOADING FEATURE EXTRACTOR"
                                
                                # For feature extractor: calculate percentage of feature extractor downloaded
                                percentage = int(100 * current_size / feature_size)
                                percentage = min(99, percentage)  # Cap at 99% to avoid confusion
                                
                                # For feature extractor: 0 to 0.2 progress
                                progress = min(0.2, 0.2 * current_size / feature_size)
                                size_ratio = f"({current_size:.1f}MB/{feature_size:.1f}MB)"
                                
                                # Auto-switch to model phase if feature extractor is done
                                if percentage >= 95 and last_size > 0 and (current_size - last_size) < 0.05:
                                    _download_phase["current"] = "model"
                                    logging.info("PHASE CHANGE: Auto-switching to model download phase")
                                    # Reset last size for new phase
                                    last_size = current_size
                                    continue
                            else:
                                # Main model is the remaining 80% of the download
                                message_prefix = "⬇️ DOWNLOADING MODEL"
                                
                                # For model: calculate percentage of model downloaded (separate from feature extractor)
                                adjusted_size = current_size - feature_size
                                if adjusted_size < 0:
                                    adjusted_size = 0
                                percentage = int(100 * adjusted_size / model_size)
                                percentage = min(99, percentage)  # Cap at 99% to avoid confusion
                                
                                # For main model: 0.2 to 0.7 progress
                                # Simplify calculation: pretend we're starting fresh for model size
                                download_progress = min(0.5, 0.5 * (adjusted_size / model_size))
                                progress = 0.2 + download_progress
                                size_ratio = f"({adjusted_size:.1f}MB/{model_size:.1f}MB)"
                            
                            # Only send updates when percentage changes or at least 1% change
                            global _last_reported_percentage
                            if percentage != _last_reported_percentage or abs(current_size - last_size) > (EXPECTED_CACHE_SIZE_MB * 0.01):
                                # Log before attempting callback
                                logging.info(f"MONITOR: Sending update - {message_prefix}: {percentage}% {size_ratio}")
                                
                                # Send progress update
                                try:
                                    model_loading_callback(
                                        progress, 
                                        f"{message_prefix}: {percentage}% {size_ratio}"
                                    )
                                    last_size = current_size
                                    _last_reported_percentage = percentage
                                except Exception as callback_error:
                                    logging.error(f"Error in progress callback: {str(callback_error)}")
                    except Exception as e:
                        logging.error(f"Error monitoring cache size: {str(e)}")
                    
                    # Adjust sleep time based on download phase
                    # More frequent updates during feature extractor download (smaller file)
                    sleep_time = 0.2 if _download_phase["current"] == "feature_extractor" else 0.5
                    time.sleep(sleep_time)
                
                logging.info("MONITOR: Thread exiting")
            
            # Start monitoring thread
            _monitor_thread = threading.Thread(target=monitor_cache_size)
            _monitor_thread.daemon = True  # Thread will exit when main thread exits
            _monitor_thread.start()
        else:
            logging.info("MONITOR: Reusing existing monitoring thread")
        
        # Set up the Hugging Face environment for better download progress reporting
        transformers_logging.set_verbosity_info()
        
        # Set environment variables for improved progress reporting
        os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
        
        # Instead of setting a callback directly (not supported in this version),
        # we'll rely on our cache directory size monitoring for progress
    
    try:
        # Load feature extractor and model
        feature_extractor = AutoFeatureExtractor.from_pretrained(
            model_name_or_path, 
            cache_dir=cache_dir,
            use_auth_token=HfFolder.get_token(),
            force_download=False
        )
        
        # Switch to model download phase
        _download_phase["current"] = "model"
        logging.info("PHASE CHANGE: Switching to model download phase")
        
        # Report after feature extractor is loaded (approximately 20% through model loading)
        if model_loading_callback:
            model_loading_callback(0.2, "✅ FEATURE EXTRACTOR LOADED - Downloading main model...")
        
        # Load the model with an additional progress update halfway through
        if model_loading_callback:
            # Add an intermediate progress update
            model_loading_callback(0.75, "⚙️ SETTING UP MODEL - Preparing architecture...")
        
        model = AutoModelForAudioClassification.from_pretrained(
            model_name_or_path,
            num_labels=len(label2id),
            label2id=label2id,
            id2label=id2label,
            cache_dir=cache_dir,
            use_auth_token=HfFolder.get_token(),
            force_download=False
        )
        
        # Add another progress update before final completion
        if model_loading_callback:
            model_loading_callback(0.85, "⚙️ INITIALIZING MODEL - Loading parameters...")
            
        # Report model loaded successfully
        if model_loading_callback:
            model_loading_callback(0.95, "✅ MODEL READY - Setup complete")
            
        # Stop the cache size monitoring thread if it exists
        if _stop_monitoring is not None:
            logging.info("MONITOR: Stopping monitoring thread")
            _stop_monitoring.set()
    
    except Exception as e:
        # Stop the cache size monitoring thread if it exists
        if _stop_monitoring is not None:
            logging.info("MONITOR: Stopping monitoring thread")
            _stop_monitoring.set()
            
        if progress_callback:
            progress_callback(0, f"Error loading model: {str(e)}")
        raise

    # Prepare dataset & dataloader
    dataset = CustomDataset(audio_paths, max_audio_len=5)
    collate_fn = CollateFunc(
        processor=feature_extractor,
        padding=True,
        sampling_rate=feature_extractor.sampling_rate
    )
    loader = DataLoader(dataset, batch_size=16, collate_fn=collate_fn, shuffle=False)

    # Run inference with the appropriate progress callback
    results = predict_with_score(
        loader, 
        model, 
        device, 
        progress_callback=inference_callback if progress_callback else None
    )
    
    # Return single dict if only one input
    return results[0] if len(results) == 1 else results