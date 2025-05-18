import os
import torch
import torchaudio
import tqdm
from typing import List, Optional, Union, Dict
from torch.nn import functional as F
from torch.utils.data import DataLoader
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification, Wav2Vec2Processor


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
    """Run inference on audio files and return probabilities for each class."""
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
            
            # Calculate progress percentage - blend model loading (15-30%) with inference (30-95%)
            # This makes the progress bar more informative from the user perspective
            if total_batches > 0:
                inference_progress = (batch_idx + 1) / total_batches
                # Map inference_progress from 0-1 to 30-95%
                progress = int(30 + inference_progress * 65)
                
                # Call progress callback if provided
                if progress_callback:
                    if batch_idx == 0:
                        progress_callback(30, "Starting inference...")
                    elif batch_idx == total_batches - 1:  # Last batch
                        progress_callback(95, "Finalizing results...")
                    else:
                        step_desc = f"Processing batch {batch_idx + 1}/{total_batches}"
                        progress_callback(progress, step_desc)
            
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
        
    Returns:
        Dictionary with gender probabilities if one audio file, 
        or list of dictionaries if multiple audio files
    """
    # load feature extractor and model
    feature_extractor = AutoFeatureExtractor.from_pretrained(model_name_or_path, cache_dir="./cache")
    model = AutoModelForAudioClassification.from_pretrained(
        model_name_or_path,
        num_labels=len(label2id),
        label2id=label2id,
        id2label=id2label,
        cache_dir="./cache"
    )

    # prepare dataset & dataloader
    dataset = CustomDataset(audio_paths, max_audio_len=5)
    collate_fn = CollateFunc(
        processor=feature_extractor,
        padding=True,
        sampling_rate=feature_extractor.sampling_rate
    )
    loader = DataLoader(dataset, batch_size=16, collate_fn=collate_fn, shuffle=False)

    # inference
    results = predict_with_score(loader, model, device, progress_callback)
    # return single dict if only one input
    return results[0] if len(results) == 1 else results