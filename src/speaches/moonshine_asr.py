from __future__ import annotations

import asyncio
import logging
import time
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from huggingface_hub import hf_hub_download
from onnxruntime import InferenceSession

from speaches.api_types import TranscriptionSegment, TranscriptionWord
from speaches.text_utils import Transcription

if TYPE_CHECKING:
    from speaches.audio import Audio

logger = logging.getLogger(__name__)

# Moonshine model configuration
SAMPLE_RATE = 16000
FRAME_LENGTH = 25  # ms
FRAME_STRIDE = 10  # ms
NUM_MEL_BINS = 80
NUM_FRAMES = 100
FRAME_LEN = int(SAMPLE_RATE * FRAME_LENGTH / 1000)
FRAME_STEP = int(SAMPLE_RATE * FRAME_STRIDE / 1000)

# Default model repository and files
DEFAULT_MODEL_REPO = "UsefulSensors/moonshine"
MODEL_FILE = "model.onnx"
MEL_BASIS_FILE = "mel_basis.npy"

# ONNX providers - prefer CUDA if available
ONNX_PROVIDERS = ["CUDAExecutionProvider", "CPUExecutionProvider"]

VOCAB = [
    "<pad>", "<s>", "</s>", "<unk>", " ", "e", "t", "a", "o", "n", "i", "h", "s",
    "r", "d", "l", "u", "m", "w", "c", "f", "g", "y", "p", "b", "v", "k", "'",
    "x", "j", "q", "z", "th", "ed", "ing", "er", "at", "on", "re", "an", "en",
    "ng", "in", "es", "or", "nd", "ly", "te", "ti", "ar", "al", "st", "le", "is",
    "nt", "ne", "se", "ve", "ea", "it", "ou", "ur", "me", "ll", "de", "to", "he",
    "and", "the", "of", "in", "that", "for", "you", "with", "was", "this", "have",
    "are", "they", "not", "but", "had", "what", "when", "were", "we", "will",
    "more", "an", "their", "about", "which", "like", "then", "him", "would",
    "make", "no", "just", "has", "them", "these", "so", "some", "can", "out",
    "time", "know", "people", "year", "into", "last", "than", "other", "says",
    "only", "new", "one", "all", "do", "she", "how", "could", "first", "way"
]


def get_model_path(model_id_or_path: str) -> tuple[Path, Path]:
    """Get the paths to the model and mel basis files.
    
    Args:
        model_id_or_path: Either a HuggingFace model ID or a local path.
            If it's a HuggingFace ID, the model will be downloaded.
            If it's a local path, it should be a directory containing the model files
            or a direct path to the model file.
    
    Returns:
        A tuple of (model_path, mel_basis_path)
    """
    path = Path(model_id_or_path)
    
    # If it's a direct file path
    if path.is_file():
        model_dir = path.parent
        model_path = path
        mel_basis_path = model_dir / MEL_BASIS_FILE
    # If it's a directory
    elif path.is_dir():
        model_dir = path
        model_path = model_dir / MODEL_FILE
        mel_basis_path = model_dir / MEL_BASIS_FILE
    # Otherwise assume it's a HuggingFace model ID
    else:
        # Extract repo name and revision
        if ":" in model_id_or_path:
            repo_id, revision = model_id_or_path.split(":", 1)
        else:
            repo_id, revision = model_id_or_path, None
            
        # Download files from HuggingFace
        model_path = Path(hf_hub_download(
            repo_id=repo_id,
            filename=MODEL_FILE,
            revision=revision,
            subfolder="onnx"  # Moonshine ONNX models are in the onnx subfolder
        ))
        mel_basis_path = Path(hf_hub_download(
            repo_id=repo_id,
            filename=MEL_BASIS_FILE,
            revision=revision,
            subfolder="onnx"
        ))
        
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found at {model_path}")
    if not mel_basis_path.exists():
        raise FileNotFoundError(f"Mel basis file not found at {mel_basis_path}")
        
    return model_path, mel_basis_path


class MoonshineASR:
    def __init__(
        self,
        model_path: str,
    ) -> None:
        """Initialize Moonshine ASR.
        
        Args:
            model_path: Either a HuggingFace model ID or a local path.
                If it's a HuggingFace ID, the model will be downloaded.
                If it's a local path, it should be a directory containing the model files
                or a direct path to the model file.
        """
        self.model_path, self.mel_basis_path = get_model_path(model_path)
        self.session = InferenceSession(str(self.model_path), providers=ONNX_PROVIDERS)
        
        # Get input/output names
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        
        # Load mel basis matrix
        self.mel_basis = np.load(self.mel_basis_path)

    def _preprocess_audio(self, audio_data: np.ndarray) -> np.ndarray:
        """Convert audio to mel spectrogram features."""
        # Normalize audio
        audio_data = audio_data.astype(np.float32) / np.iinfo(np.int16).max
        
        # Calculate number of frames
        num_frames = 1 + (len(audio_data) - FRAME_LEN) // FRAME_STEP
        
        # Extract frames
        indices = np.tile(np.arange(FRAME_LEN), (num_frames, 1)) + \
                 np.tile(np.arange(0, num_frames * FRAME_STEP, FRAME_STEP), (FRAME_LEN, 1)).T
        frames = audio_data[indices.astype(np.int32, copy=False)]
        
        # Apply window function
        frames *= np.hamming(FRAME_LEN)
        
        # Compute FFT
        fft = np.abs(np.fft.rfft(frames, n=FRAME_LEN))
        
        # Convert to mel scale using pre-loaded basis
        mel = np.dot(fft, self.mel_basis.T)
        
        # Apply log
        mel = np.log(mel + 1e-6)
        
        # Normalize
        mel = (mel - mel.mean()) / mel.std()
        
        return mel

    def _decode_output(self, output_data: np.ndarray) -> list[str]:
        """Convert model output logits to text."""
        # Get most likely tokens
        token_indices = np.argmax(output_data, axis=-1)
        
        # Convert tokens to text
        words = []
        current_word = ""
        
        for idx in token_indices[0]:  # Take first sequence
            if idx < len(VOCAB):
                token = VOCAB[idx]
                if token.startswith(" ") or token in ["</s>", "<s>", "<pad>", "<unk>"]:
                    if current_word:
                        words.append(current_word)
                        current_word = ""
                    if token.startswith(" "):
                        current_word = token[1:]
                else:
                    current_word += token
        
        if current_word:
            words.append(current_word)
        
        return words

    def _transcribe(
        self,
        audio: Audio,
        prompt: str | None = None,  # Currently unused but kept for API compatibility
    ) -> tuple[Transcription, dict]:
        start = time.perf_counter()
        
        # Preprocess audio
        features = self._preprocess_audio(audio.data)
        
        # Prepare input tensor
        input_tensor = np.expand_dims(features, axis=0).astype(np.float32)
        
        # Run inference
        output_data = self.session.run(
            [self.output_name],
            {self.input_name: input_tensor}
        )[0]
        
        # Decode output
        words = self._decode_output(output_data)
        
        # Create word timestamps (approximate based on audio length)
        word_duration = audio.duration / len(words)
        transcription_words = []
        
        for i, word in enumerate(words):
            start_time = i * word_duration + audio.start
            end_time = (i + 1) * word_duration + audio.start
            transcription_words.append(
                TranscriptionWord(
                    start=start_time,
                    end=end_time,
                    word=word,
                    probability=0.9  # Moonshine doesn't provide word-level confidence
                )
            )
        
        transcription = Transcription(transcription_words)
        
        end = time.perf_counter()
        logger.info(
            f"Transcribed {audio} in {end - start:.2f} seconds. Prompt: {prompt}. Transcription: {transcription.text}"
        )
        
        # Return minimal transcription info since Moonshine doesn't provide detailed metadata
        transcription_info = {
            "language": "en",
            "language_probability": 1.0
        }
        
        return (transcription, transcription_info)

    async def transcribe(
        self,
        audio: Audio,
        prompt: str | None = None,
    ) -> tuple[Transcription, dict]:
        """Wrapper around _transcribe so it can be used in async context."""
        return await asyncio.get_running_loop().run_in_executor(
            None,
            self._transcribe,
            audio,
            prompt,
        ) 