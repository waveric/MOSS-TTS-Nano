"""
MOSS-TTS-Nano CUDA HTTP Service

A FastAPI service for TTS synthesis using CUDA acceleration.
Uses device='cuda' for GPU inference.

Usage:
    python serve_cuda.py [--port PORT] [--host HOST]

Endpoints:
    GET  /health                           - Health check (ready after model load + warmup)
    POST /api/generate                     - Synthesize with builtin voice
    POST /api/generate-with-reference      - Synthesize with custom reference audio
"""

from __future__ import annotations

import argparse
import base64
import io
import logging
import os
import tempfile
import threading
import time
import wave
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import JSONResponse

from moss_tts_nano_runtime import (
    DEFAULT_AUDIO_TOKENIZER_PATH,
    DEFAULT_CHECKPOINT_PATH,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_PROMPT_AUDIO_DIR,
    NanoTTSService,
    VoicePreset,
)

# Configure logging
logging.basicConfig(
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger("moss-tts-cuda")

# Default voice definitions with their audio files
DEFAULT_VOICE_FILES: dict[str, tuple[str, str]] = {
    "Junhao": ("zh_1.wav", "Chinese male voice A"),
    "Zhiming": ("zh_2.wav", "Chinese male voice B"),
    "Weiguo": ("zh_5.wav", "Chinese male voice C"),
    "Xiaoyu": ("zh_3.wav", "Chinese female voice A"),
    "Yuewen": ("zh_4.wav", "Chinese female voice B"),
    "Lingyu": ("zh_6.wav", "Chinese female voice C"),
    "Trump": ("en_1.wav", "Trump reference voice"),
    "Ava": ("en_2.wav", "English female voice A"),
    "Bella": ("en_3.wav", "English female voice B"),
    "Adam": ("en_4.wav", "English male voice A"),
    "Nathan": ("en_5.wav", "English male voice B"),
    "Sakura": ("jp_1.mp3", "Japanese female voice A"),
    "Yui": ("jp_2.wav", "Japanese female voice B"),
    "Aoi": ("jp_3.wav", "Japanese female voice C"),
    "Hina": ("jp_4.wav", "Japanese female voice D"),
    "Mei": ("jp_5.wav", "Japanese female voice E"),
}

DEFAULT_VOICE = "Junhao"


def build_voice_presets_with_checks() -> dict[str, VoicePreset]:
    """Build voice presets, checking for audio file existence and skipping missing ones."""
    presets: dict[str, VoicePreset] = {}
    missing_voices: list[str] = []

    for voice_name, (file_name, description) in DEFAULT_VOICE_FILES.items():
        prompt_path = (DEFAULT_PROMPT_AUDIO_DIR / file_name).resolve()
        if not prompt_path.exists():
            logger.warning(
                "Skipping voice '%s': reference audio not found at %s",
                voice_name,
                prompt_path,
            )
            missing_voices.append(voice_name)
            continue

        presets[voice_name] = VoicePreset(
            name=voice_name,
            prompt_audio_path=prompt_path,
            description=description,
        )

    if missing_voices:
        logger.warning("Skipped %d voices due to missing reference audio: %s", len(missing_voices), missing_voices)

    if DEFAULT_VOICE not in presets:
        if presets:
            fallback_voice = next(iter(presets))
            logger.warning("Default voice '%s' not available, using '%s' as fallback", DEFAULT_VOICE, fallback_voice)
        else:
            raise RuntimeError("No voice presets available - all reference audio files are missing!")

    logger.info("Loaded %d voice presets: %s", len(presets), list(presets.keys()))
    return presets


def audio_to_wav_bytes(audio_array: np.ndarray, sample_rate: int) -> bytes:
    """Convert audio array to WAV bytes (48kHz stereo)."""
    audio_np = np.asarray(audio_array, dtype=np.float32)

    # Handle different array shapes
    if audio_np.ndim == 1:
        audio_np = audio_np[:, None]
    elif audio_np.ndim == 2 and audio_np.shape[0] <= 8 and audio_np.shape[0] < audio_np.shape[1]:
        audio_np = audio_np.T
    elif audio_np.ndim != 2:
        raise ValueError(f"Unsupported audio array shape: {audio_np.shape}")

    # Convert to 16-bit PCM
    audio_np = np.clip(audio_np, -1.0, 1.0)
    audio_int16 = (audio_np * 32767.0).astype(np.int16)

    # Write to WAV buffer
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(int(audio_int16.shape[1]))
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(audio_int16.tobytes())

    buffer.seek(0)
    return buffer.read()


class ServiceState:
    """Global service state management."""

    def __init__(self):
        self.runtime: Optional[NanoTTSService] = None
        self.ready: bool = False
        self.error: Optional[str] = None
        self.lock = threading.RLock()
        self.warmup_complete: bool = False
        self.device: str = "cuda"

    def set_ready(self) -> None:
        with self.lock:
            self.ready = True
            self.error = None

    def set_error(self, error: str) -> None:
        with self.lock:
            self.ready = False
            self.error = error

    def is_ready(self) -> bool:
        with self.lock:
            return self.ready


# Global state
state = ServiceState()
app = FastAPI(title="MOSS-TTS-Nano CUDA Service")


def initialize_service(checkpoint_path: str, audio_tokenizer_path: str, output_dir: str) -> None:
    """Initialize the TTS service in a background thread."""
    global state

    try:
        logger.info("Initializing MOSS-TTS-Nano CUDA service...")

        # Check CUDA availability
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available - this service requires GPU")

        logger.info("CUDA available: %s", torch.cuda.get_device_name(0))

        # Build voice presets with existence checks
        voice_presets = build_voice_presets_with_checks()

        # Create service with CUDA device
        state.runtime = NanoTTSService(
            checkpoint_path=checkpoint_path,
            audio_tokenizer_path=audio_tokenizer_path,
            device="cuda",  # Force CUDA, not CPU
            dtype="auto",
            attn_implementation="auto",
            output_dir=output_dir,
            voice_presets=voice_presets,
        )

        logger.info("Loading model on device: %s, dtype: %s", state.runtime.device, state.runtime.dtype)

        # Preload model
        preload_result = state.runtime.preload(load_model=True)
        logger.info("Model loaded: %s", preload_result)

        # Warmup synthesis
        logger.info("Running warmup synthesis...")
        warmup_result = state.runtime.warmup(text="你好，欢迎使用。", voice=state.runtime.default_voice)
        logger.info("Warmup complete: elapsed=%.2fs, sample_rate=%d", warmup_result["elapsed_seconds"], warmup_result["sample_rate"])

        state.device = str(state.runtime.device)
        state.warmup_complete = True
        state.set_ready()
        logger.info("MOSS-TTS-Nano CUDA service is ready!")

    except Exception as e:
        logger.exception("Failed to initialize service")
        state.set_error(str(e))


@app.on_event("startup")
async def startup_event():
    """Start initialization in background thread."""
    # Get configuration from environment or defaults
    checkpoint_path = os.getenv("MOSS_CHECKPOINT_PATH", DEFAULT_CHECKPOINT_PATH)
    audio_tokenizer_path = os.getenv("MOSS_AUDIO_TOKENIZER_PATH", DEFAULT_AUDIO_TOKENIZER_PATH)
    output_dir = os.getenv("MOSS_OUTPUT_DIR", str(DEFAULT_OUTPUT_DIR))

    # Start initialization thread
    init_thread = threading.Thread(
        target=initialize_service,
        args=(checkpoint_path, audio_tokenizer_path, output_dir),
        daemon=True,
    )
    init_thread.start()


@app.get("/health")
async def health():
    """Health check endpoint - returns ok only after model is loaded and warmed up."""
    if not state.is_ready():
        error_msg = state.error or "Service initializing"
        return JSONResponse(
            status_code=503,
            content={"status": "not_ready", "error": error_msg},
        )

    return {
        "status": "ok",
        "device": state.device,
        "dtype": str(state.runtime.dtype) if state.runtime else "unknown",
        "warmup_complete": state.warmup_complete,
        "default_voice": state.runtime.default_voice if state.runtime else DEFAULT_VOICE,
        "available_voices": state.runtime.list_voice_names() if state.runtime else [],
    }


@app.post("/api/generate")
async def generate(
    text: str = Form(...),
    voice: str = Form(DEFAULT_VOICE),
    max_new_frames: int = Form(375),
    voice_clone_max_text_tokens: int = Form(75),
    tts_max_batch_size: int = Form(0),
    codec_max_batch_size: int = Form(0),
    do_sample: str = Form("1"),
    text_temperature: float = Form(1.0),
    text_top_p: float = Form(1.0),
    text_top_k: int = Form(50),
    audio_temperature: float = Form(0.8),
    audio_top_p: float = Form(0.95),
    audio_top_k: int = Form(25),
    audio_repetition_penalty: float = Form(1.2),
    seed: str = Form("0"),
):
    """
    Generate speech from text using a builtin voice.

    Args:
        text: Text to synthesize (required)
        voice: Voice preset name (default: Junhao)
        max_new_frames: Maximum new frames to generate
        voice_clone_max_text_tokens: Max text tokens for voice clone
        tts_max_batch_size: TTS batch size (0 = auto)
        codec_max_batch_size: Codec batch size (0 = auto)
        do_sample: Enable sampling ("1" or "0")
        text_temperature: Text generation temperature
        text_top_p: Text top-p sampling
        text_top_k: Text top-k sampling
        audio_temperature: Audio generation temperature
        audio_top_p: Audio top-p sampling
        audio_top_k: Audio top-k sampling
        audio_repetition_penalty: Audio repetition penalty
        seed: Random seed (0 = random)

    Returns:
        JSON with audio_base64 (WAV 48kHz) and sample_rate
    """
    if not state.is_ready():
        return JSONResponse(
            status_code=503,
            content={"error": "Service not ready. Please wait for model initialization."},
        )

    # Validate text
    normalized_text = str(text or "").strip()
    if not normalized_text:
        return JSONResponse(
            status_code=400,
            content={"error": "text is required"},
        )

    # Validate voice
    available_voices = state.runtime.list_voice_names()
    if voice not in available_voices:
        return JSONResponse(
            status_code=400,
            content={"error": f"Unknown voice '{voice}'. Available: {available_voices}"},
        )

    try:
        # Parse seed
        normalized_seed = None if seed in {"", "0"} else int(seed)

        # Run synthesis
        result = state.runtime.synthesize(
            text=normalized_text,
            voice=voice,
            mode="voice_clone",
            max_new_frames=int(max_new_frames),
            voice_clone_max_text_tokens=int(voice_clone_max_text_tokens),
            tts_max_batch_size=int(tts_max_batch_size),
            codec_max_batch_size=int(codec_max_batch_size),
            do_sample=do_sample == "1",
            text_temperature=float(text_temperature),
            text_top_p=float(text_top_p),
            text_top_k=int(text_top_k),
            audio_temperature=float(audio_temperature),
            audio_top_p=float(audio_top_p),
            audio_top_k=int(audio_top_k),
            audio_repetition_penalty=float(audio_repetition_penalty),
            seed=normalized_seed,
        )

        # Convert to WAV bytes
        wav_bytes = audio_to_wav_bytes(result["waveform_numpy"], int(result["sample_rate"]))

        return {
            "audio_base64": base64.b64encode(wav_bytes).decode("ascii"),
            "sample_rate": int(result["sample_rate"]),
            "elapsed_seconds": result["elapsed_seconds"],
            "voice": result["voice"],
        }

    except Exception as e:
        logger.exception("Synthesis failed")
        return JSONResponse(
            status_code=500,
            content={"error": f"Synthesis failed: {str(e)}"},
        )


@app.post("/api/generate-with-reference")
async def generate_with_reference(
    text: str = Form(...),
    prompt_audio: UploadFile = File(...),
    prompt_text: str = Form(""),
    max_new_frames: int = Form(375),
    voice_clone_max_text_tokens: int = Form(75),
    tts_max_batch_size: int = Form(0),
    codec_max_batch_size: int = Form(0),
    do_sample: str = Form("1"),
    text_temperature: float = Form(1.0),
    text_top_p: float = Form(1.0),
    text_top_k: int = Form(50),
    audio_temperature: float = Form(0.8),
    audio_top_p: float = Form(0.95),
    audio_top_k: int = Form(25),
    audio_repetition_penalty: float = Form(1.2),
    seed: str = Form("0"),
):
    """
    Generate speech from text using a custom reference audio.

    Args:
        text: Text to synthesize (required)
        prompt_audio: Reference audio file for voice cloning (required)
        prompt_text: Transcription of the reference audio (optional, improves quality)
        max_new_frames: Maximum new frames to generate
        voice_clone_max_text_tokens: Max text tokens for voice clone
        tts_max_batch_size: TTS batch size (0 = auto)
        codec_max_batch_size: Codec batch size (0 = auto)
        do_sample: Enable sampling ("1" or "0")
        text_temperature: Text generation temperature
        text_top_p: Text top-p sampling
        text_top_k: Text top-k sampling
        audio_temperature: Audio generation temperature
        audio_top_p: Audio top-p sampling
        audio_top_k: Audio top-k sampling
        audio_repetition_penalty: Audio repetition penalty
        seed: Random seed (0 = random)

    Returns:
        JSON with audio_base64 (WAV 48kHz) and sample_rate
    """
    if not state.is_ready():
        return JSONResponse(
            status_code=503,
            content={"error": "Service not ready. Please wait for model initialization."},
        )

    # Validate text
    normalized_text = str(text or "").strip()
    if not normalized_text:
        return JSONResponse(
            status_code=400,
            content={"error": "text is required"},
        )

    # Save uploaded file temporarily
    temp_dir = Path(tempfile.mkdtemp(prefix="moss_tts_ref_"))
    prompt_audio_path = temp_dir / f"reference{Path(prompt_audio.filename or 'audio.wav').suffix}"

    try:
        # Write uploaded file
        content = await prompt_audio.read()
        prompt_audio_path.write_bytes(content)

        logger.info("Processing reference audio: %s (%d bytes)", prompt_audio.filename, len(content))

        # Parse seed
        normalized_seed = None if seed in {"", "0"} else int(seed)

        # Run synthesis with custom reference
        result = state.runtime.synthesize(
            text=normalized_text,
            mode="voice_clone",
            prompt_audio_path=str(prompt_audio_path),
            prompt_text=prompt_text or None,
            max_new_frames=int(max_new_frames),
            voice_clone_max_text_tokens=int(voice_clone_max_text_tokens),
            tts_max_batch_size=int(tts_max_batch_size),
            codec_max_batch_size=int(codec_max_batch_size),
            do_sample=do_sample == "1",
            text_temperature=float(text_temperature),
            text_top_p=float(text_top_p),
            text_top_k=int(text_top_k),
            audio_temperature=float(audio_temperature),
            audio_top_p=float(audio_top_p),
            audio_top_k=int(audio_top_k),
            audio_repetition_penalty=float(audio_repetition_penalty),
            seed=normalized_seed,
        )

        # Convert to WAV bytes
        wav_bytes = audio_to_wav_bytes(result["waveform_numpy"], int(result["sample_rate"]))

        return {
            "audio_base64": base64.b64encode(wav_bytes).decode("ascii"),
            "sample_rate": int(result["sample_rate"]),
            "elapsed_seconds": result["elapsed_seconds"],
        }

    except Exception as e:
        logger.exception("Synthesis with reference failed")
        return JSONResponse(
            status_code=500,
            content={"error": f"Synthesis failed: {str(e)}"},
        )

    finally:
        # Cleanup temp file
        try:
            if prompt_audio_path.exists():
                prompt_audio_path.unlink()
            if temp_dir.exists():
                temp_dir.rmdir()
        except Exception:
            pass


@app.get("/api/voices")
async def list_voices():
    """List available voice presets."""
    if not state.is_ready():
        return JSONResponse(
            status_code=503,
            content={"error": "Service not ready"},
        )

    voices = state.runtime.list_voice_names()
    return {
        "voices": voices,
        "default_voice": state.runtime.default_voice,
    }


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="MOSS-TTS-Nano CUDA HTTP Service")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=18083, help="Port to bind (default: 18083)")
    parser.add_argument("--checkpoint-path", type=str, default=DEFAULT_CHECKPOINT_PATH, help="Model checkpoint path")
    parser.add_argument("--audio-tokenizer-path", type=str, default=DEFAULT_AUDIO_TOKENIZER_PATH, help="Audio tokenizer path")
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR), help="Output directory for generated audio")
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()

    logger.info("Starting MOSS-TTS-Nano CUDA Service on %s:%d", args.host, args.port)
    logger.info("Checkpoint: %s", args.checkpoint_path)
    logger.info("Audio tokenizer: %s", args.audio_tokenizer_path)
    logger.info("Output directory: %s", args.output_dir)

    # Set environment variables for initialization
    os.environ["MOSS_CHECKPOINT_PATH"] = args.checkpoint_path
    os.environ["MOSS_AUDIO_TOKENIZER_PATH"] = args.audio_tokenizer_path
    os.environ["MOSS_OUTPUT_DIR"] = args.output_dir

    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        log_level="info",
    )


if __name__ == "__main__":
    main()
