import os
import json
import base64
import io
import hashlib
from datetime import datetime
from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic import BaseModel
import torch
import soundfile as sf
import librosa
from pathlib import Path
from index_tts import TTS as IndexTTS  # Assumes git install

# New import for runtime HF download
from huggingface_hub import snapshot_download
import shutil

# Logging
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Ephemeral internal paths (/tmp - resets on cold starts, fine for testing)
TEMP_DIR = Path("/tmp/indextts")
TEMP_DIR.mkdir(exist_ok=True)
MODEL_DIR = TEMP_DIR / "indextts2_checkpoints"
VOICE_DB_PATH = TEMP_DIR / "voice_database.json"
VOICES_DIR = TEMP_DIR / "voices"
VOICES_DIR.mkdir(exist_ok=True)
CACHE_DIR = TEMP_DIR / "cache"
CACHE_DIR.mkdir(exist_ok=True)

# Bundled presets (minimal example - REPLACE with your full embeddings from presets.json)
# Format: {"language": {"emotion": [embedding_list]}} - e.g., from your file
PRESETS = {
    "en": {
        "neutral": [0.0] * 256,  # Placeholder: Replace with real 256-dim embedding
        "happy": [0.1] * 256     # Add sad, angry, etc. for all emotions
    },
    "es": {
        "neutral": [0.0] * 256,  # Spanish example
        "happy": [0.1] * 256
    }
    # TODO: Add fr, de, it, pt, ru, zh, ja, ko, ar, hi with their emotions/embeddings
    # Load your presets.json: Parse and paste the dict here for static use
}

device = "cuda" if torch.cuda.is_available() else "cpu"
logger.info(f"Using device: {device}")
if device == "cpu":
    logger.warning("CUDA not available - running on CPU. Performance will be significantly slower.")

# Globals
model = None
in_memory_db = {"voices": {}}

class CloneRequest(BaseModel):
    user_id: str
    language: str = "en"

class TTSRequest(BaseModel):
    user_id: str | None = None
    preset: str | None = None  # e.g., "en_neutral"
    text: str
    language: str = "en"
    emotion: str = "neutral"

app = FastAPI(title="IndexTTS-2 Serverless Test API")

@app.get("/")
async def root():
    """Root endpoint for health checks"""
    gpu_available = torch.cuda.is_available()
    return {
        "status": "running",
        "model": "indextts-v2",
        "device": device,
        "gpu_available": gpu_available,
        "gpu_count": torch.cuda.device_count() if gpu_available else 0,
        "model_loaded": model is not None
    }

@app.on_event("startup")
async def startup():
    global model, in_memory_db
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    if VOICE_DB_PATH.exists():
        in_memory_db = json.loads(VOICE_DB_PATH.read_text())
    else:
        VOICE_DB_PATH.write_text(json.dumps(in_memory_db))

    # Try to ensure model is available
    model_available = await ensure_model_available()
    if not model_available:
        logger.warning("Model not available during startup - will load on-demand")

    # Try to load the model (don't fail if it doesn't work initially)
    try:
        await load_model()
    except Exception as e:
        logger.warning(f"Initial model load failed: {e}. Model will be loaded on-demand.")
        # Don't raise an exception - let the API start without the model
        # The /run endpoint will try to load it when needed

@app.get("/health")
async def health():
    gpu_available = torch.cuda.is_available()
    gpu_count = torch.cuda.device_count() if gpu_available else 0
    return {
        "status": "healthy",
        "device": device,
        "gpu_available": gpu_available,
        "gpu_count": gpu_count,
        "storage": "ephemeral",
        "model_loaded": model is not None
    }

def load_db():
    global in_memory_db
    if VOICE_DB_PATH.exists():
        in_memory_db = json.loads(VOICE_DB_PATH.read_text())
    return in_memory_db

def save_db(db):
    global in_memory_db
    in_memory_db = db
    VOICE_DB_PATH.write_text(json.dumps(db))

@app.post("/clone")
async def clone(request: CloneRequest, file: UploadFile = File(...)):
    if not model: raise HTTPException(503, "Model not ready")
    if not file.content_type.startswith("audio/"): raise HTTPException(400, "Audio required")
    try:
        audio_data = await file.read()
        voice_path = VOICES_DIR / f"{request.user_id}.wav"
        voice_path.write_bytes(audio_data)

        audio, sr = librosa.load(io.BytesIO(audio_data), sr=22050)
        embedding = model.clone_voice(audio, sr, language=request.language)

        db = load_db()
        db["voices"][request.user_id] = {
            "language": request.language,
            "file_path": str(voice_path.relative_to(TEMP_DIR)),
            "embedding": embedding.tolist(),
            "created": datetime.now().isoformat()
        }
        save_db(db)

        logger.info(f"Cloned {request.user_id}")
        return {"status": "success", "user_id": request.user_id, "note": "Ephemeral for testing"}
    except Exception as e:
        logger.error(f"Clone error: {e}")
        raise HTTPException(500, str(e))

def get_embedding(user_id: str | None, preset: str | None, lang: str):
    if user_id:
        db = load_db()
        voice_data = db["voices"].get(user_id, {})
        emb = voice_data.get("embedding")
        if emb: return emb
        # Fallback re-embed if file exists
        voice_path = VOICES_DIR / f"{user_id}.wav"
        if voice_path.exists():
            audio, sr = librosa.load(voice_path, sr=22050)
            emb = model.clone_voice(audio, sr, lang).tolist()
            voice_data["embedding"] = emb
            save_db(load_db())
            return emb
    if preset:
        parts = preset.split("_")
        if len(parts) >= 2:
            return PRESETS.get(lang, {}).get(parts[1], None)
    return None

@app.post("/tts")
async def tts(request: TTSRequest):
    if not model: raise HTTPException(503, "Model not ready")
    try:
        emb_list = get_embedding(request.user_id, request.preset, request.language)
        if not emb_list: raise HTTPException(400, "Need user_id or preset (e.g., 'en_neutral')")
        emb = torch.tensor(emb_list).unsqueeze(0).to(device)

        cache_key = hashlib.md5(f"{request.text}:{request.user_id or request.preset}:{request.language}".encode()).hexdigest()
        cache_path = CACHE_DIR / f"{cache_key}.wav"
        if cache_path.exists():
            with open(cache_path, "rb") as f:
                return {"audio": base64.b64encode(f.read()).decode(), "format": "base64_wav", "cached": True}

        audio = model.synthesize(text=request.text, speaker_embedding=emb, language=request.language, emotion=request.emotion)

        buffer = io.BytesIO()
        sf.write(buffer, audio.cpu().numpy().squeeze(), 22050, format='WAV')
        audio_data = buffer.getvalue()
        audio_b64 = base64.b64encode(audio_data).decode()
        with open(cache_path, "wb") as f: f.write(audio_data)

        return {"audio": audio_b64, "format": "base64_wav", "cached": False}
    except Exception as e:
        logger.error(f"TTS error: {e}")
        raise HTTPException(500, str(e))

class RunPodRequest(BaseModel):
    input: dict

@app.post("/run")
async def run_job(request: RunPodRequest):
    """RunPod serverless job endpoint"""
    try:
        job_input = request.input

        # Handle voice cloning requests
        if "audio_data" in job_input and "voice_name" in job_input:
            return await handle_voice_cloning(job_input)
        # Handle TTS requests
        elif "text" in job_input:
            return await handle_tts_request(job_input)
        else:
            raise HTTPException(400, "Invalid job input format")

    except Exception as e:
        logger.error(f"Run job error: {e}")
        raise HTTPException(500, str(e))

async def handle_voice_cloning(job_input: dict):
    """Handle voice cloning requests from /run endpoint"""
    try:
        # Extract parameters
        audio_data_b64 = job_input.get("audio_data")
        voice_name = job_input.get("voice_name", "unknown")
        user_id = job_input.get("user_id", "anonymous")
        language = job_input.get("language", "en")

        if not audio_data_b64:
            raise HTTPException(400, "audio_data is required")

        # Decode audio data
        audio_bytes = base64.b64decode(audio_data_b64)

        # Save audio file
        voice_path = VOICES_DIR / f"{user_id}.wav"
        voice_path.write_bytes(audio_bytes)

        # Load and process audio
        audio, sr = librosa.load(io.BytesIO(audio_bytes), sr=22050)

        if model is None:
            # Try to load model if not already loaded
            await load_model()

        if model is None:
            raise HTTPException(503, "Model not available")

        # Generate embedding
        embedding = model.clone_voice(audio, sr, language=language)

        # Save to database
        db = load_db()
        db["voices"][user_id] = {
            "language": language,
            "file_path": str(voice_path.relative_to(TEMP_DIR)),
            "embedding": embedding.tolist(),
            "created": datetime.now().isoformat()
        }
        save_db(db)

        logger.info(f"Voice cloned for user {user_id}")
        return {
            "status": "success",
            "user_id": user_id,
            "voice_id": user_id,
            "message": f"Voice '{voice_name}' cloned successfully"
        }

    except Exception as e:
        logger.error(f"Voice cloning error: {e}")
        raise HTTPException(500, str(e))

async def handle_tts_request(job_input: dict):
    """Handle TTS requests from /run endpoint"""
    try:
        # Extract parameters
        text = job_input.get("text")
        user_id = job_input.get("user_id")
        preset = job_input.get("preset")
        language = job_input.get("language", "en")
        emotion = job_input.get("emotion", "neutral")

        if not text:
            raise HTTPException(400, "text is required")

        if model is None:
            # Try to load model if not already loaded
            await load_model()

        if model is None:
            raise HTTPException(503, "Model not available")

        # Get embedding
        emb_list = get_embedding(user_id, preset, language)
        if not emb_list:
            raise HTTPException(400, f"Need user_id or preset for voice synthesis")

        emb = torch.tensor(emb_list).unsqueeze(0).to(device)

        # Check cache
        cache_key = hashlib.md5(f"{text}:{user_id or preset}:{language}".encode()).hexdigest()
        cache_path = CACHE_DIR / f"{cache_key}.wav"

        if cache_path.exists():
            with open(cache_path, "rb") as f:
                audio_data = f.read()
        else:
            # Generate audio
            audio = model.synthesize(text=text, speaker_embedding=emb, language=language, emotion=emotion)

            # Convert to WAV
            buffer = io.BytesIO()
            sf.write(buffer, audio.cpu().numpy().squeeze(), 22050, format='WAV')
            audio_data = buffer.getvalue()

            # Cache the result
            with open(cache_path, "wb") as f:
                f.write(audio_data)

        # Return base64 encoded audio
        audio_b64 = base64.b64encode(audio_data).decode()
        return {
            "audio": audio_b64,
            "format": "base64_wav"
        }

    except Exception as e:
        logger.error(f"TTS request error: {e}")
        raise HTTPException(500, str(e))

async def ensure_model_available():
    """Ensure the IndexTTS model files are available"""
    global model

    # Check if model directory has files
    if any(MODEL_DIR.iterdir()):
        logger.info("Model files already present in MODEL_DIR")
        return True

    # Try to download from HuggingFace
    hf_token = os.getenv("HF_TOKEN")
    if hf_token:
        try:
            logger.info("Downloading IndexTTS model from HuggingFace...")
            ret_path = snapshot_download(
                repo_id="IndexTeam/IndexTTS-2",
                local_dir=str(MODEL_DIR),
                token=hf_token,
                local_files_only=False
            )
            logger.info(f"Model downloaded to: {ret_path}")

            # If snapshot_download created a nested folder, move contents up
            if not any(MODEL_DIR.iterdir()):
                nested = Path(ret_path)
                if nested.exists() and nested.is_dir():
                    for child in nested.iterdir():
                        dest = MODEL_DIR / child.name
                        if child.is_dir():
                            shutil.copytree(child, dest, dirs_exist_ok=True)
                        else:
                            shutil.copy2(child, dest)

            if any(MODEL_DIR.iterdir()):
                logger.info("✅ Model files successfully downloaded")
                return True
            else:
                logger.error("❌ Model download completed but no files found")
                return False

        except Exception as e:
            logger.error(f"❌ HuggingFace download failed: {e}")

    # Try to copy from baked-in location
    try:
        src = Path("/app/indextts2_checkpoints")
        if src.exists() and any(src.iterdir()):
            logger.info("Copying model from /app/indextts2_checkpoints...")
            shutil.copytree(src, MODEL_DIR, dirs_exist_ok=True)
            logger.info("✅ Model files copied from baked-in location")
            return True
    except Exception as e:
        logger.error(f"❌ Failed to copy from /app: {e}")

    logger.error("❌ Model files not available")
    return False

async def load_model():
    """Try to load the IndexTTS model"""
    global model
    if model is not None:
        return

    try:
        logger.info("Attempting to load IndexTTS model...")
        model = IndexTTS(model_dir=str(MODEL_DIR), device=device)
        logger.info("✅ Model loaded successfully!")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        model = None

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)