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

@app.on_event("startup")
async def startup():
    global model, in_memory_db
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    if VOICE_DB_PATH.exists():
        in_memory_db = json.loads(VOICE_DB_PATH.read_text())
    else:
        VOICE_DB_PATH.write_text(json.dumps(in_memory_db))

    # If model not present in MODEL_DIR, try to download at runtime
    if not any(MODEL_DIR.iterdir()):
        hf_token = os.getenv("HF_TOKEN")
        try:
            logger.info("Model dir is empty — attempting runtime download from Hugging Face...")
            # snapshot_download will download repo contents into local_dir
            ret_path = snapshot_download(
                repo_id="IndexTeam/IndexTTS-2",
                local_dir=str(MODEL_DIR),
                token=hf_token
            )
            logger.info(f"snapshot_download returned: {ret_path}")
            # If snapshot_download created a nested folder, move contents up
            try:
                # If MODEL_DIR is empty but ret_path has files inside a nested subfolder, move them
                if not any(MODEL_DIR.iterdir()):
                    nested = Path(ret_path)
                    if nested.exists() and nested.is_dir():
                        for child in nested.iterdir():
                            dest = MODEL_DIR / child.name
                            if child.is_dir():
                                shutil.copytree(child, dest, dirs_exist_ok=True)
                            else:
                                shutil.copy2(child, dest)
                # Final check
                if not any(MODEL_DIR.iterdir()):
                    raise RuntimeError("snapshot_download completed but no files found in MODEL_DIR")
            except Exception as e_move:
                logger.warning(f"Post-download normalization warning: {e_move}")
            logger.info("Download complete (or files placed into MODEL_DIR).")
        except Exception as e:
            logger.warning(f"Hugging Face download failed: {e}. Attempting to copy /app/indextts2_checkpoints if present.")
            # Fallback: if you baked the model into the image at /app/indextts2_checkpoints, copy it
            try:
                src = Path("/app/indextts2_checkpoints")
                if src.exists() and any(src.iterdir()):
                    shutil.copytree(src, MODEL_DIR, dirs_exist_ok=True)
                    logger.info("Copied model from /app/indextts2_checkpoints to runtime MODEL_DIR.")
                else:
                    raise FileNotFoundError("/app/indextts2_checkpoints not present or empty")
            except Exception as e2:
                logger.error(f"Failed to obtain model via HF download or /app copy: {e2}")
                # Raise an HTTPException so RunPod shows a 503 and logs contain the reason
                raise HTTPException(status_code=503, detail=f"Model load failed: {e2}")

    # At this point MODEL_DIR should contain model files; load the model
    try:
        model = IndexTTS(model_dir=str(MODEL_DIR), device=device)
        logger.info("✅ Model loaded in ephemeral storage!")
    except Exception as e:
        logger.error(f"Load failed after download/copy: {e}")
        raise HTTPException(503, "Model load failed")

@app.get("/health")
async def health():
    return {"status": "healthy", "device": device, "storage": "ephemeral"}

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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)