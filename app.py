#!/usr/bin/env python3
"""
Orpheus Hindi TTS FastAPI Server
Production-grade Text-to-Speech API for banking/finance voice AI applications
Optimized for real-time streaming with <300ms latency target
"""

import os
import sys
import asyncio
import logging
from pathlib import Path
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import TTS engine
from tts_engine.inference import HindiTTSEngine
from tts_engine.config import TTSConfig

# Initialize FastAPI app
app = FastAPI(
    title="Orpheus Hindi TTS API",
    description="Production-grade Hindi Text-to-Speech with banking/finance optimization",
    version="1.0.0"
)

# CORS middleware for cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create necessary directories
Path("outputs").mkdir(exist_ok=True)
Path("static").mkdir(exist_ok=True)
Path("templates").mkdir(exist_ok=True)

# Mount static files
if Path("static").exists():
    app.mount("/static", StaticFiles(directory="static"), name="static")

# Initialize TTS configuration
config = TTSConfig(
    api_url=os.getenv("ORPHEUS_API_URL", "http://localhost:5006/v1/completions"),
    api_timeout=int(os.getenv("ORPHEUS_API_TIMEOUT", "120")),
    max_tokens=int(os.getenv("ORPHEUS_MAX_TOKENS", "8192")),
    temperature=float(os.getenv("ORPHEUS_TEMPERATURE", "0.6")),
    top_p=float(os.getenv("ORPHEUS_TOP_P", "0.9")),
    sample_rate=int(os.getenv("ORPHEUS_SAMPLE_RATE", "24000")),
    model_name=os.getenv("ORPHEUS_MODEL_NAME", "Orpheus-3b-Hindi-FT-Q8_0.gguf")
)

# Initialize TTS engine
try:
    tts_engine = HindiTTSEngine(config)
    logger.info("TTS Engine initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize TTS Engine: {e}")
    sys.exit(1)

# Request/Response Models
class SpeakRequest(BaseModel):
    """Legacy /speak endpoint request"""
    text: str = Field(..., description="Hindi text to convert to speech", min_length=1, max_length=5000)
    voice: str = Field(default="ऋतिका", description="Voice to use (default: ऋतिका)")
    speed: float = Field(default=1.0, ge=0.5, le=1.5, description="Speech speed multiplier")

class OpenAISpeechRequest(BaseModel):
    """OpenAI-compatible /v1/audio/speech endpoint request"""
    input: str = Field(..., description="Hindi text to convert to speech", min_length=1, max_length=5000)
    model: str = Field(default="orpheus-hindi", description="Model to use")
    voice: str = Field(default="ऋतिका", description="Voice to use")
    response_format: str = Field(default="wav", description="Output format (wav only)")
    speed: float = Field(default=1.0, ge=0.5, le=1.5, description="Speech speed multiplier")

class AvailableVoicesResponse(BaseModel):
    """Available voices response model"""
    voices: dict = Field(..., description="Available voices with descriptions")
    language: str = Field(default="Hindi", description="Language")
    model: str = Field(..., description="Model name")

# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        status = await tts_engine.check_health()
        return {
            "status": "healthy" if status else "unhealthy",
            "model": config.model_name,
            "language": "Hindi",
            "voice": "ऋतिका (Ritika)"
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=503, detail="Service unavailable")

# Get available voices
@app.get("/voices", response_model=AvailableVoicesResponse)
async def get_voices():
    """Get available voices"""
    voices = {
        "ऋतिका": "Female, Hindi, expressive - optimized for banking/finance conversations"
    }
    return AvailableVoicesResponse(
        voices=voices,
        language="Hindi",
        model=config.model_name
    )

# Legacy /speak endpoint
@app.post("/speak")
async def speak(
    request: SpeakRequest,
    background_tasks: BackgroundTasks
):
    """
    Legacy endpoint for speech synthesis
    Returns WAV audio file
    """
    try:
        logger.info(f"Processing speak request: {len(request.text)} chars, voice={request.voice}")
        
        audio_data = await tts_engine.synthesize(
            text=request.text,
            voice=request.voice,
            speed=request.speed
        )
        
        if not audio_data:
            raise HTTPException(status_code=500, detail="Audio generation failed")
        
        # Save temporary file
        output_file = f"outputs/speech_{os.urandom(8).hex()}.wav"
        with open(output_file, "wb") as f:
            f.write(audio_data)
        
        # Schedule cleanup
        background_tasks.add_task(lambda: Path(output_file).unlink(missing_ok=True))
        
        return FileResponse(
            path=output_file,
            media_type="audio/wav",
            filename="speech.wav"
        )
    
    except Exception as e:
        logger.error(f"Speak endpoint error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# OpenAI-compatible endpoint
@app.post("/v1/audio/speech")
async def openai_speech(
    request: OpenAISpeechRequest,
    background_tasks: BackgroundTasks
):
    """
    OpenAI-compatible /v1/audio/speech endpoint
    Drop-in replacement for OpenAI TTS API
    """
    try:
        logger.info(f"Processing OpenAI speech request: {len(request.input)} chars")
        
        audio_data = await tts_engine.synthesize(
            text=request.input,
            voice=request.voice,
            speed=request.speed
        )
        
        if not audio_data:
            raise HTTPException(status_code=500, detail="Audio generation failed")
        
        # Save temporary file
        output_file = f"outputs/speech_{os.urandom(8).hex()}.wav"
        with open(output_file, "wb") as f:
            f.write(audio_data)
        
        # Schedule cleanup
        background_tasks.add_task(lambda: Path(output_file).unlink(missing_ok=True))
        
        return FileResponse(
            path=output_file,
            media_type="audio/wav",
            filename="speech.wav"
        )
    
    except Exception as e:
        logger.error(f"OpenAI endpoint error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Stream endpoint for real-time audio
@app.post("/stream")
async def stream_speech(request: SpeakRequest):
    """
    Stream audio in real-time for low-latency applications
    Ideal for banking/finance voice calls
    """
    try:
        logger.info(f"Processing stream request: {len(request.text)} chars")
        
        async def audio_generator():
            audio_data = await tts_engine.synthesize(
                text=request.text,
                voice=request.voice,
                speed=request.speed,
                stream=True
            )
            
            if isinstance(audio_data, bytes):
                yield audio_data
            else:
                async for chunk in audio_data:
                    yield chunk
        
        return StreamingResponse(
            audio_generator(),
            media_type="audio/wav"
        )
    
    except Exception as e:
        logger.error(f"Stream endpoint error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Root endpoint
@app.get("/")
async def root():
    """Root endpoint - returns API information"""
    return {
        "name": "Orpheus Hindi TTS API",
        "version": "1.0.0",
        "language": "Hindi",
        "voice": "ऋतिका (Ritika) - Female, expressive",
        "model": config.model_name,
        "endpoints": {
            "health": "/health",
            "voices": "/voices",
            "legacy_speak": "/speak (POST)",
            "openai_compatible": "/v1/audio/speech (POST)",
            "stream": "/stream (POST)"
        },
        "docs": "/docs",
        "redoc": "/redoc"
    }

if __name__ == "__main__":
    port = int(os.getenv("ORPHEUS_PORT", "5005"))
    host = os.getenv("ORPHEUS_HOST", "0.0.0.0")
    
    logger.info(f"Starting Orpheus Hindi TTS server on {host}:{port}")
    logger.info(f"API URL: {config.api_url}")
    logger.info(f"Model: {config.model_name}")
    
    uvicorn.run(
        app,
        host=host,
        port=port,
        log_level="info",
        access_log=True
    )
