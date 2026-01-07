"""
Configuration management for Orpheus Hindi TTS
Handles environment variables and model parameters
"""

from dataclasses import dataclass
from typing import Optional
import os

@dataclass
class TTSConfig:
    """
    TTS Configuration for Orpheus Hindi model
    All parameters optimized for banking/finance use case
    """
    # API Configuration
    api_url: str = "http://localhost:5006/v1/completions"
    api_timeout: int = 120  # seconds
    
    # Model Parameters - Orpheus-specific
    max_tokens: int = 8192
    temperature: float = 0.6  # Controls randomness (lower = more deterministic)
    top_p: float = 0.9  # Nucleus sampling
    repetition_penalty: float = 1.1  # Fixed for Orpheus stability
    
    # Audio Configuration
    sample_rate: int = 24000  # Hz - Orpheus standard
    channels: int = 1  # Mono
    bit_depth: int = 16  # 16-bit PCM
    
    # Model Identification
    model_name: str = "Orpheus-3b-Hindi-FT-Q8_0.gguf"
    language: str = "hindi"
    
    # Voice Configuration - Hindi model
    available_voices: dict = None
    default_voice: str = "ऋतिका"
    
    # Performance Tuning
    batch_size: int = 32  # Token batch size
    context_window: int = 49  # 7^2 - Orpheus standard for token alignment
    workers: int = 4  # Parallel workers for high concurrency
    
    # Streaming Configuration
    chunk_size: int = 2048  # Bytes per stream chunk
    crossfade_duration: float = 0.05  # seconds for long-form audio stitching
    
    # Banking/Finance Optimization
    min_latency: int = 100  # Target minimum latency in ms
    max_latency: int = 300  # Target maximum latency in ms
    
    def __post_init__(self):
        """Initialize default voice config after init"""
        if self.available_voices is None:
            self.available_voices = {
                "ऋतिका": {
                    "name": "Ritika",
                    "gender": "female",
                    "language": "hindi",
                    "description": "Female, expressive - optimized for banking conversations",
                    "use_case": ["banking", "finance", "customer_service", "emi_loans"]
                }
            }
    
    @staticmethod
    def from_env() -> 'TTSConfig':
        """
        Load configuration from environment variables
        Useful for Docker deployment
        """
        return TTSConfig(
            api_url=os.getenv("ORPHEUS_API_URL", "http://localhost:5006/v1/completions"),
            api_timeout=int(os.getenv("ORPHEUS_API_TIMEOUT", "120")),
            max_tokens=int(os.getenv("ORPHEUS_MAX_TOKENS", "8192")),
            temperature=float(os.getenv("ORPHEUS_TEMPERATURE", "0.6")),
            top_p=float(os.getenv("ORPHEUS_TOP_P", "0.9")),
            sample_rate=int(os.getenv("ORPHEUS_SAMPLE_RATE", "24000")),
            model_name=os.getenv("ORPHEUS_MODEL_NAME", "Orpheus-3b-Hindi-FT-Q8_0.gguf")
        )
    
    def validate(self) -> bool:
        """
        Validate configuration parameters
        """
        assert self.sample_rate in [24000], "Only 24kHz supported for Orpheus"
        assert self.channels == 1, "Only mono audio supported"
        assert self.bit_depth == 16, "Only 16-bit PCM supported"
        assert self.temperature >= 0.0 and self.temperature <= 1.0
        assert self.top_p >= 0.0 and self.top_p <= 1.0
        assert self.api_timeout > 0
        return True

# Emotion tags supported by Orpheus Hindi model
EMOTION_TAGS = {
    "<laugh>": "Add laughter",
    "<chuckle>": "Add chuckle/light laughter",
    "<sigh>": "Add sighing sound",
    "<cough>": "Add cough sound",
    "<sniffle>": "Add sniffling sound",
    "<groan>": "Add groaning sound",
    "<yawn>": "Add yawning sound",
    "<gasp>": "Add gasping sound"
}

# Prompt template for Hindi TTS
# Format: voice_name: hindi_text
HINDI_PROMPT_TEMPLATE = "{voice}: {text}"

# Max text length for single generation
MAX_TEXT_LENGTH = 5000

# Long-form audio settings
LONG_FORM_THRESHOLD = 1000  # Characters
LONG_FORM_BATCH_SIZE = 500  # Characters per chunk
CROSSFADE_MS = 50  # Milliseconds for crossfade between chunks
