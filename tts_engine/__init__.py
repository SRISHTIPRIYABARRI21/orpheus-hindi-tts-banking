"""
Orpheus Hindi TTS Engine Package
Production-grade Text-to-Speech for banking/finance applications
"""

from .config import TTSConfig, EMOTION_TAGS
from .inference import HindiTTSEngine
from .speechpipe import AudioConverter

__all__ = [
    'TTSConfig',
    'HindiTTSEngine',
    'AudioConverter',
    'EMOTION_TAGS'
]

__version__ = '1.0.0'
