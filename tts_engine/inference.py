"""
Core TTS Inference Engine for Orpheus Hindi
Handles token generation, audio conversion, and streaming
Optimized for <300ms latency banking/finance applications
"""

import asyncio
import logging
import aiohttp
import time
from typing import Optional, AsyncGenerator, Union
import numpy as np
from .config import TTSConfig, HINDI_PROMPT_TEMPLATE, EMOTION_TAGS, LONG_FORM_THRESHOLD, LONG_FORM_BATCH_SIZE, CROSSFADE_MS
from .speechpipe import AudioConverter

logger = logging.getLogger(__name__)

class HindiTTSEngine:
    """
    Main TTS Engine for Orpheus Hindi
    Manages API communication, token generation, and audio synthesis
    """
    
    def __init__(self, config: TTSConfig):
        """
        Initialize TTS Engine
        
        Args:
            config: TTSConfig instance with model parameters
        """
        self.config = config
        self.config.validate()
        self.audio_converter = AudioConverter(config)
        self.session: Optional[aiohttp.ClientSession] = None
        self.metrics = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "total_latency_ms": 0
        }
        
        logger.info(f"TTS Engine initialized with config: {config.model_name}")
    
    async def _get_session(self) -> aiohttp.ClientSession:
        """
        Get or create aiohttp session with timeout
        """
        if self.session is None or self.session.closed:
            self.session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=self.config.api_timeout)
            )
        return self.session
    
    async def close(self):
        """
        Close HTTP session
        """
        if self.session and not self.session.closed:
            await self.session.close()
    
    async def check_health(self) -> bool:
        """
        Check if inference server is healthy
        """
        try:
            session = await self._get_session()
            payload = {
                "prompt": "हल्लो",  # "hello" in Hindi
                "temperature": self.config.temperature,
                "max_tokens": 10,
                "top_p": self.config.top_p,
                "repetition_penalty": self.config.repetition_penalty
            }
            
            async with session.post(
                self.config.api_url,
                json=payload,
                timeout=aiohttp.ClientTimeout(total=10)
            ) as resp:
                return resp.status == 200
        
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return False
    
    def _validate_hindi_text(self, text: str) -> bool:
        """
        Validate if text contains Hindi characters or English numerals
        Hindi Unicode range: \u0900-\u097F
        """
        # Check for Hindi characters
        hindi_chars = any('\u0900' <= char <= '\u097F' for char in text)
        # English text also works (will be pronounced)
        return len(text) > 0
    
    def _apply_emotion_tags(self, text: str) -> str:
        """
        Validate emotion tags in text
        
        Args:
            text: Text with potential emotion tags
        
        Returns:
            Validated text
        """
        for tag in EMOTION_TAGS.keys():
            if tag in text:
                logger.debug(f"Found emotion tag: {tag}")
        return text
    
    def _split_long_text(self, text: str, max_chunk_size: int = LONG_FORM_BATCH_SIZE) -> list:
        """
        Split long text into manageable chunks for processing
        Splits at sentence boundaries for naturalness
        
        Args:
            text: Hindi text to split
            max_chunk_size: Maximum characters per chunk
        
        Returns:
            List of text chunks
        """
        if len(text) <= max_chunk_size:
            return [text]
        
        chunks = []
        current_chunk = ""
        
        # Split by sentences (Devanagari danda: \u0964, period: .)
        sentences = text.replace('\u0964', '.').split('.')
        
        for sentence in sentences:
            if len(current_chunk) + len(sentence) <= max_chunk_size:
                current_chunk += sentence + "."
            else:
                if current_chunk:
                    chunks.append(current_chunk)
                current_chunk = sentence + "."
        
        if current_chunk:
            chunks.append(current_chunk)
        
        logger.info(f"Split long text ({len(text)} chars) into {len(chunks)} chunks")
        return chunks
    
    async def _generate_tokens(
        self,
        text: str,
        voice: str = "ऋतिका"
    ) -> str:
        """
        Generate audio tokens from text using inference server
        
        Args:
            text: Hindi text for synthesis
            voice: Voice name
        
        Returns:
            Token string from inference API
        """
        # Format prompt according to Orpheus convention
        prompt = HINDI_PROMPT_TEMPLATE.format(voice=voice, text=text)
        
        payload = {
            "prompt": prompt,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
            "top_p": self.config.top_p,
            "repetition_penalty": self.config.repetition_penalty,
            "stream": False
        }
        
        try:
            session = await self._get_session()
            async with session.post(
                self.config.api_url,
                json=payload
            ) as resp:
                if resp.status != 200:
                    error_text = await resp.text()
                    raise Exception(f"API error {resp.status}: {error_text}")
                
                data = await resp.json()
                
                # Extract tokens from response
                if "choices" in data and len(data["choices"]) > 0:
                    tokens = data["choices"][0].get("text", "")
                    logger.debug(f"Generated {len(tokens)} tokens")
                    return tokens
                else:
                    raise Exception("No choices in API response")
        
        except asyncio.TimeoutError:
            logger.error("Token generation timeout")
            raise Exception("Inference server timeout")
        except Exception as e:
            logger.error(f"Token generation failed: {e}")
            raise
    
    async def synthesize(
        self,
        text: str,
        voice: str = "ऋतिका",
        speed: float = 1.0,
        stream: bool = False
    ) -> Union[bytes, AsyncGenerator]:
        """
        Main synthesis function - converts Hindi text to audio
        
        Args:
            text: Hindi text to synthesize
            voice: Voice name (default: ऋतिका - Ritika)
            speed: Speech speed multiplier (0.5-1.5)
            stream: If True, return async generator for streaming
        
        Returns:
            WAV audio bytes or async generator for streaming
        """
        start_time = time.time()
        self.metrics["total_requests"] += 1
        
        try:
            # Validate input
            if not text or len(text) > 5000:
                raise ValueError("Text must be 1-5000 characters")
            
            if not self._validate_hindi_text(text):
                raise ValueError("Text must contain valid characters")
            
            # Apply emotion tags
            text = self._apply_emotion_tags(text)
            
            logger.info(f"Synthesizing: {len(text)} chars, voice={voice}, speed={speed}")
            
            # Handle long-form audio with batching
            if len(text) > LONG_FORM_THRESHOLD:
                return await self._synthesize_long_form(text, voice, speed, stream)
            else:
                return await self._synthesize_short_form(text, voice, speed, stream)
        
        except Exception as e:
            self.metrics["failed_requests"] += 1
            logger.error(f"Synthesis failed: {e}")
            raise
        finally:
            latency_ms = (time.time() - start_time) * 1000
            self.metrics["total_latency_ms"] += latency_ms
            logger.info(f"Synthesis latency: {latency_ms:.2f}ms")
    
    async def _synthesize_short_form(
        self,
        text: str,
        voice: str,
        speed: float,
        stream: bool
    ) -> Union[bytes, AsyncGenerator]:
        """
        Synthesize short text (<1000 chars) in single pass
        """
        tokens = await self._generate_tokens(text, voice)
        audio_bytes = self.audio_converter.convert_tokens_to_audio(tokens, speed)
        
        self.metrics["successful_requests"] += 1
        
        if stream:
            async def stream_gen():
                chunk_size = self.config.chunk_size
                for i in range(0, len(audio_bytes), chunk_size):
                    yield audio_bytes[i:i + chunk_size]
                    await asyncio.sleep(0.01)  # Small delay for streaming
            
            return stream_gen()
        else:
            return audio_bytes
    
    async def _synthesize_long_form(
        self,
        text: str,
        voice: str,
        speed: float,
        stream: bool
    ) -> Union[bytes, AsyncGenerator]:
        """
        Synthesize long text (>1000 chars) with intelligent batching
        """
        chunks = self._split_long_text(text)
        audio_chunks = []
        
        for i, chunk in enumerate(chunks):
            logger.info(f"Processing chunk {i+1}/{len(chunks)}")
            tokens = await self._generate_tokens(chunk, voice)
            audio = self.audio_converter.convert_tokens_to_audio(tokens, speed)
            audio_chunks.append(audio)
            await asyncio.sleep(0.1)  # Small delay between chunks
        
        # Stitch chunks with crossfade
        combined_audio = self.audio_converter.stitch_audio_chunks(
            audio_chunks,
            crossfade_ms=CROSSFADE_MS
        )
        
        self.metrics["successful_requests"] += 1
        
        if stream:
            async def stream_gen():
                chunk_size = self.config.chunk_size
                for i in range(0, len(combined_audio), chunk_size):
                    yield combined_audio[i:i + chunk_size]
                    await asyncio.sleep(0.01)
            
            return stream_gen()
        else:
            return combined_audio
    
    def get_metrics(self) -> dict:
        """
        Get performance metrics
        """
        avg_latency = (
            self.metrics["total_latency_ms"] / self.metrics["total_requests"]
            if self.metrics["total_requests"] > 0 else 0
        )
        
        return {
            **self.metrics,
            "average_latency_ms": avg_latency,
            "success_rate": (
                self.metrics["successful_requests"] / self.metrics["total_requests"]
                if self.metrics["total_requests"] > 0 else 0
            ) * 100
        }
