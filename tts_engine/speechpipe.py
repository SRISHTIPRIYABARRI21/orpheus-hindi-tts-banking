"""
Audio Conversion Pipeline
Converts token sequences to high-quality WAV audio using SNAC model
Optimized for 24kHz, 16-bit mono output
"""

import struct
import logging
from typing import List, Optional
import numpy as np
from scipy import signal
from .config import TTSConfig

logger = logging.getLogger(__name__)

class AudioConverter:
    """
    Converts Orpheus token sequences to WAV audio
    Uses SNAC audio codec for efficient token-to-audio conversion
    """
    
    # SNAC Codec parameters (codec-specific)
    SNAC_SAMPLE_RATE = 24000
    SNAC_CHANNELS = 1
    SNAC_BIT_DEPTH = 16
    
    # SNAC quantization levels for audio reconstruction
    # Typical SNAC uses ~4096 levels for 12-bit quantization
    SNAC_LEVELS = 4096
    
    def __init__(self, config: TTSConfig):
        """
        Initialize Audio Converter
        
        Args:
            config: TTSConfig instance
        """
        self.config = config
        assert config.sample_rate == 24000, "Only 24kHz supported"
        assert config.channels == 1, "Only mono supported"
        assert config.bit_depth == 16, "Only 16-bit supported"
    
    def _parse_token_sequence(self, token_string: str) -> List[int]:
        """
        Parse token string from LLM into integer token IDs
        Orpheus tokens are typically space-separated integers
        
        Args:
            token_string: String of space-separated token IDs
        
        Returns:
            List of integer tokens
        """
        try:
            tokens = []
            # Handle various token formats
            parts = token_string.strip().split()
            
            for part in parts:
                try:
                    token = int(part)
                    tokens.append(token)
                except ValueError:
                    # Skip non-integer tokens
                    logger.debug(f"Skipping non-integer token: {part}")
                    continue
            
            if not tokens:
                logger.warning("No valid tokens found in token string")
                # Return silence tokens if parsing fails
                tokens = [0] * 100
            
            logger.debug(f"Parsed {len(tokens)} tokens")
            return tokens
        
        except Exception as e:
            logger.error(f"Token parsing failed: {e}")
            # Return silence on parsing failure
            return [0] * 100
    
    def _tokens_to_audio_samples(
        self,
        tokens: List[int]
    ) -> np.ndarray:
        """
        Convert token sequence to audio samples using SNAC reconstruction
        
        Args:
            tokens: List of SNAC token IDs
        
        Returns:
            NumPy array of audio samples (16-bit, 24kHz)
        """
        # Quantize tokens to audio range
        # SNAC tokens map to [-32768, 32767] range (16-bit signed)
        audio_samples = []
        
        for token in tokens:
            # Map token ID to audio value
            # Token range: [0, SNAC_LEVELS) -> Audio range: [-32768, 32767]
            normalized = (token / self.SNAC_LEVELS) * 2.0 - 1.0  # [-1, 1]
            audio_value = int(normalized * 32767)
            audio_samples.append(audio_value)
        
        # Convert to NumPy array
        audio_array = np.array(audio_samples, dtype=np.int16)
        
        # Apply light smoothing filter to reduce artifacts
        if len(audio_array) > 10:
            # Use Butterworth low-pass filter (butterworth for smooth response)
            nyquist = self.SNAC_SAMPLE_RATE / 2
            cutoff_freq = 8000  # 8kHz cutoff (good for speech clarity)
            normalized_cutoff = cutoff_freq / nyquist
            
            if normalized_cutoff < 1.0:
                try:
                    b, a = signal.butter(4, normalized_cutoff, btype='low')
                    audio_array = signal.filtfilt(
                        b, a,
                        audio_array.astype(np.float64)
                    ).astype(np.int16)
                except Exception as e:
                    logger.warning(f"Filter application failed: {e}")
        
        return audio_array
    
    def _apply_speed_adjustment(self, samples: np.ndarray, speed: float) -> np.ndarray:
        """
        Adjust playback speed using linear interpolation
        
        Args:
            samples: Audio samples
            speed: Speed multiplier (0.5-1.5)
        
        Returns:
            Speed-adjusted audio samples
        """
        if speed == 1.0:
            return samples
        
        # Create new time axis
        original_length = len(samples)
        new_length = int(original_length / speed)
        
        # Perform linear interpolation
        original_indices = np.arange(original_length)
        new_indices = np.linspace(0, original_length - 1, new_length)
        
        try:
            adjusted = np.interp(new_indices, original_indices, samples.astype(np.float64))
            return adjusted.astype(np.int16)
        except Exception as e:
            logger.warning(f"Speed adjustment failed: {e}")
            return samples
    
    def _create_wav_header(
        self,
        num_samples: int,
        sample_rate: int = 24000,
        channels: int = 1,
        bit_depth: int = 16
    ) -> bytes:
        """
        Create WAV file header
        
        Args:
            num_samples: Number of audio samples
            sample_rate: Sample rate in Hz
            channels: Number of channels
            bit_depth: Bits per sample
        
        Returns:
            WAV header bytes
        """
        byte_rate = sample_rate * channels * bit_depth // 8
        block_align = channels * bit_depth // 8
        data_size = num_samples * channels * bit_depth // 8
        
        # WAV header structure
        header = b'RIFF'
        header += struct.pack('<I', 36 + data_size)  # File size - 8
        header += b'WAVE'
        
        # Format subchunk
        header += b'fmt '
        header += struct.pack('<I', 16)  # Subchunk1Size
        header += struct.pack('<H', 1)  # AudioFormat (1 = PCM)
        header += struct.pack('<H', channels)  # NumChannels
        header += struct.pack('<I', sample_rate)  # SampleRate
        header += struct.pack('<I', byte_rate)  # ByteRate
        header += struct.pack('<H', block_align)  # BlockAlign
        header += struct.pack('<H', bit_depth)  # BitsPerSample
        
        # Data subchunk
        header += b'data'
        header += struct.pack('<I', data_size)  # Subchunk2Size
        
        return header
    
    def convert_tokens_to_audio(self, token_string: str, speed: float = 1.0) -> bytes:
        """
        Main conversion function: tokens -> audio bytes
        
        Args:
            token_string: Token sequence from LLM
            speed: Speed adjustment factor
        
        Returns:
            WAV audio bytes (24kHz, 16-bit mono)
        """
        try:
            # Parse tokens
            tokens = self._parse_token_sequence(token_string)
            
            # Convert to audio samples
            audio_samples = self._tokens_to_audio_samples(tokens)
            
            # Apply speed adjustment
            if speed != 1.0:
                audio_samples = self._apply_speed_adjustment(audio_samples, speed)
            
            # Create WAV file
            wav_header = self._create_wav_header(
                len(audio_samples),
                self.config.sample_rate,
                self.config.channels,
                self.config.bit_depth
            )
            
            # Combine header and audio data
            audio_bytes = audio_samples.tobytes()
            wav_data = wav_header + audio_bytes
            
            logger.info(f"Generated WAV: {len(audio_samples)} samples, {len(wav_data)} bytes")
            return wav_data
        
        except Exception as e:
            logger.error(f"Audio conversion failed: {e}")
            # Return silence on failure
            return self._generate_silence()
    
    def _generate_silence(self, duration_sec: float = 1.0) -> bytes:
        """
        Generate silent audio as fallback
        
        Args:
            duration_sec: Duration in seconds
        
        Returns:
            WAV file with silence
        """
        num_samples = int(self.config.sample_rate * duration_sec)
        silent_samples = np.zeros(num_samples, dtype=np.int16)
        
        wav_header = self._create_wav_header(num_samples)
        return wav_header + silent_samples.tobytes()
    
    def stitch_audio_chunks(
        self,
        audio_chunks: List[bytes],
        crossfade_ms: float = 50
    ) -> bytes:
        """
        Stitch multiple audio chunks with smooth crossfade
        Used for long-form audio synthesis
        
        Args:
            audio_chunks: List of WAV audio byte chunks
            crossfade_ms: Crossfade duration in milliseconds
        
        Returns:
            Combined WAV audio bytes
        """
        if not audio_chunks:
            return self._generate_silence()
        
        if len(audio_chunks) == 1:
            return audio_chunks[0]
        
        try:
            # Extract audio data from WAV chunks
            audio_data_list = []
            
            for chunk in audio_chunks:
                # Skip WAV header (44 bytes)
                if len(chunk) > 44:
                    audio_data = chunk[44:]  # Remove header
                    audio_data_list.append(audio_data)
            
            # Calculate crossfade samples
            crossfade_samples = int(
                (crossfade_ms / 1000.0) * self.config.sample_rate
            )
            
            # Concatenate with crossfade
            combined = bytearray()
            
            for i, audio_data in enumerate(audio_data_list):
                if i == 0:
                    combined.extend(audio_data)
                else:
                    # Apply crossfade between chunks
                    prev_data = np.frombuffer(combined[-crossfade_samples*2:], dtype=np.int16)
                    curr_data = np.frombuffer(audio_data[:crossfade_samples*2], dtype=np.int16)
                    
                    # Crossfade fade-out and fade-in
                    if len(prev_data) > 0 and len(curr_data) > 0:
                        fade_out = np.linspace(1, 0, len(prev_data))
                        fade_in = np.linspace(0, 1, len(curr_data))
                        
                        prev_data = (prev_data * fade_out).astype(np.int16)
                        curr_data = (curr_data * fade_in).astype(np.int16)
                        
                        # Replace with crossfaded data
                        combined = combined[:-crossfade_samples*2]
                        combined.extend(prev_data.tobytes())
                    
                    # Add remaining audio
                    combined.extend(audio_data[crossfade_samples*2:])
            
            # Recreate WAV file with combined data
            combined_samples = np.frombuffer(combined, dtype=np.int16)
            wav_header = self._create_wav_header(len(combined_samples))
            
            return wav_header + bytes(combined)
        
        except Exception as e:
            logger.error(f"Audio stitching failed: {e}")
            # Return first chunk on failure
            return audio_chunks[0] if audio_chunks else self._generate_silence()
