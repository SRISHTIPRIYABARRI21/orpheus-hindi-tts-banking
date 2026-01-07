#!/usr/bin/env python3
"""
Basic usage example for Orpheus Hindi TTS
Demonstrates common banking/finance scenarios
"""

import requests
import asyncio
import aiohttp
from pathlib import Path

# API endpoint
API_URL = "http://localhost:5005"

def example_1_simple_speech():
    """Basic speech synthesis example"""
    print("\n=== Example 1: Simple Speech Synthesis ===")
    
    text = "नमस्ते! यह आपकी ईएमआई सुविधा संबंधी कॉल है।"
    
    payload = {
        "input": text,
        "model": "orpheus-hindi",
        "voice": "ऋतिका",
        "speed": 1.0
    }
    
    response = requests.post(
        f"{API_URL}/v1/audio/speech",
        json=payload
    )
    
    if response.status_code == 200:
        output_file = "output_simple.wav"
        with open(output_file, "wb") as f:
            f.write(response.content)
        print(f"✓ Audio saved to {output_file}")
    else:
        print(f"✗ Error: {response.status_code} - {response.text}")

def example_2_emotion_tags():
    """Example with emotion tags"""
    print("\n=== Example 2: Emotion Tags ===")
    
    text = "आपके खाते समारोह की सूचना देखें! <laugh> बस सब कुछ ठीक है। <sigh>"
    
    payload = {
        "input": text,
        "model": "orpheus-hindi",
        "voice": "ऋतिका"
    }
    
    response = requests.post(
        f"{API_URL}/v1/audio/speech",
        json=payload
    )
    
    if response.status_code == 200:
        output_file = "output_emotions.wav"
        with open(output_file, "wb") as f:
            f.write(response.content)
        print(f"✓ Audio with emotions saved to {output_file}")
    else:
        print(f"✗ Error: {response.status_code}")

def example_3_banking_scenario():
    """Banking scenario: EMI notification"""
    print("\n=== Example 3: Banking - EMI Notification ===")
    
    text = """
    सुकमगुप्त संध्या
    आपके वर्तमान ईऎम आई और कुल और बकी की समीक्षा के लैंघ नयी सूचना है
    आपकी सुविधाबी तथपरता सबके लने सुरक्षित है
    सेवा के लिये धन्यवाद
    """
    
    payload = {
        "input": text.strip(),
        "model": "orpheus-hindi",
        "voice": "ऋतिका",
        "speed": 0.9  # Slightly slower for clarity
    }
    
    response = requests.post(
        f"{API_URL}/v1/audio/speech",
        json=payload
    )
    
    if response.status_code == 200:
        output_file = "output_banking.wav"
        with open(output_file, "wb") as f:
            f.write(response.content)
        print(f"✓ Banking notification audio saved to {output_file}")
    else:
        print(f"✗ Error: {response.status_code}")

async def example_4_streaming():
    """Streaming example for real-time applications"""
    print("\n=== Example 4: Real-time Streaming ===")
    
    text = "इस समय, आपकी कॉल सही सपेरेंड खते में लग गई थी। कृपया आस समय की सुविधा लें।"
    
    payload = {
        "text": text,
        "voice": "ऋतिका"
    }
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{API_URL}/stream",
                json=payload
            ) as resp:
                if resp.status == 200:
                    # Stream chunks as they arrive
                    output_file = "output_stream.wav"
                    with open(output_file, "wb") as f:
                        async for chunk in resp.content.iter_chunked(1024):
                            f.write(chunk)
                    print(f"✓ Streaming audio saved to {output_file}")
                else:
                    print(f"✗ Error: {resp.status}")
    except Exception as e:
        print(f"✗ Streaming error: {e}")

def example_5_check_health():
    """Health check and system info"""
    print("\n=== Example 5: Health Check ===")
    
    try:
        response = requests.get(f"{API_URL}/health")
        if response.status_code == 200:
            data = response.json()
            print(f"✓ Server Status: {data.get('status')}")
            print(f"  Model: {data.get('model')}")
            print(f"  Language: {data.get('language')}")
            print(f"  Voice: {data.get('voice')}")
        else:
            print(f"✗ Health check failed: {response.status_code}")
    except Exception as e:
        print(f"✗ Connection error: {e}")

def example_6_get_voices():
    """Get available voices"""
    print("\n=== Example 6: Available Voices ===")
    
    try:
        response = requests.get(f"{API_URL}/voices")
        if response.status_code == 200:
            data = response.json()
            print(f"Language: {data.get('language')}")
            print(f"Model: {data.get('model')}")
            print("Available Voices:")
            for voice, desc in data.get('voices', {}).items():
                print(f"  - {voice}: {desc}")
        else:
            print(f"✗ Error: {response.status_code}")
    except Exception as e:
        print(f"✗ Error: {e}")

def main():
    """Run all examples"""
    print("""
    ╔════════════════════════════════════════════════╗
    ║   Orpheus Hindi TTS - Banking Examples        ║
    ║   Production-Grade Voice AI                   ║
    ╚════════════════════════════════════════════════╝
    """)
    
    # Synchronous examples
    example_5_check_health()
    example_6_get_voices()
    example_1_simple_speech()
    example_2_emotion_tags()
    example_3_banking_scenario()
    
    # Async example
    asyncio.run(example_4_streaming())
    
    print("\n=== All examples completed! ===")
    print("Output files saved to current directory (*.wav)")

if __name__ == "__main__":
    main()
