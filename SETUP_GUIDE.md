# Complete Setup Guide - Orpheus Hindi TTS

## Prerequisites Check

Before starting, ensure you have:

```bash
# Check Docker
docker --version
docker compose --version

# Check NVIDIA GPU
nvidia-smi

# Check NVIDIA Container Toolkit
docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi
```

## Option 1: Docker Compose (Recommended for Production)

### Step 1: Clone Repository

```bash
git clone https://github.com/SRISHTIPRIYABARRI21/orpheus-hindi-tts-banking.git
cd orpheus-hindi-tts-banking
```

### Step 2: Configure Environment

```bash
cp .env.example .env

# Edit .env if needed (for custom GPU, ports, etc.)
# Default settings are optimized for RTX 4090/A100/H100
```

### Step 3: Download Model

```bash
# Download the Hindi TTS model (one-time, ~3.5GB)
mkdir -p models
docker compose -f docker-compose-gpu.yml --profile init run model-init
```

**First time only!** This downloads the 3.52GB model from HuggingFace.

### Step 4: Start Services

```bash
# Start both inference server and FastAPI (will stay in foreground)
docker compose -f docker-compose-gpu.yml up

# OR start in background
docker compose -f docker-compose-gpu.yml up -d
```

### Step 5: Verify Running

```bash
# Health check
curl http://localhost:5005/health

# View logs
docker compose -f docker-compose-gpu.yml logs -f orpheus-fastapi

# Check services
docker ps | grep orpheus
```

### Step 6: Test the API

```bash
# Simple test
curl -X POST http://localhost:5005/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "model": "orpheus-hindi",
    "input": "नमस्ते",
    "voice": "ऋतिका"
  }' \
  --output test.wav

ffplay test.wav  # Play with FFmpeg
```

## Option 2: Native Installation (Without Docker)

### Step 1: Python Setup

```bash
# Use Python 3.8-3.11 (NOT 3.12)
python --version

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/macOS
# or
venv\Scripts\activate  # Windows
```

### Step 2: Install Dependencies

```bash
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

### Step 3: Install PyTorch with CUDA

```bash
# For CUDA 12.4 (recommended)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# For CPU only (slow, for development only)
pip install torch torchvision torchaudio
```

### Step 4: Download Model

```bash
mkdir -p models
huggingface-cli download lex-au Orpheus-3b-Hindi-FT-Q8_0.gguf --local-dir ./models
```

### Step 5: Start Inference Server

In **Terminal 1**, start llama.cpp server:

```bash
# Download llama.cpp if not present
wget https://github.com/ggerganov/llama.cpp/releases/download/b3869/llama-server-b3869-bin-ubuntu-x64.zip
unzip llama-server-b3869-bin-ubuntu-x64.zip

# Start server
./llama-server \
  -m models/Orpheus-3b-Hindi-FT-Q8_0.gguf \
  --ctx-size 8192 \
  --n-predict 8192 \
  --rope-scaling linear \
  --n-gpu-layers 99 \
  --threads 8 \
  --parallel 4 \
  -cb \
  --port 5006
```

### Step 6: Start FastAPI Server

In **Terminal 2**, start the FastAPI app:

```bash
cd orpheus-hindi-tts-banking
source venv/bin/activate

python app.py
```

Server will start on `http://localhost:5005`

### Step 7: Test

```bash
# In Terminal 3, test the API
curl http://localhost:5005/health
```

## Configuration

### Performance Tuning

Edit `.env` for your hardware:

```bash
# For RTX 4090 (24GB VRAM)
ORPHEUS_MAX_TOKENS=8192
ORPHEUS_TEMPERATURE=0.6
ORPHEUS_TOP_P=0.9

# For RTX 4080 (16GB VRAM)
ORPHEUS_MAX_TOKENS=4096

# For RTX 3090 (24GB VRAM)
ORPHEUS_MAX_TOKENS=6144

# For T4 (16GB VRAM)
ORPHEUS_MAX_TOKENS=2048
ORPHEUS_TEMPERATURE=0.5  # More deterministic for low VRAM
```

### Latency Optimization

```bash
# For <200ms latency (requires high-end GPU)
ORPHEUS_MAX_TOKENS=4096
ORPHEUS_TEMPERATURE=0.3  # Lower temp = faster

# For <300ms latency (standard)
ORPHEUS_MAX_TOKENS=8192
ORPHEUS_TEMPERATURE=0.6
```

## Common Issues

### Issue: "Connection refused" on port 5005

**Solution:**
```bash
# Check if service is running
curl http://localhost:5005/health

# If not running, check logs
docker compose logs orpheus-fastapi

# Restart service
docker compose down
docker compose up -d
```

### Issue: "CUDA out of memory"

**Solution:**
```bash
# Reduce max tokens in .env
ORPHEUS_MAX_TOKENS=2048

# Reduce batch size
ORPHEUS_TEMPERATURE=0.4

# Restart
docker compose restart
```

### Issue: "Failed to connect to inference server"

**Solution:**
```bash
# Check inference server health
curl http://localhost:5006/health

# If not running, check logs
docker compose logs llama-cpp-server

# Ensure ORPHEUS_API_URL is correct in .env
echo $ORPHEUS_API_URL
```

### Issue: "Model not found" / Download fails

**Solution:**
```bash
# Manual download
mkdir -p models
cd models

# Using git LFS
git lfs install
git clone https://huggingface.co/lex-au/Orpheus-3b-Hindi-FT-Q8_0.gguf

# Or using huggingface-cli
huggingface-cli download lex-au Orpheus-3b-Hindi-FT-Q8_0.gguf

# Verify file
ls -lh Orpheus-3b-Hindi-FT-Q8_0.gguf  # Should be ~3.5GB
```

## Testing

### Health Check

```bash
curl -X GET http://localhost:5005/health | jq
```

Expected response:
```json
{
  "status": "healthy",
  "model": "Orpheus-3b-Hindi-FT-Q8_0.gguf",
  "language": "Hindi",
  "voice": "ऋतिका (Ritika)"
}
```

### Simple Audio Generation

```bash
curl -X POST http://localhost:5005/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "input": "परीक्षा सयल है।",
    "voice": "ऋतिका",
    "speed": 1.0
  }' \
  --output test.wav

ffplay test.wav  # or
play test.wav    # SoX
vlc test.wav     # VLC
```

### Batch Testing

```bash
python examples/basic_usage.py
```

## Running Python Examples

```bash
# Install example dependencies
pip install requests aiohttp

# Run examples
python examples/basic_usage.py
```

## Performance Monitoring

### GPU Monitoring

```bash
# Real-time GPU stats
watch -n 0.5 nvidia-smi

# Or continuous logging
nvidia-smi dmon -s pcm > gpu_stats.txt
```

### API Response Time

```bash
# Measure latency with curl
time curl -X POST http://localhost:5005/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"input": "हल्लो", "voice": "ऋतिका"}' \
  --output /dev/null
```

## Production Checklist

- [ ] Model downloaded and verified (3.52 GB)
- [ ] Docker Compose working without errors
- [ ] Health check passing (`/health` endpoint)
- [ ] Sample audio generated successfully
- [ ] Latency measured and acceptable (<300ms)
- [ ] GPU memory stable during continuous generation
- [ ] Environment variables optimized for your GPU
- [ ] Monitoring setup (nvidia-smi, logs)
- [ ] Error handling tested
- [ ] Load tested with concurrent requests

## Next Steps

1. **Integration**: Connect to your banking/finance application
2. **Scaling**: Use Docker Compose replicas for multiple instances
3. **Monitoring**: Setup Prometheus/Grafana metrics
4. **Deployment**: Deploy to Kubernetes or cloud platform
5. **Testing**: Run load tests with `examples/load_test.py` (if available)

## Support

- 📖 Read [README.md](README.md) for complete documentation
- 🤗 Check [HuggingFace model card](https://huggingface.co/lex-au/Orpheus-3b-Hindi-FT-Q8_0.gguf)
- 🐛 Report issues on GitHub
