# GenAI Integration Guide - FREE Groq API

## Step 1: Get Free API Key (2 minutes)

1. Go to: https://console.groq.com
2. Sign up (free)
3. Go to API Keys section
4. Create new API key
5. Copy the key (starts with `gsk_...`)

**Free tier gives you:**
- 14,400 requests per day
- 30 requests per minute
- More than enough for your demo!

## Step 2: Set Environment Variable

### On Linux/Mac:
```bash
export GROQ_API_KEY="gsk_your_key_here"
```

### On Windows:
```cmd
set GROQ_API_KEY=gsk_your_key_here
```

### In Python (for testing):
```python
import os
os.environ["GROQ_API_KEY"] = "gsk_your_key_here"
```

## Step 3: Install Dependencies

```bash
pip install requests --break-system-packages
```

(You already have numpy, torch, flask)

## Step 4: Integrate into Your Backend

### Option A: Add to existing app.py

Add these lines to `/backend/app.py`:

```python
# At the top, after other imports:
from genai_enhanced_backend import genai_bp

# After creating Flask app:
app.register_blueprint(genai_bp)
```

### Option B: Run as separate service

```bash
cd backend
python genai_enhanced_backend.py
```

## Step 5: Update Frontend API Calls

Your frontend already calls `/genai/generate`. Update the endpoint to:
- `/api/genai/discover` - Find vulnerabilities
- `/api/genai/generate-attack` - Generate AI attacks
- `/api/genai/explain-result` - Explain detections

## Step 6: Test It

```bash
# Test vulnerability discovery
curl -X POST http://localhost:5000/api/genai/discover \
  -H "Content-Type: application/json" \
  -d '{}'

# Test attack generation
curl -X POST http://localhost:5000/api/genai/generate-attack \
  -H "Content-Type: application/json" \
  -d '{"attack_type": "genai_attack_1", "target_stage": "stage3"}'
```

## What This Gives You

### Before (Current System):
```python
# Random noise pretending to be AI
z = torch.randn_like(z) * perturbation_scale
```

### After (Real GenAI):
```python
# LLM analyzes physics, finds weaknesses, generates targeted attacks
vulnerabilities = llm.analyze_system_architecture(sensors, process)
attack = llm.generate_attack_vector(vulnerability, sensor_ranges)
```

## Live Demo Script

1. **Show vulnerability discovery:**
   - Click "Discover Gaps" button
   - LLM analyzes the entire water treatment process
   - Shows 5 specific vulnerabilities with explanations

2. **Generate AI-powered attack:**
   - Select a vulnerability
   - LLM generates attack that exploits it
   - Shows reasoning for why it should evade detection

3. **Explain results:**
   - Run the attack
   - LLM explains why it was/wasn't detected
   - Suggests improvements

## Presentation Talking Points

**OLD SYSTEM:**
"We used random noise generation, which doesn't actually use AI intelligence"

**NEW SYSTEM:**
"Our GenAI system uses Llama 3.1 to:
- Understand water treatment physics
- Reason about vulnerabilities
- Generate intelligent attacks that exploit specific weaknesses
- Explain detection failures
- This is TRUE generative AI, not random numbers"

## Fallback Mode

If Groq API is down or rate limited:
- System automatically falls back to physics-based generation
- Still smarter than random noise
- Uses process stage knowledge

## Cost

**FREE** - Groq gives you:
- 14,400 requests/day free tier
- Your entire demo needs ~50 requests
- Can run the demo 288 times per day for free

## Alternative Free APIs (if Groq doesn't work)

1. **Together.ai**: $25 free credit
2. **Hugging Face**: Free inference API
3. **OpenRouter**: Multiple free models

See `free_llm_options.md` for details.

## Troubleshooting

**Error: No GROQ_API_KEY**
- Set environment variable: `export GROQ_API_KEY="your_key"`

**Error: Rate limit**
- Free tier: 30 req/min
- Wait 1 minute or add `time.sleep(2)` between calls

**Error: JSON parse**
- LLM sometimes adds markdown
- Code handles this automatically

**Want to test locally without API?**
- Install Ollama: `curl -fsSL https://ollama.com/install.sh | sh`
- Run: `ollama run llama3.2`
- Update code to use Ollama endpoint

## Files You Need

1. `genai_vulnerability_scanner.py` - Core LLM integration
2. `genai_enhanced_backend.py` - Flask routes
3. Your existing `app.py` - Add blueprint registration

## Time to Integrate

- Get API key: 2 minutes
- Install: 1 minute  
- Integrate: 5 minutes
- Test: 2 minutes

**Total: 10 minutes to go from fake AI to real AI**
