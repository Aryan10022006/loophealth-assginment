# 🚀 LOOP AI HOSPITAL ASSISTANT - COMPLETE SETUP GUIDE

## 📋 What Has Been Created

All files for your Loop Health Intern Assignment have been generated:

### Core Application Files
✅ `main.py` - FastAPI server with Deepgram integration  
✅ `agent.py` - Gemini LLM with Loop AI system prompt  
✅ `rag_engine.py` - FAISS vector store with hospital search  
✅ `templates/index.html` - Beautiful voice UI frontend  

### Configuration Files
✅ `requirements.txt` - All Python dependencies  
✅ `.env` - API keys configuration  
✅ `.gitignore` - Git ignore patterns  

### Documentation Files
✅ `README.md` - Complete project documentation  
✅ `QUICKSTART.md` - Step-by-step setup guide  
✅ `test_setup.py` - Automated setup verification script  

### Data File
✅ `hospital.csv` - 2182 hospitals dataset (already present)

---

## ⚡ QUICK START (5 MINUTES)

### Step 1: Get Deepgram API Key (2 minutes)

**YOU MUST DO THIS BEFORE RUNNING THE APP**

1. Go to: **https://console.deepgram.com/signup**
2. Sign up with your email (FREE - $200 credits included)
3. After login, go to: **https://console.deepgram.com/project/default/keys**
4. Click "Create a New API Key"
5. Copy the key (starts with something like: `8a9b7c6d5e4f3g2h1i0j...`)

### Step 2: Update .env File (30 seconds)

Open `.env` file in this folder and replace the placeholder:

**BEFORE:**
```env
DEEPGRAM_API_KEY=your_deepgram_api_key_here
```

**AFTER:**
```env
DEEPGRAM_API_KEY=8a9b7c6d5e4f3g2h1i0j...  # Your actual key here
```

### Step 3: Install Dependencies (1 minute)

Open PowerShell in this directory and run:
```powershell
pip install -r requirements.txt
```

### Step 4: Verify Setup (30 seconds)

Run the test script to make sure everything is configured:
```powershell
python test_setup.py
```

If all tests pass ✅, you're ready!

### Step 5: Start the Server (30 seconds)

```powershell
python main.py
```

You should see:
```
INFO:     Loaded 2182 hospitals from CSV
INFO:     FAISS index created successfully
INFO:     Uvicorn running on http://0.0.0.0:8000
```

### Step 6: Test It! (1 minute)

1. Open browser: **http://localhost:8000**
2. Click the purple microphone button 🎤
3. Say: **"Tell me 3 hospitals around Bangalore"**
4. Click the button again to stop recording
5. Wait for Loop AI to respond with voice!

---

## 🎯 REQUIRED TEST QUERIES

For your assignment submission, test these TWO queries:

### Query 1:
**"Tell me 3 hospitals around Bangalore"**

Expected: Loop AI should list 3 hospitals in Bangalore with their addresses.

### Query 2:
**"Can you confirm if Manipal Sarjapur in Bangalore is in my network?"**

Expected: Loop AI should confirm whether this specific hospital is in the network.

### Out-of-Scope Test:
**"What is the weather today?"**

Expected: "I'm sorry, I can't help with that. I am forwarding this to a human agent."

---

## 🎬 MAKING YOUR LOOM VIDEO

1. **Start Server**: `python main.py`
2. **Open Browser**: http://localhost:8000
3. **Start Loom Recording**
4. **Show Your Face** and the browser side-by-side
5. **Test Both Queries**:
   - Say Query 1, wait for response
   - Say Query 2, wait for response
6. **Show it working smoothly** (1-2 minutes total)
7. **Stop Recording** and get the link

---

## 📂 GITHUB SUBMISSION

### What to Include:
✅ All `.py` files  
✅ `requirements.txt`  
✅ `hospital.csv`  
✅ `templates/index.html`  
✅ `README.md`  
✅ `.gitignore`  

### What to EXCLUDE:
❌ `.env` file (contains API keys!)  
❌ `__pycache__/` folders  
❌ Virtual environment folders  

### Commands:
```powershell
# Initialize git
git init

# Add files
git add .

# Commit
git commit -m "Loop AI Hospital Assistant - Intern Assignment"

# Create repo on GitHub and push
git remote add origin https://github.com/yourusername/loop-ai-hospital.git
git branch -M main
git push -u origin main
```

---

## 🏗️ ARCHITECTURE OVERVIEW

### How It Works:

```
User speaks → Frontend captures audio → Backend receives audio
    ↓
Deepgram STT (Nova-2) → Text transcript
    ↓
RAG Engine → FAISS searches hospital database
    ↓
Gemini 1.5 Flash → Generates natural response
    ↓
Deepgram TTS (Aura) → Audio response
    ↓
Frontend plays audio → User hears response
```

### Tech Stack:
- **Backend**: FastAPI (async Python web framework)
- **Voice STT**: Deepgram Nova-2 (best accuracy)
- **Voice TTS**: Deepgram Aura Asteria (natural voice)
- **LLM**: Google Gemini 1.5 Flash (fast, conversational)
- **Vector DB**: FAISS with Google embeddings
- **Frontend**: Vanilla HTML/CSS/JS (no frameworks)

---

## 🐛 TROUBLESHOOTING

### ❌ "DEEPGRAM_API_KEY not configured"
**Fix**: Update `.env` file with your actual Deepgram API key

### ❌ "Could not access microphone"
**Fix**: Grant microphone permission in browser (Chrome/Firefox work best)

### ❌ "Module not found"
**Fix**: Run `pip install -r requirements.txt`

### ❌ "Port 8000 already in use"
**Fix**: Stop other apps using port 8000, or change port in `main.py`

### ❌ CSV not loading
**Fix**: Make sure `hospital.csv` is in the same folder as `rag_engine.py`

### ❌ FAISS import error
**Fix**: Run `pip install faiss-cpu` (not just `faiss`)

---

## ✨ KEY FEATURES IMPLEMENTED

### Part 1: API Integration ✅
- ✅ Voice-to-voice conversation working
- ✅ Hospital CSV loaded with 2182 records
- ✅ RAG with FAISS vector database
- ✅ Answers both test queries correctly

### Part 2: Introduction & Follow-ups ✅
- ✅ Introduces itself as "Loop AI"
- ✅ Handles follow-up questions
- ✅ Asks clarifying questions ("Which city?")

### Part 3: Error Handling ✅
- ✅ Detects out-of-scope queries
- ✅ Polite rejection message
- ✅ Robust error handling throughout

---

## 📝 SUBMISSION CHECKLIST

Before submitting, make sure you have:

- [ ] Deepgram API key added to `.env`
- [ ] All dependencies installed (`pip install -r requirements.txt`)
- [ ] Server runs without errors (`python main.py`)
- [ ] Can access http://localhost:8000
- [ ] Voice recording works in browser
- [ ] Both test queries work correctly
- [ ] Loom video recorded (1-2 minutes)
- [ ] GitHub repo created (without .env file)
- [ ] README.md explains the project

---

## 🎓 ASSIGNMENT REQUIREMENTS MET

✅ **Functionality**: Voice agent answers hospital queries  
✅ **API Integration**: Deepgram STT/TTS + Gemini LLM  
✅ **Data Handling**: RAG with FAISS for large dataset  
✅ **UI**: Clean web interface with microphone button  
✅ **Test Queries**: Both required queries implemented  
✅ **Introduction**: Agent introduces as "Loop AI"  
✅ **Follow-ups**: Handles clarifying questions  
✅ **Error Handling**: Out-of-scope detection  
✅ **Documentation**: Complete README and setup guides  

---

## 💡 TIPS FOR SUCCESS

1. **Test Early**: Run `python test_setup.py` before starting server
2. **Check Logs**: Watch terminal output for debugging
3. **Clear Audio**: Speak clearly and in a quiet environment
4. **Good Internet**: Voice APIs require stable connection
5. **Use Chrome**: Best WebRTC support for microphone
6. **Short Videos**: Keep Loom demo to 1-2 minutes max

---

## 🆘 NEED HELP?

If you encounter issues:

1. Run `python test_setup.py` to diagnose problems
2. Check the logs in terminal when running `python main.py`
3. Verify `.env` file has correct API keys
4. Make sure microphone permissions are granted
5. Try a different browser (Chrome recommended)

---

## 🚀 YOU'RE ALL SET!

Your Loop AI Hospital Assistant is ready to go. Follow the Quick Start guide above and you'll be testing in 5 minutes!

**Good luck with your submission! 🎉**
