# Loop AI Hospital Assistant - Quick Start Guide

## Prerequisites
- Python 3.9 or higher installed
- Microphone access in your browser
- Internet connection

## Step-by-Step Setup

### Step 1: Install Python Dependencies
Open PowerShell in this directory and run:
```powershell
pip install -r requirements.txt
```

### Step 2: Get Your Deepgram API Key

**IMPORTANT**: You need to sign up for Deepgram to get a free API key.

1. Visit: https://console.deepgram.com/signup
2. Sign up for a free account (you get $200 in free credits)
3. After signing in, go to: https://console.deepgram.com/project/default/keys
4. Click "Create a New API Key"
5. Copy the API key

### Step 3: Update .env File

Open the `.env` file and replace `your_deepgram_api_key_here` with your actual Deepgram API key:

```env
GOOGLE_API_KEY=AIzaSyDRTLt6DaexbKZre6owW41-ggXusFa7zxg
DEEPGRAM_API_KEY=YOUR_ACTUAL_KEY_HERE
```

### Step 4: Run the Server

In PowerShell, run:
```powershell
python main.py
```

You should see:
```
INFO:     Started server process
INFO:     Uvicorn running on http://0.0.0.0:8000
```

### Step 5: Open the App

Open your browser and go to:
```
http://localhost:8000
```

### Step 6: Test the Voice Agent

1. Click the microphone button (it will turn red)
2. Say: "Tell me 3 hospitals around Bangalore"
3. Click the button again to stop recording
4. Wait for Loop AI to respond with audio

## Test Queries

Try these queries to test the system:

1. **Query 1 (Required Test)**:
   - "Tell me 3 hospitals around Bangalore"
   
2. **Query 2 (Required Test)**:
   - "Can you confirm if Manipal Sarjapur in Bangalore is in my network?"
   
3. **Additional Test Queries**:
   - "Show me hospitals in Delhi"
   - "Is Apollo Hospital available in Faridabad?"
   - "What hospitals do you have in Gurugram?"

4. **Out-of-Scope Test**:
   - "What is the weather today?" 
   - Should respond: "I'm sorry, I can't help with that. I am forwarding this to a human agent."

## Troubleshooting

### Error: "DEEPGRAM_API_KEY not configured"
- Make sure you've updated the `.env` file with your actual Deepgram API key

### Error: "Could not access microphone"
- Allow microphone permissions in your browser
- Try using Chrome or Firefox (better WebRTC support)

### Error: Module not found
- Run: `pip install -r requirements.txt`

### Error: Port 8000 already in use
- Stop any other processes using port 8000
- Or change the port in `main.py` (line: `uvicorn.run(app, host="0.0.0.0", port=8000)`)

## Making a Demo Video

For your Loom video submission:

1. Start the server (`python main.py`)
2. Open http://localhost:8000
3. Start recording with Loom
4. Show yourself asking the two required test questions:
   - "Tell me 3 hospitals around Bangalore"
   - "Can you confirm if Manipal Sarjapur in Bangalore is in my network?"
5. Demonstrate the voice interaction working
6. Show the responses from Loop AI

## Project Structure

```
LoopHealth/
├── hospital.csv          # 2182 hospitals dataset
├── requirements.txt      # Python dependencies
├── .env                  # API keys (keep secure!)
├── rag_engine.py        # FAISS vector store + search
├── agent.py             # Gemini LLM + system prompt
├── main.py              # FastAPI server + Deepgram integration
├── templates/
│   └── index.html       # Frontend UI
└── README.md            # Detailed documentation
```

## Technical Details

- **STT Model**: Deepgram Nova-2 (highest accuracy)
- **TTS Model**: Deepgram Aura Asteria (natural female voice)
- **LLM**: Google Gemini 1.5 Flash (fast responses)
- **Vector DB**: FAISS with Google embeddings
- **Server**: FastAPI with async support

## Next Steps

After testing locally:
1. Create a GitHub repository
2. Push all files (except `.env`)
3. Record your Loom demo video
4. Submit the GitHub link and Loom link

Good luck! 🚀
