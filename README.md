# Loop AI Hospital Assistant 🏥

A Voice AI Agent for Hospital Network using FastAPI, Deepgram, and Google Gemini.

## Overview

Loop AI is a conversational voice assistant that helps users find hospitals in a network. It uses:
- **Speech-to-Text**: Deepgram Nova-2
- **LLM**: Google Gemini 1.5 Flash
- **Text-to-Speech**: Deepgram Aura
- **RAG**: FAISS vector database with hospital embeddings

## Features

✅ Voice-to-voice conversation  
✅ Real-time hospital search using RAG  
✅ Handles clarifying questions (e.g., "Which city?")  
✅ Out-of-scope query detection  
✅ Clean, modern web UI  
✅ Low latency responses  

## File Structure

```
LoopHealth/
├── hospital.csv           # Hospital dataset
├── requirements.txt       # Python dependencies
├── .env                   # API keys (DO NOT COMMIT)
├── rag_engine.py         # FAISS vector store and search
├── agent.py              # Gemini LLM logic and system prompt
├── main.py               # FastAPI server
└── templates/
    └── index.html        # Frontend UI
```

## Setup Instructions

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure API Keys

Edit the `.env` file and add your API keys:

```env
GOOGLE_API_KEY=AIzaSyDRTLt6DaexbKZre6owW41-ggXusFa7zxg
DEEPGRAM_API_KEY=your_deepgram_api_key_here
```

**Get your Deepgram API key:**
1. Go to [https://deepgram.com/](https://deepgram.com/)
2. Sign up for a free trial (includes $200 credits)
3. Navigate to API Keys section
4. Copy your API key and paste it in `.env`

### 3. Run the Server

```bash
python main.py
```

The server will start at `http://localhost:8000`

### 4. Open the Frontend

Open your browser and go to:
```
http://localhost:8000
```

## Usage

1. Click the microphone button
2. Speak your query (e.g., "Tell me 3 hospitals around Bangalore")
3. The agent will respond with audio

### Example Queries

- "Tell me 3 hospitals around Bangalore"
- "Can you confirm if Manipal Sarjapur in Bangalore is in my network?"
- "Show me hospitals in Delhi"
- "Is Apollo Hospital in Faridabad available?"

## How It Works

### Backend Flow (`main.py`)

1. **Receive Audio**: User records audio in the browser
2. **STT**: Deepgram converts speech to text (Nova-2 model)
3. **RAG Search**: Query is embedded and searched in FAISS vector store
4. **LLM Processing**: Gemini generates response based on retrieved context
5. **TTS**: Deepgram converts response text to speech (Aura voice)
6. **Return Audio**: Audio is streamed back to the frontend

### RAG Engine (`rag_engine.py`)

- Loads `hospital.csv` with pandas
- Cleans and normalizes column names
- Creates text embeddings using Google Generative AI Embeddings
- Builds FAISS index for fast similarity search
- Supports both semantic search and exact name matching

### Agent (`agent.py`)

- Uses Gemini 1.5 Flash for natural language understanding
- System prompt enforces hospital-only queries
- Detects out-of-scope queries and forwards to human agent
- Asks clarifying questions when needed
- Keeps responses concise for voice interaction

## Error Handling

- ✅ Missing CSV file → Creates dummy data
- ✅ Missing API keys → Returns clear error message
- ✅ STT/TTS failures → Catches and returns HTTP 500
- ✅ Out-of-scope queries → Polite rejection message

## Testing

Test the two required queries:

1. **Query 1**: "Tell me 3 hospitals around Bangalore"
   - Should return 3 hospitals from Bangalore with addresses

2. **Query 2**: "Can you confirm if Manipal Sarjapur in Bangalore is in my network?"
   - Should confirm if the hospital exists in the network

## Deployment Notes

- The app uses port 8000 by default
- CORS is enabled for all origins (adjust for production)
- Frontend is served at the root path `/`

## Technologies Used

- **Backend**: FastAPI, Python 3.9+
- **Voice**: Deepgram API (Nova-2 STT, Aura TTS)
- **LLM**: Google Gemini 1.5 Flash (via LangChain)
- **Vector DB**: FAISS (local)
- **Frontend**: Vanilla HTML/CSS/JavaScript

## Optional: Twilio Integration

To connect with a phone number:
1. Sign up for Twilio trial
2. Configure webhook to point to `/chat` endpoint
3. Use Twilio Media Streams for audio handling

## Troubleshooting

**Issue**: "Deepgram API key not configured"  
**Solution**: Make sure `.env` file exists and contains `DEEPGRAM_API_KEY`

**Issue**: "Could not access microphone"  
**Solution**: Grant browser permission to use microphone (HTTPS required for production)

**Issue**: "Module not found" errors  
**Solution**: Run `pip install -r requirements.txt`

**Issue**: CSV not loading  
**Solution**: Ensure `hospital.csv` is in the same directory as `rag_engine.py`

## Development

Built with ❤️ for Loop Health Intern Assignment

**Note**: This project uses AI tools (Claude, ChatGPT) for code generation, which is encouraged per assignment instructions.
