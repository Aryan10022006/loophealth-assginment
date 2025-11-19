import os
import io
from fastapi import FastAPI, UploadFile, File, HTTPException, Form, Request
from fastapi.responses import StreamingResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from deepgram import DeepgramClient
from agent import get_agent
import logging
from typing import Optional

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI(title="Loop AI Hospital Assistant")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Deepgram client - SDK v5+ uses environment variable
deepgram_api_key = os.getenv("DEEPGRAM_API_KEY")
if not deepgram_api_key:
    logger.warning("DEEPGRAM_API_KEY not found in environment variables")
    deepgram = None
else:
    deepgram = DeepgramClient(api_key=deepgram_api_key)

# Initialize the agent
agent = get_agent()

# Track if this is the first message for introduction
session_state = {"is_first_message": True}


@app.get("/", response_class=HTMLResponse)
async def read_root():
    """Serve the frontend HTML page."""
    try:
        with open("templates/index.html", "r", encoding="utf-8") as f:
            html_content = f.read()
        return HTMLResponse(content=html_content)
    except FileNotFoundError:
        return HTMLResponse(
            content="<h1>Error: Frontend template not found</h1>",
            status_code=404
        )


@app.post("/chat")
async def chat(
    request: Request,
    audio: Optional[UploadFile] = File(None),
    text_query: Optional[str] = Form(None)
):
    """
    Main chat endpoint that handles voice-to-voice conversation or text queries.
    
    Supports two input modes:
    1. FormData with 'audio' file (voice queries)
    2. JSON with 'text_query' field (followup button clicks)
    
    Flow:
    1. Receive audio file OR text query from frontend
    2. Convert audio to text using Deepgram STT (if audio provided)
    3. Process query with Loop AI agent
    4. Convert response text to audio using Deepgram TTS
    5. Return audio to frontend with custom headers
    """
    try:
        # Check if it's a JSON request (followup queries)
        content_type = request.headers.get('content-type', '')
        if 'application/json' in content_type:
            body = await request.json()
            text_query = body.get('text_query')
            logger.info(f"Received JSON text query: {text_query}")
        else:
            logger.info("Received FormData request")
        
        # Check if Deepgram is initialized
        if not deepgram:
            raise HTTPException(
                status_code=500,
                detail="Deepgram API key not configured. Please set DEEPGRAM_API_KEY in .env file"
            )
        
        # Step 1: Get user input (either from audio or text)
        transcript = None
        
        if text_query:
            # Handle text query from followup buttons
            transcript = text_query
            logger.info(f"Processing text query: {transcript}")
        elif audio:
            # Handle audio query
            audio_data = await audio.read()
            logger.info(f"Audio file size: {len(audio_data)} bytes")
            
            # Step 2: Speech-to-Text using Deepgram (Prerecorded)
            logger.info("Converting speech to text...")
            
            try:
                # Determine mime type from upload if available
                mime_type = getattr(audio, 'content_type', None) or 'audio/webm'

                # Transcribe audio (pass raw bytes as 'request' and options as keywords)
                response = deepgram.listen.v1.media.transcribe_file(
                    request=audio_data,
                    model="nova-2",
                    smart_format=True,
                )

                # Extract transcript (SDK response objects have .results)
                try:
                    transcript = response.results.channels[0].alternatives[0].transcript
                except Exception:
                    # Fallback to dict-style access if needed
                    try:
                        transcript = response["results"]["channels"][0]["alternatives"][0]["transcript"]
                    except Exception:
                        transcript = None

                logger.info(f"User said: {transcript}")

                if not transcript or transcript.strip() == "":
                    raise HTTPException(
                        status_code=400,
                        detail="Could not understand audio. Please speak clearly."
                    )
            except HTTPException:
                raise
            except Exception as e:
                logger.exception("STT Error")
                raise HTTPException(
                    status_code=500,
                    detail=f"Speech recognition failed: {str(e)}"
                )
        else:
            raise HTTPException(
                status_code=400,
                detail="Either audio file or text_query must be provided"
            )
        
        # Step 3: Process with AI agent
        logger.info("Processing query with AI agent...")
        
        try:
            ai_response = agent.process_query(
                transcript,
                is_first_message=session_state["is_first_message"]
            )
            session_state["is_first_message"] = False
            logger.info(f"AI response: {ai_response}")
        
        except Exception as e:
            logger.error(f"Agent Error: {e}")
            raise HTTPException(
                status_code=500,
                detail=f"AI processing failed: {str(e)}"
            )
        
        # Step 4: Text-to-Speech using Deepgram (Aura)
        logger.info("Converting text to speech...")
        
        try:
            # Generate speech using SDK v5 API (iterator of bytes)
            audio_iter = deepgram.speak.v1.audio.generate(
                text=ai_response,
                model="aura-asteria-en",
                encoding="linear16",
                container="wav",
            )

            # Collect bytes from iterator
            audio_chunks = []
            for chunk in audio_iter:
                audio_chunks.append(chunk)
            audio_response = b"".join(audio_chunks)
            
            if not audio_response:
                raise Exception("No audio generated from TTS")
            
            logger.info(f"Generated audio size: {len(audio_response)} bytes")
        
        except Exception as e:
            logger.error(f"TTS Error: {e}")
            raise HTTPException(
                status_code=500,
                detail=f"Text-to-speech failed: {str(e)}"
            )
        
        # Step 5: Return audio response with text in headers
        # Clean strings for HTTP headers (remove newlines and control characters)
        clean_transcript = transcript.replace('\n', ' ').replace('\r', ' ')
        clean_ai_response = ai_response.replace('\n', ' ').replace('\r', ' ')
        
        return StreamingResponse(
            io.BytesIO(audio_response),
            media_type="audio/wav",
            headers={
                "Content-Disposition": "attachment; filename=response.wav",
                "X-Transcript": clean_transcript,
                "X-AI-Response": clean_ai_response
            }
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in /chat endpoint: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"An unexpected error occurred: {str(e)}"
        )


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "deepgram_configured": deepgram is not None,
        "agent_initialized": agent is not None
    }


if __name__ == "__main__":
    import uvicorn
    
    # Create templates directory if it doesn't exist
    os.makedirs("templates", exist_ok=True)
    
    logger.info("Starting Loop AI Hospital Assistant server...")
    logger.info("Make sure to set DEEPGRAM_API_KEY and GOOGLE_API_KEY in .env file")
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )
