from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import Response
from pydantic import BaseModel
from google import genai
from google.genai import types
import io
import wave
import re
import base64
import tempfile
import os

# Create our FastAPI application
app = FastAPI(title="AI Voice Dialer")

# Connect to Gemini using the environment variable
client = genai.Client(
    http_options={"api_version": "v1beta"}
)

# Customer calling information
class CallRequest(BaseModel):
    customer_name: str
    phone_number: str


# Customer text message
class ChatRequest(BaseModel):
    message: str


# Text that we want to convert into speech
class SpeechRequest(BaseModel):
    text: str


# Home endpoint
@app.get("/")
def home():
    return {"message": "AI Voice Dialer Backend is running"}


# Health-check endpoint
@app.get("/health")
def health():
    return {"status": "healthy"}


# Receive a calling request
@app.post("/call")
def make_call(request: CallRequest):
    if not re.fullmatch(r"\+[1-9]\d{7,14}", request.phone_number):
        raise HTTPException(
            status_code=400, detail="Invalid international phone number"
        )

    return {
        "status": "request_received",
        "customer": request.customer_name,
        "phone_number": request.phone_number,
        "message": "Calling request received successfully",
    }


# AI chat endpoint
@app.post("/chat")
def chat_with_ai(request: ChatRequest):
    instructions = """
    You are a polite AI customer-service voice assistant.
    Use simple English and short answers.
    Ask one question at a time.
    Never invent prices or package details.
    Clearly identify yourself as an AI assistant.
    """

    try:
        response = client.models.generate_content(
            model="gemini-3.8-flash",
            contents=request.message,
            config=types.GenerateContentConfig(system_instruction=instructions),
        )

        return {
            "customer_message": request.message,
            "ai_response": response.text,
        }

    except Exception:
        raise HTTPException(
            status_code=502, detail="The AI service could not process the request"
        )


# Convert an uploaded audio file into text
@app.post("/transcribe")
def transcribe_audio(file: UploadFile = File(...)):
    # Check that the uploaded file is audio
    if not file.content_type or not file.content_type.startswith("audio/"):
        raise HTTPException(status_code=400, detail="Please upload an audio file")

    temp_path = None

    try:
        # Create a temporary file on our computer
        suffix = os.path.splitext(file.filename)[1] or ".wav"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as temp_file:
            temp_path = temp_file.name
            temp_file.write(file.file.read())

        # Upload the audio file to Gemini
        uploaded_audio = client.files.upload(file=temp_path)
        
        # Ask Gemini to transcribe the audio
        result = client.models.generate_content(
            model="gemini-3.7-flash",
            contents=["Generate a transcript of the speech.", uploaded_audio],
        )

        return {"transcript": result.text}

    except Exception as e:
        print("TRANSCRIBE ERROR:", repr(e))
        raise HTTPException(
            status_code=502, detail="Audio transcription failed. Check the server logs."
        )

    finally:
        # Delete the temporary file from our computer
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)


# Convert written text into spoken audio






@app.post("/speak")
def generate_speech(request: SpeechRequest):
    try:
        result = client.interactions.create(
            model="gemini-3.8-flash-tts",

            input=[{
                "type": "user_input",
                "content": [{
                    "type": "text",
                    "text": request.text
                }]
            }],

            response_format={"type": "audio"},

            generation_config={
                "speech_config": [
                    {"voice": "Kore"}
                ]
            }
        )

        if not result.output_audio:
            raise RuntimeError("Gemini returned no audio")

        audio_data = base64.b64decode(
            result.output_audio.data
        )

        return Response(
            content=audio_data,
            media_type="audio/wav",
            headers={
                "Content-Disposition":
                "attachment; filename=ai_response.wav"
            }
        )

    except Exception as e:
        print("TTS ERROR:", repr(e))

        raise HTTPException(
            status_code=502,
            detail="Speech generation failed. Check server logs."
        )



