from fastapi import APIRouter
from fastapi.responses import FileResponse
from pydantic import BaseModel
from gtts import gTTS

router = APIRouter()


class TextToSpeechRequest(BaseModel):
    text: str
    language: str = "es"  # Idioma por defecto: español


@router.post("/text-to-speech")
async def generate_audio(request: TextToSpeechRequest):
    try:
        tts = gTTS(text=request.text, lang=request.language)
        audio_file = "temp.mp3"
        tts.save(audio_file)

        return FileResponse(
            audio_file, media_type="audio/mpeg", filename="audio-guide.mp3"
        )
    except Exception as e:
        return {"message": f"Error al generar audio: {str(e)}"}
