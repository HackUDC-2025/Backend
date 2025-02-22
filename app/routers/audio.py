from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
from gtts import gTTS
import os

router = APIRouter()


class TextToSpeechRequest(BaseModel):
    """
    Request model for text-to-speech conversion.

    Attributes:
    - `text` (str): The text that will be converted into audio.
    - `language` (str, optional): ISO 639-1 language code (default `"es"` for Spanish).
    """
    text: str
    language: str = "es"


@router.post("/text-to-speech", summary="Generate an audio file from text", response_class=FileResponse)
async def generate_audio(request: TextToSpeechRequest):
    """
    Converts the provided text into an MP3 audio file using `gTTS`.

    **Parameters:**
    - `request` (TextToSpeechRequest): JSON containing the text and language.

    **Returns:**
    - `audio/mp3`: The generated audio file.
    - `dict`: Error message if the generation fails.

    **Example JSON Request:**
    ```json
    {
        "text": "Hello, welcome to the museum.",
        "language": "en"
    }
    ```

    **Responses:**
    - `200`: Returns an `audio-guide.mp3` file.
    - `400`: Error if audio generation fails.
    """
    try:
        audio_file = "temp.mp3"
        tts = gTTS(text=request.text, lang=request.language)
        tts.save(audio_file)

        return FileResponse(
            audio_file, media_type="audio/mpeg", filename="audio-guide.mp3"
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error generating audio: {str(e)}")
    finally:
        if os.path.exists(audio_file):
            os.remove(audio_file)  # Clean up temporary file
