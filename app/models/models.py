from pydantic import BaseModel


class SearchResponse(BaseModel):
    art_class: str
    description: str


class ProfileConfig(BaseModel):
    name: str
    technicalLevel: str
    languageStyle: str
    maxTokens: int
    specialInstructions: str


class Base64ImageRequest(BaseModel):
    image_base64: str
    profile: ProfileConfig


class TextToSpeechRequest(BaseModel):
    """
    Request model for text-to-speech conversion.

    Attributes:
    - `text` (str): The text that will be converted into audio.
    - `language` (str, optional): ISO 639-1 language code (default `"es"` for Spanish).
    """

    text: str
    language: str = "es"
