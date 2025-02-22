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
