from pydantic import BaseModel


class SearchResponse(BaseModel):
    art_class: str
    description: str


class ProfileConfig(BaseModel):
    name: str
    technical_level: str
    language_style: str
    max_tokens: int
    special_instructions: str


class Base64ImageRequest(BaseModel):
    image_base64: str
    profile: ProfileConfig
