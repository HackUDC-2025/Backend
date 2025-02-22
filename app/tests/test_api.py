import pytest
from fastapi.testclient import TestClient
from app.main import app  # Make sure to import your FastAPI app
import os
import time  # Add delay to avoid race conditions

client = TestClient(app)

### ✅ Existing Tests
def test_root():
    """✅ Test if the API root endpoint is accessible"""
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "FastAPI is running!"}  # Adjusted expected value


def test_search_image():
    """✅ Test the /milvus/search endpoint with a valid image"""
    
    image_path = os.path.join(os.path.dirname(__file__), "test_image.jpg")  # ✅ Correct relative path
    assert os.path.exists(image_path), f"❌ Test image not found at {image_path}"  # Ensure file exists

    with open(image_path, "rb") as image:
        files = {"file": image}
        data = {"profile": "6 years student"}

        response = client.post("/milvus/search", files=files, data=data)
        
        assert response.status_code in {200, 404}  # Must return valid response codes
        if response.status_code == 200:
            json_response = response.json()
            assert "art_class" in json_response
            assert "description" in json_response

def test_generate_audio_invalid_language():
    """🚨 Test if the API returns an error for an unsupported language."""
    payload = {
        "text": "Hello, welcome to the museum.",
        "language": "xx"  # Invalid language code
    }
    response = client.post("/text-to-speech", json=payload)

    assert response.status_code == 400
    assert "Error generating audio" in response.json()["detail"]

def test_generate_audio_empty_text():
    """🚨 Test if the API returns an error when text is empty."""
    payload = {
        "text": "",
        "language": "en"
    }
    response = client.post("/text-to-speech", json=payload)

    assert response.status_code == 400
    assert "Error generating audio" in response.json()["detail"]

