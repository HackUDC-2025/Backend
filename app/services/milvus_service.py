from collections import defaultdict
import os
import clip
import numpy as np
from PIL import Image
import torch
from tqdm import tqdm
from app.config import settings
from app.services.milvus_config import collection
import requests

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load('ViT-B/32', device=device)
OLLAMA_API_URL = "http://193.144.51.204:11434/api/generate"

def get_image_embedding(image: Image.Image):
    """Convert image to CLIP embedding."""
    image = preprocess(image).unsqueeze(0).to(device)

    with torch.no_grad():
        embedding = model.encode_image(image)
    return embedding.cpu().numpy().tolist()[0]

def populate_database():
    """Compute one representative embedding per class and store in Milvus."""
    print("📂 Loading dataset images into Milvus...")
    class_embeddings = defaultdict(list)

    for class_name in tqdm(os.listdir(settings.DATASET_PATH)):
        class_path = os.path.join(settings.DATASET_PATH, class_name)
        if os.path.isdir(class_path):
            for img_name in os.listdir(class_path)[:30]:
                img_path = os.path.join(class_path, img_name)
                if os.path.isdir(img_path):
                    continue
                image = Image.open(img_path)
                embedding = get_image_embedding(image)
                class_embeddings[class_name].append(embedding)

    data_to_insert = {"embedding": [], "class_name": []}
    for class_name, embeddings in class_embeddings.items():
        avg_embedding = np.mean(embeddings, axis=0).astype(np.float32)

        data_to_insert["embedding"].append(avg_embedding)
        data_to_insert["class_name"].append(class_name)

    collection.insert([data_to_insert["embedding"], data_to_insert["class_name"]])

def find_similar_class(image: Image.Image,profile: str):
    """Finds the most similar class from Milvus."""
    query_embedding = get_image_embedding(image)

    if not collection.is_empty:
        collection.load()

    # Search
    results = collection.search(
        data=[query_embedding],
        anns_field="embedding",
        param=settings.PARAMS_SEARCH,
        limit=1,
        output_fields=["class_name"]
    )    
    if results and results[0]:
        art_name = results[0][0].entity.get("class_name")
        art_name = results[0][0].entity.get("class_name")
        ollama_response = generate_description_with_ollama(art_name, profile)
        return {"predicted_class": art_name, "description": ollama_response}

    return {"predicted_class": "Unknown", "description": "No match found."}

def generate_description_with_ollama(art_name: str, profile: str) -> str:
    art_name = art_name.replace("_", " ")
    prompt = f"You are a museum guide, imagine you have to describe the artwork '{art_name}' for a '{profile}' in a maximum of 100 characters, give me only the description of the artwork, dont give me anything that it is not description."

    payload = {
        "model": "llama3.2",
        "prompt": prompt,
        "stream": False,
        "max_tokens": 100,
    }

    try:
        response = requests.post(OLLAMA_API_URL, json=payload)
        response_json = response.json()
        return extract_final_text(response_json.get("response", "Response not found."))
    except Exception as e:
        return f"Error when connecting with ollama: {str(e)}"
    
def extract_final_text(input_text: str) -> str:
    parts = input_text.split("</think>")
    return parts[-1].strip() if len(parts) > 1 else input_text.strip()