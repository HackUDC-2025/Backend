from collections import defaultdict
import json
import os
import re
import clip
import numpy as np
from PIL import Image
import torch
from tqdm import tqdm
from app.config import settings
from app.models.models import ProfileConfig
from app.services.milvus_config import collection
import requests
from app.services.style_config import (
    get_language_instruction,
    get_technical_description,
)

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)
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
    class_metadata = dict()

    for class_name in tqdm(os.listdir(settings.DATASET_PATH)):
        class_path = os.path.join(settings.DATASET_PATH, class_name)
        if os.path.isdir(class_path):
            metadata_path = os.path.join(class_path, "metadata.json")

            if os.path.exists(metadata_path):
                with open(metadata_path, "r") as f:
                    metadata = json.load(f)
                    class_metadata[class_name] = metadata

            for img_name in os.listdir(class_path):
                img_path = os.path.join(class_path, img_name)

                if os.path.isdir(img_path) or not img_path.endswith(".jpg"):
                    continue
                image = Image.open(img_path)
                embedding = get_image_embedding(image)
                class_embeddings[class_name].append(embedding)

    data_to_insert = {"embedding": [], "class_name": [], "wikipedia": [], "prado": []}
    for class_name, embeddings in class_embeddings.items():
        avg_embedding = np.mean(embeddings, axis=0).astype(np.float32)

        data_to_insert["embedding"].append(avg_embedding)
        data_to_insert["class_name"].append(class_name)

        metadata = class_metadata.get(class_name, {})
        data_to_insert["wikipedia"].append(metadata.get("wikipedia", ""))
        data_to_insert["prado"].append(metadata.get("prado", ""))

    collection.insert(
        [
            data_to_insert["embedding"],
            data_to_insert["class_name"],
            data_to_insert["wikipedia"],
            data_to_insert["prado"],
        ]
    )


def find_similar_class(image: Image.Image, profile: ProfileConfig):
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
        output_fields=["class_name", "wikipedia", "prado"],
    )
    if results and results[0]:
        art_name = results[0][0].entity.get("class_name")
        wikipedia_url = results[0][0].entity.get("wikipedia")
        prado_url = results[0][0].entity.get("prado")
        ollama_response = generate_description_with_ollama(art_name, profile)
        return {
            "predicted_class": art_name,
            "description": ollama_response,
            "wikipedia_url": wikipedia_url,
            "prado_url": prado_url,
        }

    return {"predicted_class": "Unknown", "description": "No match found."}


def generate_description_with_ollama(art_name: str, profile: ProfileConfig) -> str:
    art_name = art_name.replace("_", " ")

    prompt = f"""
        Eres un guía de museo. Explica la obra de arte {art_name} de manera concisa, sobre todo enfocado para un perfil {profile.name}.

        - Nivel técnico: {profile.technicalLevel} ({get_technical_description(profile.technicalLevel)})
        - Estilo de lenguaje: {profile.languageStyle}
        - Público objetivo: {profile.name}

        ## Instrucciones específicas:
        1. Estilo de comunicación: {get_language_instruction(profile.languageStyle)}
        2. {profile.specialInstructions}
        3. Maximo de palabras: {profile.maxTokens}

        El formato de salida debe ser un JSON con la siguiente estructura, siendo todo strings de texto:
        {{
            "titulo": "Título de la obra",
            "autor": "Nombre del autor",
            "año": "Año de creación en formato de cadena de texto (string)",
            "descripcion": "Descripción detallada de la obra"
        }}

        ¡NO incluyas texto fuera del JSON! Asegúrate de que SOLO MANDAS el JSON y esté correctamente cerrado.
    """

    payload = {
        "model": "llama3.2",
        "prompt": prompt,
        "stream": False,
        "num_predict": profile.maxTokens + 100,
    }

    try:
        response = requests.post(OLLAMA_API_URL, json=payload)
        response_json = response.json()
        response_str = response_json.get("response", "")

        if response_str:
            print(response_str)
            try:
                json_str = repair_json(response_str)
                response_json = json.loads(json_str)
                return response_json
            except json.JSONDecodeError:
                return {"error": "Failed to decode JSON from response"}
        else:
            return {"error": "Response field is empty"}
    except Exception as e:
        return f"Error when connecting with ollama: {str(e)}"


def repair_json(raw_response: str) -> str:
    """
    Intenta reparar un JSON malformado:
    1. Extrae el fragmento que parece JSON.
    2. Añade llaves de cierre si faltan.
    3. Elimina texto sobrante.
    """
    match = re.search(r"\{.*", raw_response, re.DOTALL)
    if not match:
        raise ValueError("No se encontró JSON en la respuesta.")

    json_str = match.group(0).strip()

    open_braces = json_str.count("{")
    close_braces = json_str.count("}")

    if open_braces > close_braces:
        json_str += "}" * (open_braces - close_braces)

    last_brace = json_str.rfind("}")
    if last_brace != -1:
        json_str = json_str[: last_brace + 1]

    try:
        json.loads(json_str)
        return json_str
    except json.JSONDecodeError:
        return json.loads(json_str + "}")
    except Exception as e:
        raise ValueError(f"JSON irreparable: {e}")
