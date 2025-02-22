from collections import defaultdict
import json
import os
import clip
import numpy as np
from PIL import Image
import torch
from tqdm import tqdm
from app.config import settings
from app.services.milvus_config import collection
import requests
from app.core.logger import logger

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


def find_similar_class(image: Image.Image, profile: str):
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
        output_fields=["class_name"],
    )
    if results and results[0]:
        art_name = results[0][0].entity.get("class_name")
        ollama_response = generate_description_with_ollama(art_name, profile)
        return {"predicted_class": art_name, "description": ollama_response}

    return {"predicted_class": "Unknown", "description": "No match found."}


def classify_profile(profile: str) -> str:
    prompt = f"Clasifica el siguiente perfil de usuario en uno de estos tres niveles: principiante, intermedio o avanzado, basándote en su nivel de conocimiento en arte. El perfil es: {profile}. Responde solo con uno de estos tres niveles: 'principiante', 'intermedio' o 'avanzado'.\n\nEjemplos:\n'Principiante': Un estudiante que está comenzando a estudiar arte, sin mucho conocimiento previo sobre técnicas o historia del arte.\n'Intermedio': Alguien que tiene algunos años de experiencia o estudio en arte, entiende las técnicas básicas y la historia, y puede hablar con cierta profundidad sobre el tema.\n'Avanzado': Un experto, artista profesional o alguien con un amplio conocimiento sobre la historia, teorías y técnicas avanzadas del arte, como un doctor en historia del arte, que tiene un conocimiento profundo de las obras y contextos históricos, y puede hacer investigaciones detalladas o enseñar a otros a nivel académico."

    payload = {
        "model": "llama3.2",
        "prompt": prompt,
        "stream": False,
        "num_predict": 10,
    }

    try:
        response = requests.post(OLLAMA_API_URL, json=payload)
        response_json = response.json()
        level = response_json.get("response", "No response found.")
        logger.success(level)
        return level.strip().lower().replace("'", "")
    except Exception as e:
        return f"Error when classifying profile: {str(e)}"


def generate_description_with_ollama(art_name: str, profile: str) -> str:
    art_name = art_name.replace("_", " ")

    # profile_level = classify_profile(profile)
    # print(f"Profile level: {profile_level}")
    # if profile_level == "principiante":
    #     max_tokens = 100
    # elif profile_level == "intermedio":
    #     max_tokens = 200
    # elif profile_level == "avanzado":
    #     max_tokens = 300
    # else:
    #     max_tokens = 200

    max_tokens = 200

    prompt = f"""
    Eres un experto guía de museo y tu tarea es proporcionar una explicación detallada sobre la obra de arte "{art_name}".
    Adapta la explicación al perfil de {profile}, asegurándote de que la información sea clara y adecuada para su nivel de conocimiento.

    🔹 **Formato de salida esperado:**
    - **Título**: Nombre completo de la obra.
    - **Autor**: Nombre del creador.
    - **Año**: Año en que fue realizada.
    - **Descripción**: Un análisis detallado de al menos {max_tokens + 100} palabras, destacando el contexto histórico, los materiales utilizados, el significado de la obra y su impacto en el arte.

    No incluyas saludos ni introducciones, y asegúrate de que la respuesta sea coherente y con suficiente profundidad.
    """


    payload = {
    "model": "llama3.2:1b",
    "prompt": prompt,
    "stream": False,
    "num_predict": 500,  # 🔥 Prueba con 500 tokens o más
    "format": {
        "type": "object",
        "properties": {
            "description": {"type": "string"},
            "author": {"type": "string"},
            "year": {"type": "string"},
            "title": {"type": "string"}  
        },
        "required": ["description", "author", "year", "title"]
        }
    }


    try:
        response = requests.post(OLLAMA_API_URL, json=payload)
        response_json = response.json()

        response_str = response_json.get("response", "")
        if response_str:
            try:
                response_json = json.loads(response_str)

                return response_json
            except json.JSONDecodeError:
                return {"error": "Failed to decode JSON from response"}
        else:
            return {"error": "Response field is empty"}
    except Exception as e:
        return f"Error when connecting with ollama: {str(e)}"
