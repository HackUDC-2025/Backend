from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    MILVUS_HOST: str = "127.0.0.1"
    MILVUS_PORT: int = 19530
    DATASET_PATH:str  = "dataset/art_images"
    MILVUS_COLLECTION_NAME: str = "museum_embeddings"
    PARAMS_SEARCH: dict = {
        "index_type": "IVF_FLAT",
        "metric_type": "COSINE",
        "params": {"nlist": 128}
    }

settings = Settings()
