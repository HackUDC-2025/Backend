from loguru import logger
from pymilvus import (
    connections, Collection, CollectionSchema, FieldSchema, DataType, utility
)
from app.config import settings

connections.connect("default", host=settings.MILVUS_HOST, port=str(settings.MILVUS_PORT))

fields = [
    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
    FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=512),
    FieldSchema(name="class_name", dtype=DataType.VARCHAR, max_length=100)
]

schema = CollectionSchema(fields, description="Artwork class embeddings")

if not utility.has_collection(settings.MILVUS_COLLECTION_NAME):
    collection = Collection(name=settings.MILVUS_COLLECTION_NAME, schema=schema)
    logger.info(f"✅ Collection '{settings.MILVUS_COLLECTION_NAME}' created successfully.")
else:
    collection = Collection(name=settings.MILVUS_COLLECTION_NAME)

index_params = settings.PARAMS_SEARCH
if not collection.indexes:
    collection.create_index(field_name="embedding", index_params=index_params)
    logger.info("✅ Index created for collection.")

collection.load()
logger.info("📌 Collection loaded into memory.")
