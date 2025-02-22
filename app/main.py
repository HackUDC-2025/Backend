from contextlib import asynccontextmanager
from fastapi import FastAPI
from app.routers import router
from app.services.milvus_config import collection
from app.services.milvus_service import populate_database
from app.config import settings
from pymilvus import Collection, utility

@asynccontextmanager
async def lifespan(app: FastAPI):
    if not utility.has_collection(settings.MILVUS_COLLECTION_NAME):
        print("No collection found. Populating database...")
        populate_database()
    else:
        print(f"Database already populated with {collection.num_entities} embeddings.")
    yield

app = FastAPI(
    lifespan=lifespan,
    title="ArtLens HackUDC2025 API",
    description="API for the ArtLens HackUDC2025 project. This API is used to interact with the Milvus database.",
    version="1.0.0",
    contact={
        "name": "Mario Casado Diez",
        "email": "mario.diez@udc.es",
        "url": "https://github.com/mario-diez",
        
    },              )

print("🚀 Starting FastAPI server")

app.include_router(router.router, prefix="/milvus", tags=["Milvus"])

@app.get("/")
def home():
    print("🏠 Home route accessed")
    return {"message": "FastAPI + Milvus is running!"}