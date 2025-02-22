from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routers.search import router as search_router
from app.routers.audio import router as audio_router
from app.services.milvus_config import collection
from app.services.search_service import populate_database
from app.config import settings
from pymilvus import utility


@asynccontextmanager
async def lifespan(app: FastAPI):
    if (
        not utility.has_collection(settings.MILVUS_COLLECTION_NAME)
        or collection.is_empty
    ):
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
    },
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
print("🚀 Starting FastAPI server")

app.include_router(search_router, tags=["search"])
app.include_router(audio_router, tags=["audio"])


@app.get("/")
def home():
    print("🏠 Home route accessed")
    return {"message": "FastAPI is running!"}
