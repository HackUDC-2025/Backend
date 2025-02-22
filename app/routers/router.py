from fastapi import APIRouter, HTTPException, UploadFile, File
from app.services.milvus_service import find_similar_class
from app.core.logger import logger
from PIL import Image
from pydantic import BaseModel

router = APIRouter()

class SearchResponse(BaseModel):
    art_class: str
    description: str

@router.post(
    "/search",
    response_model=SearchResponse,
    summary="Search for a similar image in Milvus",
    description="""
    This endpoint takes an image and searches for the most similar artwork in the Milvus database.

    **Parameters:**
    - `file`: The uploaded image in JPEG/PNG format.
    - `profile`: (Optional) A profile string that may influence the search.

    **Returns:**
    - The predicted art class and its description.
    """,
    responses={
        200: {
            "description": "Successfully found a matching artwork",
            "content": {
                "application/json": {
                    "example":{
                        "art_class": "Las Meninas Diego Velázquez",
                        "description": "A royal family scene, with King Philip IV and his mistress, Infanta Margarita, reflected in mirror behind them, observing Velázquez painting."
                    }
                }
            },
        },
        404: {
            "description": "No similar classes found",
            "content": {
                "application/json": {
                    "example": {"detail": "No match found."}
                }
            },
        },
        500: {
            "description": "Internal server error",
            "content": {
                "application/json": {
                    "example": {"detail": "Milvus search failed."}
                }
            },
        },
    }
)
@router.post("/search")
async def search_image(
    file: UploadFile = File(...),
    profile: str = None
    ):
    """🔍 Search for similar embeddings in Milvus."""
    
    logger.info("📸 Received image search request.")

    try:
        logger.info("🔍 Searching for similar images...")

        image = Image.open(file.file).convert("RGB")
        logger.success("sss{profile}ssss")
        result = find_similar_class(image,profile)
        logger.success(f"result")

        if result["predicted_class"] != 'Unknown':
            predicted_class = result["predicted_class"]
            description = result["description"]
            logger.success(f"✅ Search completed successfully. Found: {predicted_class}")

            return {"art_class": predicted_class, "description": description}
        
        logger.warning("❌ No similar classes found.")
        raise HTTPException(status_code=404, detail="No match found.")

    except Exception as e:
        logger.error("❌ Error while searching in Milvus: {}", str(e))
        raise HTTPException(status_code=500, detail="Milvus search failed.")
