from fastapi import APIRouter, HTTPException, UploadFile, File
from app.services.milvus_service import find_similar_class
from app.core.logger import logger
from PIL import Image

router = APIRouter()

@router.post("/search")
async def search_image(file: UploadFile = File(...)):
    """🔍 Search for similar embeddings in Milvus."""
    
    logger.info("📸 Received image search request.")

    try:
        logger.info("🔍 Searching for similar images...")

        image = Image.open(file.file).convert("RGB")
        result = find_similar_class(image)

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
