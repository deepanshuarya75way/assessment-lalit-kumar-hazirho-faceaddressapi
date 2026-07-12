import os
import shutil
from fastapi import FastAPI, File, HTTPException, UploadFile, status

from src.face_engine import DeepFaceRecognitionEngine
from src.geocoder import geocode_text_block
from src.ocr_engine_rapid import RapidOCREngine

app = FastAPI(title="Vision & Spatial Processing API")

ocr_service = RapidOCREngine()
face_service = DeepFaceRecognitionEngine()

TEMP_DIR = "data"
os.makedirs(TEMP_DIR, exist_ok=True)

ALLOWED_MEDIA_TYPES = {"image/jpeg", "image/jpg", "image/png"}

def validate_image_type(
    content_type: str,
) -> None:
    """Centralized validation helper to clean up duplicate endpoint validation."""
    if content_type not in ALLOWED_MEDIA_TYPES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Input media must be JPEG, JPG, or PNG formats.",
        )

# --- ADDRESS PROCESSING ---
@app.post("/v1/address/process", status_code=status.HTTP_200_OK)
async def process_address_image(
    file: UploadFile = File(...),
):
    validate_image_type(file.content_type)

    temp_file_path = os.path.join(TEMP_DIR, f"api_ocr_{os.getpid()}_{file.filename}")
    try:
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        extracted_text = ocr_service.extract_text(temp_file_path)
        if not extracted_text or not extracted_text.strip():
            return {
                "ocr_status": "NO_TEXT_FOUND",
                "extracted_text": "",
                "geocoding_results": None,
            }

        spatial_payload = geocode_text_block(extracted_text)
        return {
            "ocr_status": "SUCCESS",
            "extracted_text": extracted_text,
            "geocoding_results": spatial_payload,
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal OCR or Geocoding subsystem error: {str(e)}",
        )
    finally:
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)


# --- FACE ENCODING ---
@app.post("/v1/face/encoding", status_code=status.HTTP_200_OK)
async def generate_face_embedding(
    file: UploadFile = File(...),
):
    validate_image_type(file.content_type)

    temp_file_path = os.path.join(TEMP_DIR, f"api_face_enc_{os.getpid()}_{file.filename}")
    try:
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # face_service.create_encoding raises ValueError if DeepFace fails
        vector_list = face_service.create_encoding(temp_file_path)
        return {
            "status": "SUCCESS",
            "dimension_length": len(vector_list),
            "face_encoding": vector_list,
        }
    except ValueError as ve:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(ve)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal face extraction error: {str(e)}",
        )
    finally:
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)


# --- FACE VERIFICATION ---import os
import shutil
from typing import List
from fastapi import FastAPI, File, HTTPException, UploadFile, status

from src.face_engine import DeepFaceRecognitionEngine
from src.geocoder import geocode_text_block
from src.ocr_engine_rapid import RapidOCREngine

app = FastAPI(title="Vision & Spatial Processing API")

# Initialize engines
ocr_service = RapidOCREngine()
face_service = DeepFaceRecognitionEngine()

TEMP_DIR = "data"
os.makedirs(TEMP_DIR, exist_ok=True)

ALLOWED_MEDIA_TYPES = {"image/jpeg", "image/jpg", "image/png"}

def validate_image_type(content_type: str) -> None:
    """Centralized validation for supported image formats."""
    if content_type not in ALLOWED_MEDIA_TYPES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Input media must be JPEG, JPG, or PNG formats.",
        )

# --- ADDRESS ENDPOINT ---
@app.post("/v1/address/process", status_code=status.HTTP_200_OK)
async def process_address_image(file: UploadFile = File(...)):
    validate_image_type(file.content_type)

    temp_file_path = os.path.join(TEMP_DIR, f"api_ocr_{os.getpid()}_{file.filename}")
    try:
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        extracted_text = ocr_service.extract_text(temp_file_path)
        if not extracted_text or not extracted_text.strip():
            return {
                "ocr_status": "NO_TEXT_FOUND",
                "extracted_text": "",
                "geocoding_results": None,
            }

        spatial_payload = geocode_text_block(extracted_text)
        return {
            "ocr_status": "SUCCESS",
            "extracted_text": extracted_text,
            "geocoding_results": spatial_payload,
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal OCR or Geocoding error: {str(e)}",
        )
    finally:
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)


# --- FACE RECOGNITION ENDPOINTS ---
@app.post("/v1/face/encoding", status_code=status.HTTP_200_OK)
async def generate_face_embedding(file: UploadFile = File(...)):
    validate_image_type(file.content_type)

    temp_file_path = os.path.join(TEMP_DIR, f"api_face_enc_{os.getpid()}_{file.filename}")
    try:
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        vector_list = face_service.create_encoding(temp_file_path)
        return {
            "status": "SUCCESS",
            "dimension_length": len(vector_list),
            "face_encoding": vector_list,
        }
    except ValueError as ve:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(ve)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal face extraction error: {str(e)}",
        )
    finally:
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)


@app.post("/v1/face/verify", status_code=status.HTTP_200_OK)
async def verify_two_faces(
    file_a: UploadFile = File(...),
    file_b: UploadFile = File(...),
):
    validate_image_type(file_a.content_type)
    validate_image_type(file_b.content_type)

    path_a = os.path.join(TEMP_DIR, f"verify_{os.getpid()}_A_{file_a.filename}")
    path_b = os.path.join(TEMP_DIR, f"verify_{os.getpid()}_B_{file_b.filename}")
    try:
        with open(path_a, "wb") as buffer_a:
            shutil.copyfileobj(file_a.file, buffer_a)
        with open(path_b, "wb") as buffer_b:
            shutil.copyfileobj(file_b.file, buffer_b)

        verification_result = face_service.verify_two_images(path_a, path_b)

        if isinstance(verification_result, dict) and "error" in verification_result:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=verification_result["error"],
            )

        return verification_result
    except ValueError as ve:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(ve)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal verification error: {str(e)}",
        )
    finally:
        if os.path.exists(path_a):
            os.remove(path_a)
        if os.path.exists(path_b):
            os.remove(path_b)
@app.post("/v1/face/verify", status_code=status.HTTP_200_OK)
async def verify_two_faces(
    file_a: UploadFile = File(...),
    file_b: UploadFile = File(...),
):
    validate_image_type(file_a.content_type)
    validate_image_type(file_b.content_type)

    path_a = os.path.join(TEMP_DIR, f"verify_{os.getpid()}_A_{file_a.filename}")
    path_b = os.path.join(TEMP_DIR, f"verify_{os.getpid()}_B_{file_b.filename}")
    try:
        with open(path_a, "wb") as buffer_a:
            shutil.copyfileobj(file_a.file, buffer_a)
        with open(path_b, "wb") as buffer_b:
            shutil.copyfileobj(file_b.file, buffer_b)

        # face_service.verify_two_images catches its own internal exceptions
        # and returns {"error": "..."} instead of raising them.
        verification_result = face_service.verify_two_images(path_a, path_b)

        # Catching the engine's internal error dictionary to prevent a false 200 OK status
        if isinstance(verification_result, dict) and "error" in verification_result:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=verification_result["error"],
            )

        return verification_result
    except HTTPException:
        # Re-raise HTTPExceptions caught from the block above
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal verification error: {str(e)}",
        )
    finally:
        if os.path.exists(path_a):
            os.remove(path_a)
        if os.path.exists(path_b):
            os.remove(path_b)
