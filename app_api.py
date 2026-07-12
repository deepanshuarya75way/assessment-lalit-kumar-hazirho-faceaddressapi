import os
import shutil
from sys import path

from fastapi import Body, FastAPI, File, HTTPException, UploadFile, status

from src.face_engine import DeepFaceRecognitionEngine
from src.geocoder import geocode_text_block
from src.ocr_engine_rapid import RapidOCREngine

app = FastAPI()

ocr_service = RapidOCREngine()
face_service = DeepFaceRecognitionEngine()

TEMP_DIR = "data"
os.makedirs(TEMP_DIR, exist_ok=True)


# ADDRESS
@app.post("/v1/address/process", status_code=status.HTTP_200_OK)
async def process_address_image(file: UploadFile = File(...)):
    if file.content_type not in [
        "image/jpeg",
        "image/png",
        "image/jpg",
    ]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Input media must be JPEG or PNG formats.",
        )

    temp_file_path = os.path.join(TEMP_DIR, f"api_ocr_{os.getpid()}_{file.filename}")
    try:
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        extracted_text = ocr_service.extract_text(temp_file_path)
        if not extracted_text.strip():
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
    finally:
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)


# FACE recognition
@app.post("/v1/face/encoding", status_code=status.HTTP_200_OK)
async def generate_face_embedding(
    file: UploadFile = File(...),
):
    if file.content_type not in [
        "image/jpeg",
        "image/jpg",
        "image/png",
    ]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Input media must be JPEG, JPG, or PNG formats.",
        )

    temp_file_path = os.path.join(
        TEMP_DIR, f"api_face_enc_{os.getpid()}_{file.filename}"
    )
    try:
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        vector_list = face_service.create_encoding(temp_file_path)
        return {
            "status": "SUCCESS",
            "dimension_length": len(vector_list),  # Will output 512
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
    if file_b.content_type not in ["image/jpeg", "image/png", "image/jpg"]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded verification media must be JPEG or PNG.",
        )
    if file_a.content_type not in ["image/jpeg", "image/png", "image/jpg"]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded verification media must be JPEG or PNG.",
        )

    path_b = os.path.join(TEMP_DIR, f"verify_{os.getpid()}_B_{file_b.filename}")
    path_a = os.path.join(TEMP_DIR, f"verify_{os.getpid()}_A_{file_a.filename}")
    try:
        with open(path_a, "wb") as buffer_a:
            shutil.copyfileobj(file_a.file, buffer_a)
        with open(path_b, "wb") as buffer_b:
            shutil.copyfileobj(file_b.file, buffer_b)

        verification_result = face_service.verify_two_images(path_a, path_b)

        if "error" in verification_result:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=verification_result["error"],
            )

        return verification_result
    finally:
        if os.path.exists(path_b):
            os.remove(path_b)
        if os.path.exists(path_a):
            os.remove(path_a)
