import io
import os
import tempfile
from typing import Annotated, Optional

import cv2
import dotenv
import numpy as np
from fastapi import (
    FastAPI,
    File,
    HTTPException,
    UploadFile,
    status,
)
from fastapi.concurrency import run_in_threadpool
from PIL import Image, ImageOps

from src.face_engine import DeepFaceRecognitionEngine
from src.geocoder_geoapify import geocode
from src.liveness_engine import LivenessDetectionEngine
from src.ocr_engine_rapid import RapidOCREngine

dotenv.load_dotenv()
apikey: str = str(os.environ.get("GEO_API_FY"))

app = FastAPI(title="Vision & Spatial Processing API")

ocr_service = RapidOCREngine()
face_service = DeepFaceRecognitionEngine()
liveness_service = LivenessDetectionEngine(threshold=0.75)

ALLOWED_MEDIA_TYPES = {
    "image/jpeg",
    "image/jpg",
    "image/png",
}


def validate_image_type(content_type: Optional[str]) -> None:
    if content_type not in ALLOWED_MEDIA_TYPES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only JPEG and PNG images are supported.",
        )


def is_valid_image_bytes(raw_bytes: bytes) -> bool:
    """Checks actual JPEG/PNG file signatures (magic bytes) rather than HTTP headers."""
    is_jpeg = raw_bytes.startswith(b"\xff\xd8\xff")
    is_png = raw_bytes.startswith(b"\x89PNG\r\n\x1a\n")
    return is_jpeg or is_png


async def read_and_decode_image(upload_file: UploadFile) -> np.ndarray:
    """Reads uploaded bytes, validates image signature, fixes EXIF rotation, and returns BGR array."""
    raw_bytes = await upload_file.read()
    if not raw_bytes:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Uploaded file is empty: {upload_file.filename}",
        )

    if not is_valid_image_bytes(raw_bytes):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Only valid JPEG and PNG images are supported: {upload_file.filename}",
        )

    try:
        pil_img = Image.open(io.BytesIO(raw_bytes))
        pil_img = ImageOps.exif_transpose(pil_img).convert("RGB")
        frame = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        return frame
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid or corrupted image file: {upload_file.filename}",
        )


async def save_upload_file(upload_file: UploadFile) -> str:
    def write_file() -> str:
        ext = os.path.splitext(upload_file.filename or "")[1] or ".png"
        fd, path = tempfile.mkstemp(suffix=ext)

        with os.fdopen(fd, "wb") as file:
            file.write(upload_file.file.read())

        return path

    return await run_in_threadpool(write_file)


##############
#  ADDRESS
##############


@app.post(
    "/v1/address/process",
    status_code=status.HTTP_200_OK,
)
async def process_address_image(
    file: Annotated[UploadFile, File()],
):
    validate_image_type(file.content_type)
    temp_path = await save_upload_file(file)

    try:
        text = await run_in_threadpool(
            ocr_service.extract_text,
            temp_path,
        )

        if not text.strip():
            return {
                "ocr_status": "NO_TEXT_FOUND",
                "extracted_text": "",
                "geocoding_results": None,
            }

        result = await run_in_threadpool(
            geocode,
            address=text,
            api_key=apikey,
        )

        return {
            "ocr_status": "SUCCESS",
            "extracted_text": text,
            "geocoding_results": result,
        }

    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(exc),
        )

    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


###############
# FACE ENCODING
###############


@app.post(
    "/v1/face/encoding",
    status_code=status.HTTP_200_OK,
)
async def generate_face_embedding(
    file: Annotated[UploadFile, File()],
):
    validate_image_type(file.content_type)
    temp_path = await save_upload_file(file)

    try:
        encoding = await run_in_threadpool(
            face_service.create_encoding,
            temp_path,
        )

        return {
            "status": "SUCCESS",
            "dimension_length": len(encoding),
            "face_encoding": encoding,
        }

    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        )

    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(exc),
        )

    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


###############
# FACE VERIFY
###############


@app.post(
    "/v1/face/verify",
    status_code=status.HTTP_200_OK,
)
async def verify_two_faces(
    file_a: Annotated[UploadFile, File()],
    file_b: Annotated[UploadFile, File()],
):
    validate_image_type(file_a.content_type)
    validate_image_type(file_b.content_type)

    path_a = await save_upload_file(file_a)
    path_b = await save_upload_file(file_b)

    try:
        result = await run_in_threadpool(
            face_service.verify_two_images,
            path_a,
            path_b,
        )

        if isinstance(result, dict) and "error" in result:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=result["error"],
            )

        return result

    except HTTPException:
        raise

    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(exc),
        )

    finally:
        if os.path.exists(path_a):
            os.remove(path_a)

        if os.path.exists(path_b):
            os.remove(path_b)


###################
# LIVENESS CHECK
###################


@app.post(
    "/v1/face/liveness/single",
    status_code=status.HTTP_200_OK,
)
async def check_single_face_liveness(
    file: Annotated[UploadFile, File()],
):
    frame = await read_and_decode_image(file)

    try:
        result = await run_in_threadpool(
            liveness_service.evaluate,
            [frame],
        )
        return result

    except HTTPException:
        raise

    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(exc),
        )


@app.post(
    "/v1/face/liveness",
    status_code=status.HTTP_200_OK,
)
async def check_face_liveness(
    files: Annotated[list[UploadFile], File()],
):
    if not (1 <= len(files) <= 5):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Provide between 1 and 5 images.",
        )

    frames: list[np.ndarray] = []
    for file in files:
        frame = await read_and_decode_image(file)
        frames.append(frame)

    try:
        result = await run_in_threadpool(
            liveness_service.evaluate,
            frames,
        )
        return result

    except HTTPException:
        raise

    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(exc),
        )