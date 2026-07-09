import os

from deepface import DeepFace


class DeepFaceRecognitionEngine:
    def __init__(self):
        print("[INFO] Initializing DeepFace Engine with Facenet512 + OpenCV...")

    def create_encoding(self, image_path: str) -> list:
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"File missing: {image_path}")

        print(f"[INFO] Creating encoding for {image_path}...")

        try:
            # Generate 512-D embedding using Facenet512 and OpenCV face detection
            embedding_objs: list[dict[str, any]] = DeepFace.represent(  # type: ignore
                img_path=image_path,
                model_name="Facenet512",
                detector_backend="opencv",
                enforce_detection=True,
            )
            return embedding_objs[0]["embedding"]  # Return raw list representation
        except Exception as e:
            raise ValueError(
                f"No face isolated within target space or processing error: {str(e)}"
            )

    def verify_two_images(self, encodings_1: list, image_path_2: str) -> dict:
        if not os.path.exists(image_path_2):
            return {"error": f"File missing: {image_path_2}"}

        try:
            print("[INFO] Executing biometric verification via DeepFace...")

            # Compare pre-computed embedding list with the second image via Cosine metric
            result = DeepFace.verify(
                img1_path=encodings_1,
                img2_path=image_path_2,
                model_name="Facenet512",
                detector_backend="opencv",
                distance_metric="cosine",
                enforce_detection=True,
            )

            # Map framework output directly back to your downstream dictionary keys
            return {
                "same_person": bool(result["verified"]),
                "distance_score": round(float(result["distance"]), 4),
                "tolerance_threshold": result["threshold"],
                "metric_used": "cosine",
            }
        except Exception as e:
            return {"error": f"Biometric calculation failure: {str(e)}"}
