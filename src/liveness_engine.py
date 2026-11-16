from typing import Any

import cv2
import numpy as np
from deepface import DeepFace

GUIDANCE_MAP: dict[str, str] = {
    "no_face": "Center your face clearly inside the camera frame and try again.",
    "flatness": (
        "Present your real face directly to the camera—do not use a printed "
        "photo or another screen."
    ),
    "reflection": (
        "Reduce glare or bright reflections on your face or screen and move to "
        "even lighting."
    ),
    "motion_inconsistency": (
        "Hold the camera steady and blink or slightly turn your head naturally "
        "during capture."
    ),
}


class LivenessDetectionEngine:
    """
    Evaluates facial liveness across three criteria:
      1. Flatness (DeepFace MiniFASNet anti-spoofing + normalized Laplacian skin texture)
      2. Reflection (HSV specular glare + 2D FFT screen moire energy)
      3. Motion consistency (Background-compensated Farneback optical flow across 2-5 frames)
    """

    def __init__(self, threshold: float = 0.75) -> None:
        self.threshold = threshold
        self._warmup_models()

    def _warmup_models(self) -> None:
        """Preloads DeepFace detector and MiniFASNet weights into RAM on startup."""
        dummy_frame = np.zeros((224, 224, 3), dtype=np.uint8)
        try:
            DeepFace.extract_faces(
                img_path=dummy_frame,
                detector_backend="yunet",
                anti_spoofing=True,
                enforce_detection=False,
            )
        except Exception:
            pass

    def _analyze_flatness_and_spoof(
        self, frame: np.ndarray
    ) -> tuple[float, dict[str, Any]]:
        """
        Detects 2D flat surfaces (screens/prints) using DeepFace anti-spoofing
        combined with scale-normalized Laplacian skin texture variance.
        """
        faces = []
        for backend in ("yunet", "opencv"):
            try:
                faces = DeepFace.extract_faces(
                    img_path=frame,
                    detector_backend=backend,
                    anti_spoofing=True,
                    enforce_detection=True,
                )
                if faces:
                    break
            except ValueError:
                continue

        if not faces:
            return 0.0, {"error": "no_face"}

        primary = max(
            faces,
            key=lambda f: f.get("facial_area", {}).get("w", 0)
            * f.get("facial_area", {}).get("h", 0),
        )
        antispoof_score = float(primary.get("antispoof_score", 0.0))
        is_real = bool(primary.get("is_real", False))

        area = primary.get("facial_area", {})
        x = max(0, int(area.get("x", 0)))
        y = max(0, int(area.get("y", 0)))
        w = int(area.get("w", 0))
        h = int(area.get("h", 0))

        face_crop = frame[y : y + h, x : x + w]
        if face_crop.size == 0 or w < 40 or h < 40:
            return 0.0, {"error": "no_face"}

        gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
        norm_gray = cv2.resize(gray, (160, 160))
        laplacian_var = float(cv2.Laplacian(norm_gray, cv2.CV_64F).var())
        texture_score = min(1.0, laplacian_var / 120.0)

        flatness_score = (0.8 * antispoof_score) + (0.2 * texture_score)
        if not is_real:
            flatness_score *= 0.4

        return round(flatness_score, 4), {
            "face_crop": face_crop,
            "bbox": (x, y, w, h),
        }

    def _analyze_reflection(self, face_crop: np.ndarray) -> float:
        """
        Detects blown-out specular screen glare in HSV space and periodic
        pixel-grid moire interference via 2D Fast Fourier Transform.
        """
        hsv = cv2.cvtColor(face_crop, cv2.COLOR_BGR2HSV)
        _, s_channel, v_channel = cv2.split(hsv)

        glare_pixels = np.logical_and(v_channel > 242, s_channel < 25)
        glare_ratio = float(np.mean(glare_pixels))

        gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (128, 128), interpolation=cv2.INTER_AREA)
        f_shift = np.fft.fftshift(np.fft.fft2(resized))
        magnitude = 20.0 * np.log(np.abs(f_shift) + 1e-8)

        center = 64
        magnitude[center - 12 : center + 12, center - 12 : center + 12] = 0.0
        high_freq_energy = float(np.mean(magnitude))

        glare_penalty = min(1.0, glare_ratio / 0.08)
        moire_penalty = max(0.0, min(1.0, (high_freq_energy - 135.0) / 40.0))

        reflection_score = max(
            0.0, 1.0 - (0.6 * glare_penalty + 0.4 * moire_penalty)
        )
        return round(reflection_score, 4)

    def _analyze_motion_consistency(
        self,
        frames: list[np.ndarray],
        bboxes: list[tuple[int, int, int, int]],
    ) -> float:
        """
        Compares inner-face optical flow against global background flow to
        filter out handheld camera shake and flat photo translation.
        """
        if len(frames) < 2:
            return 1.0

        relative_metrics: list[tuple[float, float]] = []
        for idx in range(len(frames) - 1):
            prev_gray = cv2.cvtColor(frames[idx], cv2.COLOR_BGR2GRAY)
            curr_gray = cv2.cvtColor(frames[idx + 1], cv2.COLOR_BGR2GRAY)

            h_orig, w_orig = prev_gray.shape
            scale_x = 320.0 / w_orig
            scale_y = 240.0 / h_orig

            prev_small = cv2.resize(prev_gray, (320, 240))
            curr_small = cv2.resize(curr_gray, (320, 240))

            flow = cv2.calcOpticalFlowFarneback(
                prev_small, curr_small, None, 0.5, 3, 15, 3, 5, 1.2, 0
            )

            x1, y1, w1, h1 = bboxes[idx]
            x2, y2, w2, h2 = bboxes[idx + 1]
            ux = int(min(x1, x2) * scale_x)
            uy = int(min(y1, y2) * scale_y)
            uw = max(1, int((max(x1 + w1, x2 + w2) - min(x1, x2)) * scale_x))
            uh = max(1, int((max(y1 + h1, y2 + h2) - min(y1, y2)) * scale_y))

            bg_mask = np.ones((240, 320), dtype=bool)
            bg_mask[uy : uy + uh, ux : ux + uw] = False
            if np.any(bg_mask):
                bg_flow_x = float(np.median(flow[..., 0][bg_mask]))
                bg_flow_y = float(np.median(flow[..., 1][bg_mask]))
            else:
                bg_flow_x, bg_flow_y = 0.0, 0.0

            face_flow_x = flow[uy : uy + uh, ux : ux + uw, 0] - bg_flow_x
            face_flow_y = flow[uy : uy + uh, ux : ux + uw, 1] - bg_flow_y
            rel_mag = np.hypot(face_flow_x, face_flow_y)

            if rel_mag.size == 0:
                continue

            mean_rel_motion = float(np.mean(rel_mag))
            non_rigid_variance = float(np.std(rel_mag))
            relative_metrics.append((mean_rel_motion, non_rigid_variance))

        if not relative_metrics:
            return 0.0

        avg_motion = float(np.mean([m[0] for m in relative_metrics]))
        avg_non_rigid = float(np.mean([m[1] for m in relative_metrics]))

        if avg_motion < 0.15:
            return 0.25
        if avg_non_rigid < 0.12:
            return 0.35

        return round(min(1.0, 0.5 + (avg_non_rigid / 1.5)), 4)

    def evaluate(self, frames: list[np.ndarray]) -> dict[str, Any]:
        """
        Runs liveness verification on a single frame or short sequence
        (2-5 frames) and returns a PASS/FAIL verdict with actionable guidance.
        """
        is_sequence = len(frames) > 1
        flatness_scores: list[float] = []
        reflection_scores: list[float] = []
        bboxes: list[tuple[int, int, int, int]] = []

        for frame in frames:
            f_score, f_meta = self._analyze_flatness_and_spoof(frame)
            if "error" in f_meta:
                return {
                    "status": "FAIL",
                    "is_live": False,
                    "liveness_score": 0.0,
                    "mode": "sequence" if is_sequence else "single",
                    "failed_checks": ["no_face"],
                    "user_guidance": [GUIDANCE_MAP["no_face"]],
                }

            r_score = self._analyze_reflection(f_meta["face_crop"])
            flatness_scores.append(f_score)
            reflection_scores.append(r_score)
            bboxes.append(f_meta["bbox"])

        avg_flatness = round(float(np.mean(flatness_scores)), 4)
        avg_reflection = round(float(np.mean(reflection_scores)), 4)

        failed_checks: list[str] = []
        if avg_flatness < 0.65:
            failed_checks.append("flatness")
        if avg_reflection < 0.65:
            failed_checks.append("reflection")

        if is_sequence:
            motion_score = self._analyze_motion_consistency(frames, bboxes)
            if motion_score < 0.60:
                failed_checks.append("motion_inconsistency")
            final_score = round(
                (0.45 * avg_flatness)
                + (0.25 * avg_reflection)
                + (0.30 * motion_score),
                4,
            )
        else:
            motion_score = None
            final_score = round(
                (0.65 * avg_flatness) + (0.35 * avg_reflection), 4
            )

        passed = final_score >= self.threshold and len(failed_checks) == 0

        if not passed and not failed_checks:
            candidates: dict[str, float] = {
                "flatness": avg_flatness,
                "reflection": avg_reflection,
            }
            if motion_score is not None:
                candidates["motion_inconsistency"] = motion_score
            weakest = min(candidates, key=candidates.__getitem__)
            failed_checks.append(weakest)

        guidance = (
            ["Liveness check passed."]
            if passed
            else [GUIDANCE_MAP[code] for code in failed_checks]
        )

        return {
            "status": "PASS" if passed else "FAIL",
            "is_live": passed,
            "liveness_score": final_score,
            "mode": "sequence" if is_sequence else "single",
            "scores": {
                "flatness": avg_flatness,
                "reflection": avg_reflection,
                "motion_consistency": motion_score,
            },
            "failed_checks": failed_checks,
            "user_guidance": guidance,
        }