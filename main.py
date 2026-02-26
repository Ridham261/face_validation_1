"""
Production-Grade Multi-Layer Facial Validation & Image Screening System
========================================================================
Architecture : MTCNN + MediaPipe FaceDetection + MediaPipe FaceMesh
               + OpenCV SolvePnP + Weighted Scoring

Validation Layers (Priority Order):
  1. No Human Detection
  2. Irrelevant Image Detection
  3. Multiple Person Detection  (max 1 face allowed)
  4. Dark Goggle Detection      (S1-S4 vote 4/4, S5 hard veto)
  5. Closed Eyes Detection      (EAR method)
  6. Pose Classification        (Frontal / Side)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Fixes Applied — v2.5.0  (full mathematical + flow audit)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

BUG-01 [CRITICAL] Coordinate space mismatch in _match_landmarks_to_faces.
  MTCNN bboxes are in original-image pixel space; after FIX-21 downscaling,
  MediaPipe landmark coords are in proc_frame space.  On a 4K image (scale=0.5)
  the bbox centre appeared ~1000 px from the nose-tip → wrong face-landmark
  pairing for every resized image.
  Fix: multiply bbox centre by proc_scale before computing distance.

BUG-02 [HIGH] EAR closed-eye logic used average(L,R) < threshold.
  A person with ONE narrow/drooping eye (ptosis, occlusion) lowered the
  average even when the other eye was fully open.
  Fix: eyes_closed only when BOTH individual EARs are below threshold.

BUG-03 [HIGH] EAR resolution guard used scaled_bbox (proc space) but
  EAR_MIN_RESOLUTION=80 px was calibrated for original resolution.
  A 160-px original face downscaled to 80 px was borderline and skipped.
  Fix: pass original (unscaled) bbox to detect_closed_eyes for the size guard.

BUG-04 [MEDIUM] NO_HUMAN_CONFIDENCE_MIN=0.80 was defined but never used.
  detect_no_human only saw faces already filtered at 0.90.  Removed the
  dead constant and clarified the single-filter-chain design.

BUG-05 [MEDIUM] S3 yaw fallback used s3 = s1*0.5, double-counting the MTCNN
  S1 signal through the highest-weight slot (x2.5) without actual yaw data.
  Fix: when yaw is unavailable, S3 abstains (s3=0, not added to dynamic_max).

BUG-06 [MEDIUM] dynamic_max for the S3 fallback inflated the confidence
  denominator, making confidence misleading when yaw was missing.
  Fix: resolved by BUG-05 (unavailable slot excluded from dynamic_max).

BUG-09 [LOW] SG_CONTRAST_MAX=18 was too tight; dark-tinted (not fully opaque)
  sunglasses (std~20-25) would pass S2 even though they should be caught.
  Fix: raised SG_CONTRAST_MAX to 22.

BUG-10 [LOW] SolvePnP used SOLVEPNP_ITERATIVE which can diverge at yaw>60°.
  Fix: replaced with SOLVEPNP_SQPNP (globally convergent).

BUG-11 [LOW] face_id used local_idx+1; non-contiguous when faces skipped.
  Fix: separate face_counter that increments only for processed faces.

NOTE-01 [CLEANUP] _get_nose_visibility_ratio multiplied by img_w in both
  numerator and denominator (cancelled to a pure ratio).  Removed redundancy;
  function now uses normalised [0,1] coords directly.

NOTE-02 [PERF] detect_persons_and_size now receives proc_frame for speed.

Inherited fixes v2.3.0/v2.4.0:
  FIX-01..FIX-21 (coordinate guards, thread safety, iris veto, flip guard, etc.)

Deployment note:
  threading.local() is safe for Uvicorn/Gunicorn multi-process workers.
"""

import os
import io
import cv2
import math
import time
import logging
import threading
import numpy as np
import torch
import uvicorn
import mediapipe as mp

from contextlib import asynccontextmanager
from datetime import datetime
from PIL import Image
from facenet_pytorch import MTCNN
from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.responses import JSONResponse, FileResponse
from typing import Optional

import asyncio, io, os, tempfile, time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import aiohttp, pandas as pd

# ─────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger("FaceValidationEngine")

# ─────────────────────────────────────────────────────────────
# Device
# ─────────────────────────────────────────────────────────────
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
logger.info(f"Running on device: {device}")

# ─────────────────────────────────────────────────────────────
# Output folders
# ─────────────────────────────────────────────────────────────
BASE_OUTPUT = "detection_outputs"
CATEGORY_FOLDERS = {
    "front":           os.path.join(BASE_OUTPUT, "front"),
    "side":            os.path.join(BASE_OUTPUT, "side"),
    "mixed":           os.path.join(BASE_OUTPUT, "mixed"),
    "eyes_closed":     os.path.join(BASE_OUTPUT, "eyes_closed"),
    "multiple_person": os.path.join(BASE_OUTPUT, "multiple_person"),
    "sunglasses":      os.path.join(BASE_OUTPUT, "sunglasses"),
    "no_human":        os.path.join(BASE_OUTPUT, "no_human"),
    "irrelevant":      os.path.join(BASE_OUTPUT, "irrelevant"),
    "no_landmarks":    os.path.join(BASE_OUTPUT, "no_landmarks"),
}
for _folder in CATEGORY_FOLDERS.values():
    os.makedirs(_folder, exist_ok=True)

# ─────────────────────────────────────────────────────────────
# Constants & thresholds
# ─────────────────────────────────────────────────────────────

# SolvePnP 3-D reference model (generic head, millimetre scale)
# Order: nose-tip, chin, left-eye-corner, right-eye-corner,
#        left-mouth-corner, right-mouth-corner
MODEL_POINTS_3D = np.array([
    (   0.0,    0.0,    0.0),
    (   0.0, -330.0,  -65.0),
    (-225.0,  170.0, -135.0),
    ( 225.0,  170.0, -135.0),
    (-150.0, -150.0, -125.0),
    ( 150.0, -150.0, -125.0),
], dtype=np.float64)

# MediaPipe FaceMesh landmark indices
NOSE_TIP     = 1
CHIN         = 152
LEFT_EYE_MP  = 263
RIGHT_EYE_MP = 33
LEFT_MOUTH   = 287
RIGHT_MOUTH  = 57
NOSE_LEFT    = 279
NOSE_RIGHT   = 49
LEFT_CHEEK   = 234
RIGHT_CHEEK  = 454

# EAR landmarks — Soukupova-Cech order:
# [outer, upper-outer, upper-inner, inner, lower-inner, lower-outer]
LEFT_EYE_EAR  = [362, 385, 387, 263, 373, 380]
RIGHT_EYE_EAR = [33,  160, 158, 133, 153, 144]

# Sunglasses eye-polygon landmarks
SG_LEFT_EYE_POLY  = [33,  133, 160, 159, 158, 144, 145, 153]
SG_RIGHT_EYE_POLY = [362, 263, 387, 386, 385, 380, 381, 373]

# Detection / filtering
MTCNN_CONFIDENCE_THRESHOLD = 0.90
# BUG-04: NO_HUMAN_CONFIDENCE_MIN removed — single filter chain at 0.90
MIN_FACE_SIZE_PX           = 1     # minimum bbox side (original px space)

# Image dimension limits
IMG_MIN_DIM            = 1
IMG_MAX_DIM            = 8000
IMG_PROCESSING_MAX_DIM = 1600   # downscale target for all MediaPipe ops

# Batch limit
BATCH_MAX_FILE_BYTES = 25 * 1024 * 1024   # 25 MB

# Pose thresholds
YAW_FRONTAL_MAX       = 20.0
YAW_SLIGHTLY_SIDE_MAX = 35.0
ASYM_FRONTAL_MAX      = 12.0
ASYM_SLIGHTLY_MAX     = 22.0
MTCNN_FRONTAL_R_MIN   = 30
MTCNN_FRONTAL_R_MAX   = 62
MTCNN_FRONTAL_L_MIN   = 30
MTCNN_FRONTAL_L_MAX   = 63
NOSE_VISIBILITY_MIN   = 0.08

# EAR
EAR_CLOSED_THRESHOLD = 0.22  # individual-eye threshold 
EAR_MIN_RESOLUTION   = 80    # min face height in ORIGINAL px 

# Sunglasses
SG_BRIGHTNESS_MAX    = 50
SG_CONTRAST_MAX      = 22    
SG_HSV_V_MAX         = 55
SG_DARK_COVERAGE_MIN = 0.70
SG_IRIS_CONTRAST_MIN = 25
SG_IRIS_BRIGHT_MIN   = 55
SG_VOTE_THRESHOLD    = 4

# Drawing
LINE_COLOR = (255, 255, 0)
FONT       = cv2.FONT_HERSHEY_PLAIN
FONT_SCALE = 1
FONT_THICK = 2

# HTTP status codes
CATEGORY_STATUS_MAP = {
    "front":           200,
    "side":            400,
    "mixed":           400,
    "eyes_closed":     422,
    "sunglasses":      423,
    "multiple_person": 409,
    "no_human":        404,
    "irrelevant":      415,
    "no_landmarks":    422,
}

# ─────────────────────────────────────────────────────────────
# Thread-safe MediaPipe instances
# ─────────────────────────────────────────────────────────────
_thread_local            = threading.local()
mp_face_detection_module = mp.solutions.face_detection
mp_face_mesh_module      = mp.solutions.face_mesh


def _get_face_detector():
    if not hasattr(_thread_local, "face_detector"):
        _thread_local.face_detector = mp_face_detection_module.FaceDetection(
            model_selection=1, min_detection_confidence=0.70
        )
    return _thread_local.face_detector


def _get_face_mesh():
    if not hasattr(_thread_local, "face_mesh"):
        _thread_local.face_mesh = mp_face_mesh_module.FaceMesh(
            static_image_mode=True, max_num_faces=10, refine_landmarks=True,
            min_detection_confidence=0.5, min_tracking_confidence=0.5,
        )
    return _thread_local.face_mesh


# ─────────────────────────────────────────────────────────────
# MTCNN global
# ─────────────────────────────────────────────────────────────
mtcnn = MTCNN(
    image_size=160, margin=0, min_face_size=20,
    thresholds=[0.6, 0.7, 0.7], factor=0.709,
    post_process=True, keep_all=True, device=device,
)

# ─────────────────────────────────────────────────────────────
# Lifespan
# ─────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app_: "FastAPI"):
    logger.info("Face Validation Engine v2.5.0 starting.")
    yield
    logger.info("Face Validation Engine shut down.")

app = FastAPI(
    title="Production Face Validation Engine",
    description="Multi-layer facial validation: pose · eyes · sunglasses · liveness",
    version="2.5.0",
    lifespan=lifespan,
)


# ═══════════════════════════════════════════════════════════════
# MODULE 0 — Image pre-processing
# ═══════════════════════════════════════════════════════════════

def validate_image_dimensions(frame_cv2: np.ndarray) -> Optional[str]:
    h, w = frame_cv2.shape[:2]
    if h < IMG_MIN_DIM or w < IMG_MIN_DIM:
        return f"Image too small: {w}x{h}px (min {IMG_MIN_DIM}px per side)"
    if h > IMG_MAX_DIM or w > IMG_MAX_DIM:
        return f"Image too large: {w}x{h}px (max {IMG_MAX_DIM}px per side)"
    return None


def _resize_for_processing(frame_cv2: np.ndarray) -> tuple:
    """
    Proportionally downscale so longest side <= IMG_PROCESSING_MAX_DIM.
    Returns (proc_frame, proc_scale).   proc_scale=1.0 means no resize.

    Coordinate contract:
      original_coord * proc_scale  = proc_coord
      proc_coord    / proc_scale   = original_coord
    """
    h, w    = frame_cv2.shape[:2]
    max_dim = max(h, w)
    if max_dim <= IMG_PROCESSING_MAX_DIM:
        return frame_cv2, 1.0
    scale   = IMG_PROCESSING_MAX_DIM / max_dim
    new_w   = max(1, int(w * scale))
    new_h   = max(1, int(h * scale))
    resized = cv2.resize(frame_cv2, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return resized, scale


# ═══════════════════════════════════════════════════════════════
# MODULE 1 — No Human / Irrelevant
# ═══════════════════════════════════════════════════════════════

def _skin_ratio(frame_cv2: np.ndarray) -> float:
    """Fraction of pixels that match skin-tone HSV ranges (light and dark skin)."""
    hsv      = cv2.cvtColor(frame_cv2, cv2.COLOR_BGR2HSV)
    total_px = frame_cv2.shape[0] * frame_cv2.shape[1]
    mask1    = cv2.inRange(hsv, np.array([0, 20, 60], np.uint8),
                                np.array([25, 255, 255], np.uint8))
    mask2    = cv2.inRange(hsv, np.array([0, 10, 30], np.uint8),
                                np.array([30, 180, 180], np.uint8))
    return float(np.sum(cv2.bitwise_or(mask1, mask2) > 0)) / (total_px + 1e-8)


def detect_no_human(valid_bboxes: list, valid_probs: list) -> dict:
    """Called AFTER size+confidence filtering (BUG-04: single filter chain)."""
    if not valid_bboxes:
        return {"is_human": False, "reason": "No face passed filter",
                "max_confidence": 0.0}
    probs = [float(p) for p in valid_probs if p is not None]
    if not probs:
        return {"is_human": False, "reason": "All detections returned null confidence",
                "max_confidence": 0.0}
    return {"is_human": True, "reason": "Human face detected",
            "max_confidence": round(max(probs), 4)}


def detect_irrelevant_image(valid_bboxes: list, frame_cv2: np.ndarray) -> dict:
    """
    irrelevant : no face AND skin < 3%
    no_human   : no face BUT skin >= 3%
    """
    if valid_bboxes:
        return {"is_irrelevant": False, "reason": "Face detected"}
    skin = _skin_ratio(frame_cv2)
    if skin < 0.03:
        return {"is_irrelevant": True,
                "reason": f"No face + very low skin coverage ({skin:.3f})",
                "skin_ratio": round(skin, 4)}
    return {"is_irrelevant": False,
            "reason": f"No face but skin present ({skin:.3f}) -> no_human",
            "skin_ratio": round(skin, 4)}


# ═══════════════════════════════════════════════════════════════
# MODULE 2 — Multiple person detection
# ═══════════════════════════════════════════════════════════════

def detect_persons_and_size(proc_frame: np.ndarray) -> dict:
    """NOTE-02: uses proc_frame. FIX-04: triggers on mp_count > 1 only."""
    face_detector = _get_face_detector()
    rgb           = cv2.cvtColor(proc_frame, cv2.COLOR_BGR2RGB)
    result        = face_detector.process(rgb)
    if not result.detections:
        return {"valid": True, "total_faces": 0,
                "category": "ok", "reason": "No detections — deferred to MTCNN"}
    total = len(result.detections)
    if total > 1:
        return {"valid": False, "total_faces": total,
                "category": "multiple_person",
                "reason": f"{total} persons detected. Only one allowed."}
    return {"valid": True, "total_faces": 1,
            "category": "ok", "reason": "Single face detected."}


# ═══════════════════════════════════════════════════════════════
# MODULE 3 — Sunglasses / dark-goggle detection
# ═══════════════════════════════════════════════════════════════

def _extract_eye_roi(proc_frame: np.ndarray, face_lm, indices: list):
    """Returns (gray, hsv) for the eye polygon, or (None, None) if too small."""
    h, w = proc_frame.shape[:2]
    lm   = face_lm.landmark
    xs   = [int(lm[i].x * w) for i in indices]
    ys   = [int(lm[i].y * h) for i in indices]
    x1, x2 = max(0, min(xs) - 4), min(w, max(xs) + 4)
    y1, y2 = max(0, min(ys) - 4), min(h, max(ys) + 4)
    roi    = proc_frame[y1:y2, x1:x2]
    if roi.size < 16:
        return None, None
    return cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY), cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)


def detect_sunglasses(proc_frame: np.ndarray, face_landmarks) -> dict:
    """
    S5 (iris guard) — hard veto if BOTH eyes show iris (contrast>25 & brightness>55).
    S1–S4 vote 4/4:
      S1 mean grey   < 50  (SG_BRIGHTNESS_MAX)
      S2 std  grey   < 22  (SG_CONTRAST_MAX — raised from 18, BUG-09)
      S3 HSV-V mean  < 55  (SG_HSV_V_MAX)
      S4 dark pixels >= 0.70 (SG_DARK_COVERAGE_MIN)
    Inconclusive ROI -> that eye skipped (not forced to "clear").
    Single-eye detections flagged with warning.
    """
    if face_landmarks is None:
        return {"detected": False, "reason": "No landmarks", "votes": 0, "details": {}}

    details = {}

    l_gray, l_hsv = _extract_eye_roi(proc_frame, face_landmarks, SG_LEFT_EYE_POLY)
    r_gray, r_hsv = _extract_eye_roi(proc_frame, face_landmarks, SG_RIGHT_EYE_POLY)

    def _eye_stats(gray, hsv, side):
        if gray is None:
            details[f"{side}_roi"] = "inconclusive"
            return None, None, None, None
        b  = round(float(np.mean(gray)), 2)
        c  = round(float(np.std(gray)),  2)
        v  = round(float(np.mean(hsv[:, :, 2])), 2)
        dc = round(float(np.sum(gray < 60)) / (gray.size + 1e-8), 4)
        details.update({f"{side}_brightness": b, f"{side}_contrast": c,
                         f"{side}_v_channel": v, f"{side}_dark_coverage": dc})
        return b, c, v, dc

    l_b, l_c, l_v, l_dc = _eye_stats(l_gray, l_hsv, "left")
    r_b, r_c, r_v, r_dc = _eye_stats(r_gray, r_hsv, "right")

    # S5: iris guard — hard veto (both eyes must confirm)
    iris_l = (l_c is not None and l_c > SG_IRIS_CONTRAST_MIN
              and l_b is not None and l_b > SG_IRIS_BRIGHT_MIN)
    iris_r = (r_c is not None and r_c > SG_IRIS_CONTRAST_MIN
              and r_b is not None and r_b > SG_IRIS_BRIGHT_MIN)
    if iris_l and iris_r:
        details["s5_iris_guard"] = "iris_visible_both_eyes — hard veto"
        return {"detected": False, "votes": 0,
                "reason": "Iris visible in both eyes (clear/prescription glass or bare)",
                "details": details}
    details["s5_iris_guard"] = f"iris NOT confirmed both eyes (L={iris_l}, R={iris_r})"

    # Measurability
    measurable = sum(1 for b in (l_b, r_b) if b is not None)
    if measurable == 0:
        return {"detected": False, "votes": 0,
                "reason": "Both eye ROIs inconclusive",
                "details": details}
    single_eye_only = (measurable == 1)

    def _passes(vl, vr, thr, op):
        results = [v < thr if op == "lt" else v >= thr
                   for v in (vl, vr) if v is not None]
        return bool(results) and all(results)

    votes = 0
    if _passes(l_b, r_b, SG_BRIGHTNESS_MAX, "lt"):
        votes += 1; details["s1_brightness"] = f"dark (L={l_b}, R={r_b})"
    else:
        details["s1_brightness"] = f"clear (L={l_b}, R={r_b})"

    if _passes(l_c, r_c, SG_CONTRAST_MAX, "lt"):
        votes += 1; details["s2_contrast"] = f"dark (L={l_c}, R={r_c})"
    else:
        details["s2_contrast"] = f"clear (L={l_c}, R={r_c})"

    if _passes(l_v, r_v, SG_HSV_V_MAX, "lt"):
        votes += 1; details["s3_hsv_v"] = f"dark (L={l_v}, R={r_v})"
    else:
        details["s3_hsv_v"] = f"clear (L={l_v}, R={r_v})"

    if _passes(l_dc, r_dc, SG_DARK_COVERAGE_MIN, "ge"):
        votes += 1; details["s4_dark_coverage"] = f"dark (L={l_dc}, R={r_dc})"
    else:
        details["s4_dark_coverage"] = f"clear (L={l_dc}, R={r_dc})"

    details.update({"votes": votes, "measurable_eyes": measurable,
                    "single_eye_only": single_eye_only})

    detected = votes >= SG_VOTE_THRESHOLD
    suffix   = " [single-eye — low confidence]" if (detected and single_eye_only) else ""
    reason   = (f"Dark goggles detected ({votes}/4 signals){suffix}"
                if detected else f"No dark goggles ({votes}/4 signals)")
    return {"detected": detected, "reason": reason, "details": details}


# ═══════════════════════════════════════════════════════════════
# MODULE 4 — Closed-eye detection  (EAR)
# ═══════════════════════════════════════════════════════════════

def _compute_ear(landmarks_2d: list, eye_idx: list) -> Optional[float]:
    """
    EAR = (||p1-p5|| + ||p2-p4||) / (2 * ||p0-p3||)
    Returns None on failure (callers must handle None explicitly).
    """
    try:
        pts = [landmarks_2d[i] for i in eye_idx]
        v1  = np.linalg.norm(np.array(pts[1]) - np.array(pts[5]))
        v2  = np.linalg.norm(np.array(pts[2]) - np.array(pts[4]))
        h   = np.linalg.norm(np.array(pts[0]) - np.array(pts[3]))
        return round(float((v1 + v2) / (2.0 * h)), 4) if h > 0 else None
    except Exception:
        return None


def detect_closed_eyes(
    face_landmarks,
    proc_w: int,
    proc_h: int,
    orig_bbox,     # BUG-03: original-space bbox — for resolution guard only
    proc_bbox,     # proc-space bbox  (kept for completeness, not used in math)
) -> dict:
    """
    BUG-02: eyes_closed = True only when BOTH individual EARs < threshold.
    BUG-03: resolution guard uses original-space bbox height vs EAR_MIN_RESOLUTION.
    """
    result = {"eyes_closed": False, "left_ear": None, "right_ear": None,
              "avg_ear": None, "reason": "Not checked", "checked": False}

    if face_landmarks is None:
        result["reason"] = "MediaPipe landmarks unavailable"
        return result

    # BUG-03: guard in original pixel space
    if orig_bbox is not None:
        x1, y1, x2, y2 = orig_bbox
        orig_face_h = abs(y2 - y1)
        if orig_face_h < EAR_MIN_RESOLUTION:
            result["reason"] = (
                f"Face too small in original space "
                f"({orig_face_h:.0f}px < {EAR_MIN_RESOLUTION}px)"
            )
            return result

    lm           = face_landmarks.landmark
    landmarks_2d = [(lm[i].x * proc_w, lm[i].y * proc_h) for i in range(len(lm))]
    left_ear     = _compute_ear(landmarks_2d, LEFT_EYE_EAR)
    right_ear    = _compute_ear(landmarks_2d, RIGHT_EYE_EAR)

    result.update({"left_ear": left_ear, "right_ear": right_ear, "checked": True})

    if left_ear is None or right_ear is None:
        result["reason"] = (
            f"EAR computation failed (L={left_ear}, R={right_ear}) — treating as open"
        )
        return result

    avg_ear           = round((left_ear + right_ear) / 2.0, 4)
    result["avg_ear"] = avg_ear

    # BUG-02: both-eye requirement
    if left_ear < EAR_CLOSED_THRESHOLD and right_ear < EAR_CLOSED_THRESHOLD:
        result["eyes_closed"] = True
        result["reason"] = (
            f"Both eyes closed: L={left_ear:.3f}, R={right_ear:.3f} "
            f"< {EAR_CLOSED_THRESHOLD}"
        )
    else:
        open_eye = "left" if left_ear >= EAR_CLOSED_THRESHOLD else "right"
        result["reason"] = (
            f"At least one eye open ({open_eye}): "
            f"L={left_ear:.3f}, R={right_ear:.3f}, avg={avg_ear:.3f}"
        )
    return result


# ═══════════════════════════════════════════════════════════════
# MODULE 5 — Head-pose estimation
# ═══════════════════════════════════════════════════════════════

def _np_angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """Angle in degrees at vertex b between rays ba and bc."""
    ba    = a - b;  bc = c - b
    denom = np.linalg.norm(ba) * np.linalg.norm(bc)
    if denom < 1e-8:
        return 0.0
    return float(np.degrees(np.arccos(np.clip(np.dot(ba, bc) / denom, -1.0, 1.0))))


def _estimate_focal_length(img_w: int, img_h: int) -> float:
    """
    Focal length in pixels via 70-degree diagonal-FOV heuristic.
      f = sqrt(w^2+h^2) / (2 * tan(FOV_diag/2))
    Diagonal is rotation-invariant: same f for portrait and landscape.
    """
    diag = math.sqrt(img_w ** 2 + img_h ** 2)
    return diag / (2.0 * math.tan(math.radians(70.0) / 2.0))


def _get_mediapipe_angles(face_landmarks, img_w: int, img_h: int) -> Optional[dict]:
    """
    Head yaw/pitch/roll via SolvePnP + ZYX Euler decomposition.

    Camera model: diagonal-FOV focal length, principal point at image centre,
    zero distortion.

    180-deg flip guard (FIX-13c):
      rm[:,2] is the face forward-Z in camera space.
      Normal: forward[2] < 0 (Z points into scene, away from camera).
      Flipped: forward[2] > 0  ->  negate rm.

    BUG-10: SOLVEPNP_SQPNP (globally convergent) replaces SOLVEPNP_ITERATIVE.

    ZYX Euler decomposition:
      sy  = sqrt(R[0,0]^2 + R[1,0]^2)
      if sy > eps:
        pitch = atan2( R[2,1], R[2,2])
        yaw   = atan2(-R[2,0], sy)
        roll  = atan2( R[1,0], R[0,0])
      else (gimbal lock, pitch ~ +-90 deg):
        pitch = atan2(-R[1,2], R[1,1])
        yaw   = atan2(-R[2,0], sy)  # -> ~ +-90 deg
        roll  = 0  (lost at gimbal lock — standard behaviour)
    """
    try:
        lm  = face_landmarks.landmark
        pts = np.array([
            (lm[NOSE_TIP].x     * img_w, lm[NOSE_TIP].y     * img_h),
            (lm[CHIN].x         * img_w, lm[CHIN].y          * img_h),
            (lm[LEFT_EYE_MP].x  * img_w, lm[LEFT_EYE_MP].y  * img_h),
            (lm[RIGHT_EYE_MP].x * img_w, lm[RIGHT_EYE_MP].y * img_h),
            (lm[LEFT_MOUTH].x   * img_w, lm[LEFT_MOUTH].y   * img_h),
            (lm[RIGHT_MOUTH].x  * img_w, lm[RIGHT_MOUTH].y  * img_h),
        ], dtype=np.float64)

        fl  = _estimate_focal_length(img_w, img_h)
        cm  = np.array([[fl, 0,  img_w / 2],
                        [0,  fl, img_h / 2],
                        [0,  0,  1.0      ]], dtype=np.float64)

        # BUG-10: globally convergent solver
        ok, rvec, _tvec = cv2.solvePnP(
            MODEL_POINTS_3D, pts, cm, np.zeros((4, 1)),
            flags=cv2.SOLVEPNP_SQPNP,
        )
        if not ok:
            return None

        rm, _ = cv2.Rodrigues(rvec)

        # FIX-13c: flip guard via forward-axis sign of rotation matrix
        if rm[2, 2] > 0:
            rm = -rm

        sy = math.sqrt(rm[0, 0] ** 2 + rm[1, 0] ** 2)
        if sy > 1e-6:
            pitch = math.atan2( rm[2, 1], rm[2, 2])
            yaw   = math.atan2(-rm[2, 0], sy)
            roll  = math.atan2( rm[1, 0], rm[0, 0])
        else:   # gimbal lock
            pitch = math.atan2(-rm[1, 2], rm[1, 1])
            yaw   = math.atan2(-rm[2, 0], sy)
            roll  = 0.0

        return {
            "yaw":   round(math.degrees(yaw),   2),
            "pitch": round(math.degrees(pitch),  2),
            "roll":  round(math.degrees(roll),   2),
        }
    except Exception:
        return None


def _get_nose_visibility_ratio(face_landmarks) -> float:
    """
    nose-wing width / cheek-to-cheek width — purely normalised ratio.
    NOTE-01: removed redundant *img_w (cancelled in numerator/denominator).
    ~0.25-0.30 frontal; < 0.08 full side.
    """
    try:
        lm = face_landmarks.landmark
        nw = abs(lm[NOSE_RIGHT].x - lm[NOSE_LEFT].x)
        fw = abs(lm[RIGHT_CHEEK].x - lm[LEFT_CHEEK].x)
        return round(nw / fw, 4) if fw > 1e-8 else 1.0
    except Exception:
        return 1.0


def classify_pose_strong(
    angR: float,
    angL: float,
    yaw_data: Optional[dict],
    nose_ratio: float,
) -> tuple:
    """
    Weighted 4-signal pose classifier.

    Weights:  S1 x1.0  S2 x1.5  S3 x2.5  S4 x1.5   (theoretical max ±6.5)

    BUG-05: S3 abstains when yaw unavailable (was s1*0.5 phantom penalty).
    BUG-06: unavailable S3 slot excluded from dynamic_max.
    FIX-07: classification boundary score >= 0 -> Frontal.

    confidence = |total_score| / dynamic_max * 100
    """
    total_score = 0.0
    dynamic_max = 0.0
    signals     = {}

    # S1 — MTCNN eye-to-nose angle range
    ri, li = int(angR), int(angL)
    if (MTCNN_FRONTAL_R_MIN <= ri <= MTCNN_FRONTAL_R_MAX and
            MTCNN_FRONTAL_L_MIN <= li <= MTCNN_FRONTAL_L_MAX):
        s1 = 1.0;  signals["mtcnn_angle"] = "Frontal"
    elif 25 <= ri <= 68 and 25 <= li <= 70:
        s1 = 0.3;  signals["mtcnn_angle"] = "Slightly Side"
    else:
        s1 = -1.0; signals["mtcnn_angle"] = "Side"
    total_score += s1 * 1.0
    dynamic_max += 1.0

    # S2 — Eye-angle asymmetry
    asym = abs(angR - angL)
    if asym <= ASYM_FRONTAL_MAX:
        s2 = 1.0;  signals["asymmetry"] = f"Frontal ({asym:.1f}deg)"
    elif asym <= ASYM_SLIGHTLY_MAX:
        s2 = 0.3;  signals["asymmetry"] = f"Slightly Side ({asym:.1f}deg)"
    else:
        s2 = -1.0; signals["asymmetry"] = f"Side ({asym:.1f}deg)"
    total_score += s2 * 1.5
    dynamic_max += 1.5

    # S3 — MediaPipe yaw  (highest weight; abstains when unavailable)
    if yaw_data is not None:
        yaw_abs = abs(yaw_data["yaw"])
        signals.update(yaw_data)
        if yaw_abs <= YAW_FRONTAL_MAX:
            s3 = 1.0;  signals["mediapipe_yaw"] = f"Frontal ({yaw_abs}deg)"
        elif yaw_abs <= YAW_SLIGHTLY_SIDE_MAX:
            decay = 1.0 - ((yaw_abs - YAW_FRONTAL_MAX) /
                            (YAW_SLIGHTLY_SIDE_MAX - YAW_FRONTAL_MAX))
            s3 = max(0.1, decay * 0.8)
            signals["mediapipe_yaw"] = f"Slightly Side ({yaw_abs}deg, s3={s3:.3f})"
        else:
            s3 = -1.0; signals["mediapipe_yaw"] = f"Full Side ({yaw_abs}deg)"
        total_score += s3 * 2.5
        dynamic_max += abs(s3) * 2.5
    else:
        # BUG-05: abstain — no phantom penalty, no dynamic_max contribution
        signals["mediapipe_yaw"] = "unavailable — S3 abstained"

    # S4 — Nose visibility ratio
    signals["nose_visibility_ratio"] = nose_ratio
    if nose_ratio >= 0.18:
        s4 = 1.0;  signals["nose_visibility"] = f"Frontal ({nose_ratio:.4f})"
    elif nose_ratio >= NOSE_VISIBILITY_MIN:
        s4 = 0.2;  signals["nose_visibility"] = f"Slightly Side ({nose_ratio:.4f})"
    else:
        s4 = -1.0; signals["nose_visibility"] = f"Full Side ({nose_ratio:.4f})"
    total_score += s4 * 1.5
    dynamic_max += 1.5

    final    = "Frontal" if total_score >= 0 else "Side"
    conf_pct = (abs(total_score) / dynamic_max * 100.0) if dynamic_max > 0 else 0.0
    signals.update({
        "total_score": round(total_score, 4),
        "max_score":   round(dynamic_max, 4),
        "confidence":  f"{round(conf_pct, 1)}%",
    })
    return final, signals


# ═══════════════════════════════════════════════════════════════
# MODULE 6 — MTCNN <-> MediaPipe face matching
# ═══════════════════════════════════════════════════════════════

def _match_landmarks_to_faces(
    orig_bboxes: list,
    mp_faces:    list,
    proc_w:      int,
    proc_h:      int,
    proc_scale:  float,
) -> list:
    """
    Matches each MTCNN bbox (original space) to the closest MediaPipe face
    by Euclidean distance to the nose-tip landmark.

    BUG-01 fix: bbox centre is multiplied by proc_scale to convert it to
    proc-frame space before computing distance.  Without this, on a 4K image
    (scale=0.5) the centre was ~1000 px from the correct nose-tip.

    Distance formula:
      d = sqrt( (lm.x*proc_w  -  cx_orig*proc_scale)^2
              + (lm.y*proc_h  -  cy_orig*proc_scale)^2 )
    """
    matched = []
    for bbox in orig_bboxes:
        if bbox is None:
            matched.append(None)
            continue
        bx1, by1, bx2, by2 = bbox
        # BUG-01: convert original-space centre -> proc-frame space
        cx_proc = ((bx1 + bx2) / 2.0) * proc_scale
        cy_proc = ((by1 + by2) / 2.0) * proc_scale

        best_mp, best_dist = None, float("inf")
        for mp_face in mp_faces:
            lm   = mp_face.landmark
            dist = math.sqrt(
                (lm[NOSE_TIP].x * proc_w - cx_proc) ** 2 +
                (lm[NOSE_TIP].y * proc_h - cy_proc) ** 2
            )
            if dist < best_dist:
                best_dist = dist
                best_mp   = mp_face
        matched.append(best_mp)
    return matched


# ═══════════════════════════════════════════════════════════════
# CORE PIPELINE
# ═══════════════════════════════════════════════════════════════

def run_full_validation(
    pil_image: Image.Image,
    frame_cv2: np.ndarray,
) -> dict:
    """
    Priority chain:
      dimension_guard -> no_human -> irrelevant -> multiple_person
      -> no_landmarks -> sunglasses -> eyes_closed -> pose
    """
    img_h, img_w = frame_cv2.shape[:2]
    t0           = time.time()

    # Step 0: dimension guard
    dim_error = validate_image_dimensions(frame_cv2)
    if dim_error:
        return {"category": "irrelevant", "total_faces": 0,
                "frontal_count": 0, "side_count": 0, "faces": [],
                "validation": {"dimension_error": dim_error},
                "processing_ms": round((time.time() - t0) * 1000, 2)}

    # Step 1: downscale for all MediaPipe operations
    proc_frame, proc_scale = _resize_for_processing(frame_cv2)
    proc_h, proc_w         = proc_frame.shape[:2]

    # Step 2: MTCNN on original full-resolution image
    bbox_, prob_, mtcnn_lms_ = mtcnn.detect(pil_image, landmarks=True)

    # Step 3: confidence + size filter
    valid_indices: list = []
    if bbox_ is not None and prob_ is not None:
        for i, p in enumerate(prob_):
            if p is None or float(p) < MTCNN_CONFIDENCE_THRESHOLD:
                continue
            fw = abs(float(bbox_[i][2]) - float(bbox_[i][0]))
            fh = abs(float(bbox_[i][3]) - float(bbox_[i][1]))
            if fw < MIN_FACE_SIZE_PX or fh < MIN_FACE_SIZE_PX:
                logger.warning(f"Skipping small face #{i+1}: {fw:.0f}x{fh:.0f}px")
                continue
            valid_indices.append(i)

    valid_bboxes = [bbox_[i] for i in valid_indices] if valid_indices else []
    valid_probs  = [prob_[i]  for i in valid_indices] if valid_indices else []

    # Step 4: no-human / irrelevant
    no_human = detect_no_human(valid_bboxes, valid_probs)
    if not no_human["is_human"]:
        irr      = detect_irrelevant_image(valid_bboxes, frame_cv2)
        category = "irrelevant" if irr["is_irrelevant"] else "no_human"
        return {"category": category, "total_faces": 0,
                "frontal_count": 0, "side_count": 0, "faces": [],
                "validation": {"no_human": no_human, "irrelevant": irr},
                "processing_ms": round((time.time() - t0) * 1000, 2)}

    # Step 5: multiple-person (NOTE-02: proc_frame)
    person_check = detect_persons_and_size(proc_frame)
    if not person_check["valid"]:
        return {"category": person_check["category"],
                "total_faces": person_check["total_faces"],
                "frontal_count": 0, "side_count": 0, "faces": [],
                "validation": {"no_human": no_human, "person_check": person_check},
                "processing_ms": round((time.time() - t0) * 1000, 2)}

    # Step 6: FaceMesh (proc_frame)
    face_mesh  = _get_face_mesh()
    frame_rgb  = cv2.cvtColor(proc_frame, cv2.COLOR_BGR2RGB)
    mp_results = face_mesh.process(frame_rgb)
    mp_faces   = mp_results.multi_face_landmarks or []
    mp_count   = len(mp_faces)
    mtcnn_count = len(valid_indices)

    if mtcnn_count > 0 and mp_count == 0:
        return {"category": "no_landmarks", "total_faces": 0,
                "frontal_count": 0, "side_count": 0, "faces": [],
                "validation": {"no_human": no_human,
                               "reason": "FaceMesh failed (lighting/occlusion/resolution)"},
                "processing_ms": round((time.time() - t0) * 1000, 2)}

    if mp_count > 1:
        return {"category": "multiple_person", "total_faces": mp_count,
                "frontal_count": 0, "side_count": 0, "faces": [],
                "validation": {"no_human": no_human,
                               "multi_person": {"mtcnn_faces": mtcnn_count,
                                                "mediapipe_faces": mp_count,
                                                "reason": "Multiple faces by FaceMesh"}},
                "processing_ms": round((time.time() - t0) * 1000, 2)}

    # Step 7: match bboxes -> landmarks  (BUG-01 fix inside)
    matched_mp = _match_landmarks_to_faces(
        valid_bboxes, mp_faces, proc_w, proc_h, proc_scale
    )

    # Step 8: per-face analysis
    face_results: list = []
    all_poses:    list = []
    face_counter = 0   # BUG-11: contiguous IDs

    for local_idx, global_idx in enumerate(valid_indices):
        bbox      = bbox_[global_idx]
        prob      = float(prob_[global_idx])
        mtcnn_lms = mtcnn_lms_[global_idx]
        mp_face   = matched_mp[local_idx]

        if mtcnn_lms is None:
            logger.warning(f"MTCNN landmarks None for detection #{local_idx+1} — skip")
            continue

        face_counter += 1
        face_data = {
            "face_id":    face_counter,
            "confidence": round(prob, 4),
            "bbox":       [round(float(v), 2) for v in bbox],
        }

        angR = _np_angle(np.array(mtcnn_lms[0]), np.array(mtcnn_lms[1]),
                         np.array(mtcnn_lms[2]))
        angL = _np_angle(np.array(mtcnn_lms[1]), np.array(mtcnn_lms[0]),
                         np.array(mtcnn_lms[2]))
        face_data["angle_right_eye"] = round(angR, 2)
        face_data["angle_left_eye"]  = round(angL, 2)
        face_data["landmarks"] = {
            "left_eye":    [float(mtcnn_lms[0][0]), float(mtcnn_lms[0][1])],
            "right_eye":   [float(mtcnn_lms[1][0]), float(mtcnn_lms[1][1])],
            "nose":        [float(mtcnn_lms[2][0]), float(mtcnn_lms[2][1])],
            "left_mouth":  [float(mtcnn_lms[3][0]), float(mtcnn_lms[3][1])],
            "right_mouth": [float(mtcnn_lms[4][0]), float(mtcnn_lms[4][1])],
        }

        # Sunglasses: proc_frame + proc-space landmarks
        face_data["sunglasses"] = detect_sunglasses(proc_frame, mp_face)

        # Closed eyes: BUG-03 pass original bbox for size guard
        proc_bbox = [v * proc_scale for v in bbox] if bbox is not None else None
        face_data["eyes"] = detect_closed_eyes(
            mp_face, proc_w, proc_h,
            orig_bbox=bbox,
            proc_bbox=proc_bbox,
        )

        # Pose: proc-space dims for camera matrix
        yaw_data   = _get_mediapipe_angles(mp_face, proc_w, proc_h) if mp_face else None
        nose_ratio = _get_nose_visibility_ratio(mp_face)             if mp_face else 1.0
        pose, pose_signals = classify_pose_strong(angR, angL, yaw_data, nose_ratio)

        face_data["pose"]         = pose
        face_data["pose_signals"] = pose_signals
        all_poses.append(pose)
        face_results.append(face_data)

    # Step 9: guard
    if not face_results:
        return {"category": "no_human", "total_faces": 0,
                "frontal_count": 0, "side_count": 0, "faces": [],
                "validation": {"no_human": no_human, "person_check": person_check},
                "processing_ms": round((time.time() - t0) * 1000, 2)}

    # Step 10: priority decision
    sunglasses_any  = any(f["sunglasses"].get("detected", False) for f in face_results)
    eyes_closed_any = any(f["eyes"].get("eyes_closed",   False) for f in face_results)

    if sunglasses_any:
        final_category = "sunglasses"
    elif eyes_closed_any:
        final_category = "eyes_closed"
    else:
        fc = all_poses.count("Frontal")
        sc = all_poses.count("Side")
        if   fc > 0 and sc == 0: final_category = "front"
        elif sc > 0 and fc == 0: final_category = "side"
        else:                    final_category = "mixed"

    return {
        "category":      final_category,
        "total_faces":   len(face_results),
        "frontal_count": all_poses.count("Frontal"),
        "side_count":    all_poses.count("Side"),
        "faces":         face_results,
        "validation":    {"no_human": no_human, "person_check": person_check},
        "processing_ms": round((time.time() - t0) * 1000, 2),
    }


# ═══════════════════════════════════════════════════════════════
# ANNOTATION
# ═══════════════════════════════════════════════════════════════

def draw_annotations(frame: np.ndarray, result: dict) -> np.ndarray:
    h_f, w_f  = frame.shape[:2]
    category  = result["category"]
    color_map = {
        "front": (0,255,0), "side": (0,0,255), "mixed": (255,165,0),
        "eyes_closed": (0,255,255), "multiple_person": (255,0,255),
        "sunglasses": (255,0,255), "no_human": (128,128,128),
        "irrelevant": (128,128,128), "no_landmarks": (100,100,200),
    }
    box_color = color_map.get(category, (200,200,200))

    def _cx(v): return max(0, min(int(v), w_f - 1))
    def _cy(v): return max(0, min(int(v), h_f - 1))

    for face in result.get("faces", []):
        bbox = face.get("bbox"); pose = face.get("pose","Unknown")
        eyes = face.get("eyes",{}); sigs = face.get("pose_signals",{})
        lms  = face.get("landmarks",{}); sg = face.get("sunglasses",{})

        if bbox:
            x1,y1,x2,y2 = _cx(bbox[0]),_cy(bbox[1]),_cx(bbox[2]),_cy(bbox[3])
            cv2.rectangle(frame, (x1,y1), (x2,y2), box_color, 2)
            cv2.putText(frame, f"Conf:{face.get('confidence',0):.2f}",
                        (x1,max(15,y1-75)), FONT, FONT_SCALE, (200,200,200), 1, cv2.LINE_AA)
            if sg.get("detected"):
                cv2.putText(frame, "SUNGLASSES", (x1,max(15,y1-60)),
                            FONT, FONT_SCALE, (255,0,255), FONT_THICK, cv2.LINE_AA)
            cv2.putText(frame,
                        f"{pose}|Score:{sigs.get('total_score','?')}|{sigs.get('confidence','?')}",
                        (x1,max(15,y1-45)), FONT, FONT_SCALE, box_color, FONT_THICK, cv2.LINE_AA)
            if "yaw" in sigs:
                cv2.putText(frame,
                            f"Yaw:{sigs['yaw']} Pitch:{sigs['pitch']} Roll:{sigs['roll']}",
                            (x1,max(15,y1-30)), FONT, FONT_SCALE, (200,200,200), 1, cv2.LINE_AA)
            ear_text  = "EYES CLOSED" if eyes.get("eyes_closed") else f"EAR:{eyes.get('avg_ear','?')}"
            ear_color = (0,255,255) if eyes.get("eyes_closed") else (150,255,150)
            cv2.putText(frame, ear_text, (x1,max(15,y1-15)),
                        FONT, FONT_SCALE, ear_color, 1, cv2.LINE_AA)

        for key in ("left_eye","right_eye","nose","left_mouth","right_mouth"):
            pt = lms.get(key)
            if pt: cv2.circle(frame, (_cx(pt[0]),_cy(pt[1])), 4, (0,255,255), -1)
        try:
            le,re,ns = lms["left_eye"],lms["right_eye"],lms["nose"]
            for p1,p2 in [(le,re),(le,ns),(re,ns)]:
                cv2.line(frame, (_cx(p1[0]),_cy(p1[1])), (_cx(p2[0]),_cy(p2[1])),
                         LINE_COLOR, 2)
        except Exception:
            pass

    cv2.rectangle(frame, (0,0), (w_f,55), (30,30,30), -1)
    cv2.putText(frame,
                f"Category:{category.upper()} | Faces:{result['total_faces']} | "
                f"F:{result['frontal_count']} S:{result['side_count']} | "
                f"{result['processing_ms']}ms",
                (10,35), FONT, FONT_SCALE, (255,255,255), FONT_THICK, cv2.LINE_AA)
    return frame


# ═══════════════════════════════════════════════════════════════
# API endpoints
# ═══════════════════════════════════════════════════════════════

@app.post("/validate-face")
async def validate_face(
    file: UploadFile = File(...),
    save: bool       = Query(False),
):
    if file.content_type not in ("image/jpeg","image/png","image/jpg"):
        raise HTTPException(400, "Only JPEG/PNG supported")
    try:
        contents  = await file.read();  await file.close()
        pil_image = Image.open(io.BytesIO(contents))
        if pil_image.mode != "RGB": pil_image = pil_image.convert("RGB")
        cv2_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        result    = run_full_validation(pil_image, cv2_image)
        category  = result["category"]
        output_path = filename = None
        if save:
            annotated   = draw_annotations(cv2_image.copy(), result)
            timestamp   = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            filename    = f"{category}_{timestamp}.jpg"
            output_path = os.path.join(CATEGORY_FOLDERS.get(category, BASE_OUTPUT), filename)
            cv2.imwrite(output_path, annotated)
            logger.info(f"Saved: {filename} -> {category}")
        result.update({"output_image": output_path, "filename": filename,
                        "saved": save, "success": category == "front"})
        status_code = CATEGORY_STATUS_MAP.get(category, 400)
        logger.info(f"Done | {category} | HTTP {status_code} | "
                    f"faces={result['total_faces']} | {result['processing_ms']}ms")
        return JSONResponse(status_code=status_code, content=result)
    except Exception as exc:
        logger.error(f"Processing error: {exc}")
        raise HTTPException(500, str(exc))


@app.get("/get-image")
async def get_image(filename: str = Query(...), category: Optional[str] = Query(None)):
    folder = CATEGORY_FOLDERS.get(category, BASE_OUTPUT) if category else BASE_OUTPUT
    path   = os.path.join(folder, filename)
    if not os.path.exists(path):
        for f in CATEGORY_FOLDERS.values():
            cand = os.path.join(f, filename)
            if os.path.exists(cand): path = cand; break
    if not os.path.exists(path):
        raise HTTPException(404, f"'{filename}' not found")
    return FileResponse(path, media_type="image/jpeg")


@app.get("/get-latest")
async def get_latest(
    category: str = Query("front", description=(
        "front|side|mixed|eyes_closed|multiple_person|sunglasses|"
        "no_human|irrelevant|no_landmarks"
    ))
):
    folder = CATEGORY_FOLDERS.get(category)
    if not folder: raise HTTPException(400, f"Unknown category: {category}")
    files = sorted(f for f in os.listdir(folder) if f.endswith(".jpg"))
    if not files: raise HTTPException(404, f"No images in '{category}'")
    return FileResponse(os.path.join(folder, files[-1]), media_type="image/jpeg")


@app.get("/list-outputs")
async def list_outputs():
    output = {}; total = 0
    for cat, folder in CATEGORY_FOLDERS.items():
        files = sorted(os.listdir(folder), reverse=True)
        output[cat] = {"count": len(files), "files": files}
        total += len(files)
    return JSONResponse({"success": True, "total": total, "categories": output})


@app.get("/health")
async def health():
    return {"status": "ok", "device": str(device),
            "gpu": torch.cuda.is_available(),
            "version": "2.5.0",
            "categories": list(CATEGORY_FOLDERS.keys())}


@app.post("/validate-folder")
async def validate_folder(
    folder_path: str = Query(...),
    save: bool       = Query(False),
):
    if not os.path.exists(folder_path):
        raise HTTPException(404, "Folder does not exist")
    if not os.path.isdir(folder_path):
        raise HTTPException(400, "Path is not a folder")
    images = [f for f in os.listdir(folder_path)
              if f.lower().endswith((".jpg",".jpeg",".png"))]
    if not images: raise HTTPException(404, "No supported images found")

    batch_results: list = []; total_processed = 0; t0 = time.time()
    for img_name in sorted(images):
        img_path  = os.path.join(folder_path, img_name)
        file_size = os.path.getsize(img_path)
        if file_size > BATCH_MAX_FILE_BYTES:
            logger.warning(f"Skipping {img_name}: {file_size/1e6:.1f}MB > limit")
            batch_results.append({"original_file": img_name,
                "error": f"File too large ({file_size/1e6:.1f}MB)"})
            continue
        try:
            with Image.open(img_path) as pil_img:
                if pil_img.mode != "RGB": pil_img = pil_img.convert("RGB")
                pil_copy = pil_img.copy()
            cv2_img  = cv2.cvtColor(np.array(pil_copy), cv2.COLOR_RGB2BGR)
            result   = run_full_validation(pil_copy, cv2_img)
            category = result["category"]
            saved_as = None
            if save:
                annotated = draw_annotations(cv2_img.copy(), result)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                filename  = f"{category}_{timestamp}.jpg"
                out_path  = os.path.join(CATEGORY_FOLDERS.get(category, BASE_OUTPUT), filename)
                if not cv2.imwrite(out_path, annotated):
                    raise RuntimeError("cv2.imwrite failed")
                saved_as = filename
            batch_results.append({"original_file": img_name, "category": category,
                                   "faces_detected": result["total_faces"], "saved_as": saved_as})
            total_processed += 1
        except Exception as exc:
            logger.error(f"Batch error — {img_name}: {exc}")
            batch_results.append({"original_file": img_name, "error": str(exc)})

    return JSONResponse({"success": True,
                         "folder_path": os.path.abspath(folder_path),
                         "total_images_found": len(images),
                         "total_processed": total_processed,
                         "processing_time_ms": round((time.time() - t0)*1000, 2),
                         "results": batch_results})



"""
/validate-csv  — Bulk CSV Face Validation Endpoint  (PRODUCTION-READY)
=======================================================================
Drop this block into your existing main.py (paste after your other endpoints,
before the  `if __name__ == "__main__":` line).

IMPROVEMENTS APPLIED
─────────────────────
  🔴 #1  get_event_loop() → get_running_loop()  (Python 3.10+ safe)
  🟠 #2  Filename now includes microseconds  → zero collision risk
  🟡 #3  url_list built lazily inside batch loop  → lower memory
  🟡 #4  Retry logic (2 retries) for timeout / 503 errors
  🟡 #5  Background job system  → POST returns job_id instantly
          GET /validate-csv/status/{job_id}   → poll progress
          GET /validate-csv/download/{job_id} → download result

Extra dependencies (add to requirements.txt if not already present):
  aiohttp
  pandas
"""

import asyncio
import io
import os
import tempfile
import time
import uuid
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
from typing import Literal

import aiohttp
import numpy as np
import pandas as pd
from fastapi import BackgroundTasks, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from PIL import Image

# ─────────────────────────────────────────────────────────────
# URL prefixes
# ─────────────────────────────────────────────────────────────
PREFIX_REVISE   = "https://ih.imagicahealth.in/teammindnext/doc_kyc/revise_doctors/"
PREFIX_ORIGINAL = "https://ih.imagicahealth.in/teammindnext/doctors/"

# ─────────────────────────────────────────────────────────────
# Required CSV columns
# ─────────────────────────────────────────────────────────────
REQUIRED_COLUMNS = {"photograph", "revice_photograph"}

# ─────────────────────────────────────────────────────────────
# Human-readable comments per category
# ─────────────────────────────────────────────────────────────
CATEGORY_COMMENT: dict[str, str] = {
    "front":            "Valid frontal face",
    "side":             "Face is not frontal — side pose detected",
    "mixed":            "Mixed poses detected",
    "eyes_closed":      "Eyes are closed",
    "sunglasses":       "Dark goggles or sunglasses detected",
    "multiple_person":  "Multiple persons detected in image",
    "no_human":         "No human face detected",
    "irrelevant":       "Image does not contain a human",
    "no_landmarks":     "Face landmarks could not be extracted",
}

# ─────────────────────────────────────────────────────────────
# Concurrency / retry settings
# ─────────────────────────────────────────────────────────────
MAX_CONCURRENT_DOWNLOADS = 10
IMAGE_DOWNLOAD_TIMEOUT   = 15       # seconds per attempt
BATCH_SIZE               = 200
THREAD_WORKERS           = min(8, max(4, os.cpu_count() or 4))
MAX_RETRIES              = 2        # FIX #4: retry transient failures
RETRY_DELAY              = 1.5      # seconds between retries

# ─────────────────────────────────────────────────────────────
# FIX #5: In-memory job store
# Replace with Redis / DB for multi-process deployments
# ─────────────────────────────────────────────────────────────
_jobs: dict[str, dict] = {}
# Schema per job:
# {
#   "status":       "pending" | "running" | "done" | "failed",
#   "submitted_at": str,
#   "started_at":   str | None,
#   "finished_at":  str | None,
#   "total_rows":   int,
#   "rows_done":    int,
#   "total_valid":  int | None,
#   "total_invalid":int | None,
#   "elapsed_ms":   float | None,
#   "output_path":  str | None,
#   "output_name":  str | None,
#   "error":        str | None,
# }


# ═══════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════

def _build_image_url(row: pd.Series) -> tuple[str | None, str]:
    revise = str(row.get("revice_photograph", "") or "").strip()
    photo  = str(row.get("photograph",        "") or "").strip()
    if revise and revise.lower() not in ("nan", "none", ""):
        return PREFIX_REVISE + revise, "revice_photograph"
    if photo and photo.lower() not in ("nan", "none", ""):
        return PREFIX_ORIGINAL + photo, "photograph"
    return None, "none"


def _empty_result(comment: str, status_code: int = 404) -> dict:
    return {
        "status_code":       status_code,
        "validation_status": "no_human",
        "is_valid":          0,
        "comment":           comment,
    }


def _run_validation_sync(pil_image: Image.Image) -> dict:
    import cv2
    cv2_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    return run_full_validation(pil_image, cv2_image)   # from your main.py


async def _immediate_result(res: dict) -> dict:
    return res


async def _download_and_validate(
    session:   aiohttp.ClientSession,
    semaphore: asyncio.Semaphore,
    executor:  ThreadPoolExecutor,
    url:       str,
) -> dict:
    """
    Download one image and validate.
    FIX #4: retries up to MAX_RETRIES times on timeout / 503.
    FIX #1: uses get_running_loop() instead of deprecated get_event_loop().
    """
    async with semaphore:
        raw = None
        last_error = ""

        for attempt in range(1, MAX_RETRIES + 2):   # attempts: 1, 2, 3
            try:
                async with session.get(
                    url,
                    timeout=aiohttp.ClientTimeout(total=IMAGE_DOWNLOAD_TIMEOUT),
                ) as resp:
                    if resp.status == 503:
                        last_error = "HTTP 503 Service Unavailable"
                        if attempt <= MAX_RETRIES:
                            await asyncio.sleep(RETRY_DELAY * attempt)
                            continue
                        return _empty_result(last_error, status_code=503)

                    if resp.status != 200:
                        return _empty_result(
                            f"HTTP {resp.status} when downloading image",
                            status_code=resp.status,
                        )
                    raw = await resp.read()
                    break   # success

            except asyncio.TimeoutError:
                last_error = "Image request timed out"
                if attempt <= MAX_RETRIES:
                    await asyncio.sleep(RETRY_DELAY * attempt)
                    continue
                return _empty_result(last_error, status_code=408)

            except aiohttp.ClientError as exc:
                return _empty_result(f"Network error: {exc}", status_code=503)

            except Exception as exc:
                return _empty_result(f"Download error: {exc}", status_code=500)

    if raw is None:
        return _empty_result(last_error or "Unknown download failure", status_code=500)

    # Decode image
    try:
        with Image.open(io.BytesIO(raw)) as img:
            pil_image = img.convert("RGB")
    except Exception as exc:
        return _empty_result(f"Cannot decode image: {exc}", status_code=422)

    # FIX #1: get_running_loop() — correct for async context (Python 3.10+)
    loop = asyncio.get_running_loop()
    try:
        result = await loop.run_in_executor(executor, _run_validation_sync, pil_image)
    except Exception as exc:
        return _empty_result(f"Validation error: {exc}", status_code=500)

    category = result.get("category", "no_human")
    return {
        "status_code":       CATEGORY_STATUS_MAP.get(category, 400),
        "validation_status": category,
        "is_valid":          1 if category == "front" else 0,
        "comment":           CATEGORY_COMMENT.get(category, category),
    }


async def _process_batch(
    session:   aiohttp.ClientSession,
    semaphore: asyncio.Semaphore,
    executor:  ThreadPoolExecutor,
    batch:     list[tuple[str | None, str]],
) -> list[dict]:
    tasks = [
        _immediate_result(_empty_result("No image address provided in this row"))
        if url is None
        else _download_and_validate(session, semaphore, executor, url)
        for url, _ in batch
    ]
    return await asyncio.gather(*tasks, return_exceptions=False)


# ═══════════════════════════════════════════════════════════════
# FIX #5: Background worker  (runs after POST returns job_id)
# ═══════════════════════════════════════════════════════════════

async def _run_validation_job(job_id: str, df: pd.DataFrame) -> None:
    job = _jobs[job_id]
    job["status"]     = "running"
    job["started_at"] = datetime.now().isoformat()
    t0 = time.time()

    total_rows = len(df)
    job["total_rows"] = total_rows
    all_results: list[dict] = []

    try:
        semaphore = asyncio.Semaphore(MAX_CONCURRENT_DOWNLOADS)
        executor  = ThreadPoolExecutor(max_workers=THREAD_WORKERS)
        connector = aiohttp.TCPConnector(limit=MAX_CONCURRENT_DOWNLOADS)

        async with aiohttp.ClientSession(connector=connector) as session:
            # FIX #3: build url_list lazily inside the loop — no full list in memory
            for batch_start in range(0, total_rows, BATCH_SIZE):
                batch_df = df.iloc[batch_start : batch_start + BATCH_SIZE]
                batch    = [_build_image_url(row) for _, row in batch_df.iterrows()]

                batch_results = await _process_batch(session, semaphore, executor, batch)
                all_results.extend(batch_results)

                job["rows_done"] = min(batch_start + BATCH_SIZE, total_rows)
                logger.info(
                    f"job={job_id} | progress {job['rows_done']}/{total_rows}"
                )

        executor.shutdown(wait=True)

        # Append result columns
        df["status_code"]       = [r["status_code"]       for r in all_results]
        df["validation_status"] = [r["validation_status"] for r in all_results]
        df["is_valid"]          = [r["is_valid"]           for r in all_results]
        df["comment"]           = [r["comment"]            for r in all_results]

        total_valid   = int(df["is_valid"].sum())
        total_invalid = total_rows - total_valid
        elapsed_ms    = round((time.time() - t0) * 1000, 2)

        # FIX #2: microseconds in filename → zero collision risk
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        out_name  = f"validated_doctors_{timestamp}.csv"
        out_path  = os.path.join(tempfile.gettempdir(), out_name)
        df.to_csv(out_path, index=False)

        job.update({
            "status":        "done",
            "finished_at":   datetime.now().isoformat(),
            "total_valid":   total_valid,
            "total_invalid": total_invalid,
            "elapsed_ms":    elapsed_ms,
            "output_path":   out_path,
            "output_name":   out_name,
        })
        logger.info(
            f"job={job_id} complete | rows={total_rows} | "
            f"valid={total_valid} | invalid={total_invalid} | {elapsed_ms}ms"
        )

    except Exception as exc:
        job.update({
            "status":      "failed",
            "finished_at": datetime.now().isoformat(),
            "error":       str(exc),
        })
        logger.error(f"job={job_id} failed: {exc}")


"""
/validate-csv  — Bulk CSV Face Validation Endpoint (JSON Response)
==================================================================
Upload a CSV → get back a JSON response with results for every row.
No file download. Pure JSON.
"""

import asyncio
import io
import os
import time
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor

import aiohttp
import numpy as np
import pandas as pd
from fastapi import File, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image

# ─────────────────────────────────────────────────────────────
# URL prefixes
# ─────────────────────────────────────────────────────────────
PREFIX_REVISE   = "https://ih.imagicahealth.in/teammindnext/doc_kyc/revise_doctors/"
PREFIX_ORIGINAL = "https://ih.imagicahealth.in/teammindnext/doctors/"

# ─────────────────────────────────────────────────────────────
# Required CSV columns
# ─────────────────────────────────────────────────────────────
REQUIRED_COLUMNS = {"photograph", "revice_photograph"}

# ─────────────────────────────────────────────────────────────
# Human-readable comments per category
# ─────────────────────────────────────────────────────────────
CATEGORY_COMMENT: dict[str, str] = {
    "front":            "Valid frontal face",
    "side":             "Face is not frontal — side pose detected",
    "mixed":            "Mixed poses detected",
    "eyes_closed":      "Eyes are closed",
    "sunglasses":       "Dark goggles or sunglasses detected",
    "multiple_person":  "Multiple persons detected in image",
    "no_human":         "No human face detected",
    "irrelevant":       "Image does not contain a human",
    "no_landmarks":     "Face landmarks could not be extracted",
}

# ─────────────────────────────────────────────────────────────
# Concurrency / retry settings
# ─────────────────────────────────────────────────────────────
MAX_CONCURRENT_DOWNLOADS = 10
IMAGE_DOWNLOAD_TIMEOUT   = 15
BATCH_SIZE               = 200
THREAD_WORKERS           = min(8, max(4, os.cpu_count() or 4))
MAX_RETRIES              = 2
RETRY_DELAY              = 1.5


# ═══════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════

def _build_image_url(row: pd.Series) -> tuple[str | None, str]:
    revise = str(row.get("revice_photograph", "") or "").strip()
    photo  = str(row.get("photograph",        "") or "").strip()
    if revise and revise.lower() not in ("nan", "none", ""):
        return PREFIX_REVISE + revise, "revice_photograph"
    if photo and photo.lower() not in ("nan", "none", ""):
        return PREFIX_ORIGINAL + photo, "photograph"
    return None, "none"


def _empty_result(comment: str, status_code: int = 404) -> dict:
    return {
        "status_code":       status_code,
        "validation_status": "no_human",
        "is_valid":          0,
        "comment":           comment,
    }


def _run_validation_sync(pil_image: Image.Image) -> dict:
    import cv2
    cv2_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    return run_full_validation(pil_image, cv2_image)   # from your main.py


async def _immediate_result(res: dict) -> dict:
    return res


async def _download_and_validate(
    session:   aiohttp.ClientSession,
    semaphore: asyncio.Semaphore,
    executor:  ThreadPoolExecutor,
    url:       str,
) -> dict:
    async with semaphore:
        raw       = None
        last_error = ""

        for attempt in range(1, MAX_RETRIES + 2):
            try:
                async with session.get(
                    url,
                    timeout=aiohttp.ClientTimeout(total=IMAGE_DOWNLOAD_TIMEOUT),
                ) as resp:
                    if resp.status == 503:
                        last_error = "HTTP 503 Service Unavailable"
                        if attempt <= MAX_RETRIES:
                            await asyncio.sleep(RETRY_DELAY * attempt)
                            continue
                        return _empty_result(last_error, status_code=503)
                    if resp.status != 200:
                        return _empty_result(
                            f"HTTP {resp.status} when downloading image",
                            status_code=resp.status,
                        )
                    raw = await resp.read()
                    break
            except asyncio.TimeoutError:
                last_error = "Image request timed out"
                if attempt <= MAX_RETRIES:
                    await asyncio.sleep(RETRY_DELAY * attempt)
                    continue
                return _empty_result(last_error, status_code=408)
            except aiohttp.ClientError as exc:
                return _empty_result(f"Network error: {exc}", status_code=503)
            except Exception as exc:
                return _empty_result(f"Download error: {exc}", status_code=500)

    if raw is None:
        return _empty_result(last_error or "Unknown download failure", status_code=500)

    try:
        with Image.open(io.BytesIO(raw)) as img:
            pil_image = img.convert("RGB")
    except Exception as exc:
        return _empty_result(f"Cannot decode image: {exc}", status_code=422)

    loop = asyncio.get_running_loop()
    try:
        result = await loop.run_in_executor(executor, _run_validation_sync, pil_image)
    except Exception as exc:
        return _empty_result(f"Validation error: {exc}", status_code=500)

    category = result.get("category", "no_human")
    return {
        "status_code":       CATEGORY_STATUS_MAP.get(category, 400),
        "validation_status": category,
        "is_valid":          1 if category == "front" else 0,
        "comment":           CATEGORY_COMMENT.get(category, category),
    }


async def _process_batch(
    session:   aiohttp.ClientSession,
    semaphore: asyncio.Semaphore,
    executor:  ThreadPoolExecutor,
    batch:     list[tuple[str | None, str]],
) -> list[dict]:
    tasks = [
        _immediate_result(_empty_result("No image address provided in this row"))
        if url is None
        else _download_and_validate(session, semaphore, executor, url)
        for url, _ in batch
    ]
    return await asyncio.gather(*tasks, return_exceptions=False)


"""
/validate-csv  — Bulk CSV Face Validation Endpoint  (MERGED FINAL)
===================================================================
3 endpoints:
  POST /validate-csv          → upload CSV, returns job_id immediately
  GET  /job-status/{job_id}   → live progress + results so far (pure JSON)
  GET  /job-download/{job_id} → download final result as CSV file

Features:
  ✅ Background job (no browser/NGINX timeout)
  ✅ Live per-row progress via /job-status
  ✅ Pure JSON results (no file download needed if you prefer JSON)
  ✅ CSV download also available when done
  ✅ Retry logic (2 retries on timeout / 503)
  ✅ Chunked processing (memory safe)
  ✅ asyncio.get_running_loop() (Python 3.10+ safe)
  ✅ Microsecond filename (zero collision risk)

Extra dependencies:
  aiohttp, pandas
"""

import asyncio
import io
import os
import tempfile
import time
import uuid
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor

import aiohttp
import numpy as np
import pandas as pd
from fastapi import BackgroundTasks, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from PIL import Image

# ─────────────────────────────────────────────────────────────
# URL prefixes
# ─────────────────────────────────────────────────────────────
PREFIX_REVISE   = "https://ih.imagicahealth.in/teammindnext/doc_kyc/revise_doctors/"
PREFIX_ORIGINAL = "https://ih.imagicahealth.in/teammindnext/doctors/"

# ─────────────────────────────────────────────────────────────
# Required CSV columns
# ─────────────────────────────────────────────────────────────
REQUIRED_COLUMNS = {"photograph", "revice_photograph"}

# ─────────────────────────────────────────────────────────────
# Human-readable comments per category
# ─────────────────────────────────────────────────────────────
CATEGORY_COMMENT: dict[str, str] = {
    "front":            "Valid frontal face",
    "side":             "Face is not frontal — side pose detected",
    "mixed":            "Mixed poses detected",
    "eyes_closed":      "Eyes are closed",
    "sunglasses":       "Dark goggles or sunglasses detected",
    "multiple_person":  "Multiple persons detected in image",
    "no_human":         "No human face detected",
    "irrelevant":       "Image does not contain a human",
    "no_landmarks":     "Face landmarks could not be extracted",
}

# ─────────────────────────────────────────────────────────────
# Concurrency / retry settings
# ─────────────────────────────────────────────────────────────
MAX_CONCURRENT_DOWNLOADS = 10
IMAGE_DOWNLOAD_TIMEOUT   = 15
BATCH_SIZE               = 200
THREAD_WORKERS           = min(8, max(4, os.cpu_count() or 4))
MAX_RETRIES              = 2
RETRY_DELAY              = 1.5

# ─────────────────────────────────────────────────────────────
# In-memory job store  {job_id: job_dict}
# ─────────────────────────────────────────────────────────────
_JOBS: dict[str, dict] = {}


# ═══════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════

def _build_image_url(row: pd.Series) -> tuple[str | None, str]:
    revise = str(row.get("revice_photograph", "") or "").strip()
    photo  = str(row.get("photograph",        "") or "").strip()
    if revise and revise.lower() not in ("nan", "none", ""):
        return PREFIX_REVISE + revise, "revice_photograph"
    if photo and photo.lower() not in ("nan", "none", ""):
        return PREFIX_ORIGINAL + photo, "photograph"
    return None, "none"


def _empty_result(comment: str, status_code: int = 404) -> dict:
    return {
        "status_code":       status_code,
        "validation_status": "no_human",
        "is_valid":          0,
        "comment":           comment,
    }


def _run_validation_sync(pil_image: Image.Image) -> dict:
    import cv2
    cv2_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    return run_full_validation(pil_image, cv2_image)   # from your main.py


async def _immediate_result(res: dict) -> dict:
    return res


async def _download_and_validate(
    session:   aiohttp.ClientSession,
    semaphore: asyncio.Semaphore,
    executor:  ThreadPoolExecutor,
    url:       str,
) -> dict:
    """Download one image with retry, then run face validation."""
    async with semaphore:
        raw        = None
        last_error = ""

        for attempt in range(1, MAX_RETRIES + 2):   # attempts: 1, 2, 3
            try:
                async with session.get(
                    url,
                    timeout=aiohttp.ClientTimeout(total=IMAGE_DOWNLOAD_TIMEOUT),
                ) as resp:
                    if resp.status == 503:
                        last_error = "HTTP 503 Service Unavailable"
                        if attempt <= MAX_RETRIES:
                            await asyncio.sleep(RETRY_DELAY * attempt)
                            continue
                        return _empty_result(last_error, status_code=503)
                    if resp.status != 200:
                        return _empty_result(
                            f"HTTP {resp.status} when downloading image",
                            status_code=resp.status,
                        )
                    raw = await resp.read()
                    break   # success

            except asyncio.TimeoutError:
                last_error = f"Image request timed out (attempt {attempt})"
                if attempt <= MAX_RETRIES:
                    await asyncio.sleep(RETRY_DELAY * attempt)
                    continue
                return _empty_result(last_error, status_code=408)
            except aiohttp.ClientError as exc:
                return _empty_result(f"Network error: {exc}", status_code=503)
            except Exception as exc:
                return _empty_result(f"Download error: {exc}", status_code=500)

    if raw is None:
        return _empty_result(last_error or "Unknown download failure", status_code=500)

    # Decode image
    try:
        with Image.open(io.BytesIO(raw)) as img:
            pil_image = img.convert("RGB")
    except Exception as exc:
        return _empty_result(f"Cannot decode image: {exc}", status_code=422)

    # Run CPU-bound validation in thread pool
    loop = asyncio.get_running_loop()
    try:
        result = await loop.run_in_executor(executor, _run_validation_sync, pil_image)
    except Exception as exc:
        return _empty_result(f"Validation error: {exc}", status_code=500)

    category = result.get("category", "no_human")
    return {
        "status_code":       CATEGORY_STATUS_MAP.get(category, 400),
        "validation_status": category,
        "is_valid":          1 if category == "front" else 0,
        "comment":           CATEGORY_COMMENT.get(category, category),
    }


async def _process_batch(
    session:   aiohttp.ClientSession,
    semaphore: asyncio.Semaphore,
    executor:  ThreadPoolExecutor,
    batch_rows: list[pd.Series],
) -> list[dict]:
    """Run a batch of rows concurrently."""
    tasks = []
    for row in batch_rows:
        url, _ = _build_image_url(row)
        tasks.append(
            _immediate_result(_empty_result("No image address provided in this row"))
            if url is None
            else _download_and_validate(session, semaphore, executor, url)
        )
    return await asyncio.gather(*tasks, return_exceptions=False)


# ═══════════════════════════════════════════════════════════════
# Background worker
# ═══════════════════════════════════════════════════════════════

async def _run_validation_job(job_id: str, df: pd.DataFrame) -> None:
    """
    Runs in background after POST returns.
    - Processes rows in BATCH_SIZE chunks (memory safe).
    - Appends each result to job["results"] immediately (live polling).
    - Saves final CSV to temp dir for download.
    """
    job   = _JOBS[job_id]
    t0    = time.time()
    total = len(df)
    rows  = [row for _, row in df.iterrows()]

    semaphore = asyncio.Semaphore(MAX_CONCURRENT_DOWNLOADS)
    executor  = ThreadPoolExecutor(max_workers=THREAD_WORKERS)
    connector = aiohttp.TCPConnector(limit=MAX_CONCURRENT_DOWNLOADS)

    job["status"]    = "processing"
    job["processed"] = 0
    job["progress"]  = 0.0
    job["results"]   = []   # live per-row results, readable via /job-status at any time

    try:
        async with aiohttp.ClientSession(connector=connector) as session:
            for batch_start in range(0, total, BATCH_SIZE):
                chunk         = rows[batch_start : batch_start + BATCH_SIZE]
                batch_results = await _process_batch(session, semaphore, executor, chunk)

                # Append each row result with full context immediately
                for i, result in enumerate(batch_results):
                    row        = chunk[i]
                    row_number = batch_start + i + 1
                    job["results"].append({
                        "row":               row_number,
                        "photograph":        str(row.get("photograph",        "") or "").strip(),
                        "revice_photograph": str(row.get("revice_photograph", "") or "").strip(),
                        "status_code":       result["status_code"],
                        "validation_status": result["validation_status"],
                        "is_valid":          result["is_valid"],
                        "comment":           result["comment"],
                    })

                done             = min(batch_start + BATCH_SIZE, total)
                job["processed"] = done
                job["progress"]  = round(done / total * 100, 1)
                logger.info(f"job={job_id} | {done}/{total} rows done")

        executor.shutdown(wait=True)

        # ── Build summary ─────────────────────────────────────
        total_valid   = sum(r["is_valid"] for r in job["results"])
        total_invalid = total - total_valid
        elapsed_ms    = round((time.time() - t0) * 1000, 2)

        # ── Save result CSV (microsecond timestamp = no collision) ──
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        out_name  = f"validated_doctors_{timestamp}.csv"
        out_path  = os.path.join(tempfile.gettempdir(), out_name)

        result_df = df.copy()
        result_df["status_code"]       = [r["status_code"]       for r in job["results"]]
        result_df["validation_status"] = [r["validation_status"] for r in job["results"]]
        result_df["is_valid"]          = [r["is_valid"]           for r in job["results"]]
        result_df["comment"]           = [r["comment"]            for r in job["results"]]
        result_df.to_csv(out_path, index=False)

        job.update({
            "status":        "done",
            "total_rows":    total,
            "total_valid":   total_valid,
            "total_invalid": total_invalid,
            "elapsed_ms":    elapsed_ms,
            "out_path":      out_path,
            "out_name":      out_name,
        })
        logger.info(
            f"job={job_id} complete | rows={total} | "
            f"valid={total_valid} | invalid={total_invalid} | {elapsed_ms}ms"
        )

    except Exception as exc:
        executor.shutdown(wait=False)
        job["status"] = "error"
        job["error"]  = str(exc)
        logger.error(f"job={job_id} failed: {exc}")


# ═══════════════════════════════════════════════════════════════
# Endpoint 1 — Upload CSV → get job_id immediately
# ═══════════════════════════════════════════════════════════════

@app.post(
    "/validate-csv",
    summary="Bulk CSV face validation — start job",
    description=(
        "Upload a CSV file containing `photograph` and `revice_photograph` columns. "
        "Returns a `job_id` immediately — no waiting. "
        "Poll `/job-status/{job_id}` for live progress and per-row results. "
        "Download the final CSV from `/job-download/{job_id}` when done."
    ),
    tags=["Batch"],
)
async def validate_csv(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(..., description="Upload your CSV file here"),
):
    # ── 1. Validate file type ─────────────────────────────────
    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(
            status_code=400,
            detail=f"Expected a .csv file, got: '{file.filename}'",
        )

    # ── 2. Read and parse ─────────────────────────────────────
    raw_bytes = await file.read()
    await file.close()

    try:
        df = pd.read_csv(io.BytesIO(raw_bytes))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Cannot parse CSV: {exc}")

    # ── 3. Validate required columns ──────────────────────────
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Missing required column(s): {', '.join(sorted(missing))}. "
                f"CSV must contain: photograph, revice_photograph"
            ),
        )

    # ── 4. Register job and kick off background task ──────────
    job_id = str(uuid.uuid4())
    _JOBS[job_id] = {
        "status":    "queued",
        "filename":  file.filename,
        "total":     len(df),
        "processed": 0,
        "progress":  0.0,
        "results":   [],
    }

    background_tasks.add_task(_run_validation_job, job_id, df)
    logger.info(f"validate-csv job queued | job_id={job_id} | rows={len(df)}")

    return JSONResponse(
        status_code=202,
        content={
            "job_id":       job_id,
            "total_rows":   len(df),
            "status":       "queued",
            "status_url":   f"/job-status/{job_id}",
            "download_url": f"/job-download/{job_id}",
            "message":      "Job started. Poll status_url for live progress.",
        },
    )


# ═══════════════════════════════════════════════════════════════
# Endpoint 2 — Live status + per-row results (JSON)
# ═══════════════════════════════════════════════════════════════

@app.get(
    "/job-status/{job_id}",
    summary="Poll CSV validation job — live progress + results",
    description=(
        "Returns live processing status and all results collected so far. "
        "Keep polling until `status` is `done` or `error`."
    ),
    tags=["Batch"],
)
async def job_status(job_id: str):
    job = _JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")

    response: dict = {
        "job_id":          job_id,
        "status":          job["status"],          # queued | processing | done | error
        "filename":        job["filename"],
        "total_rows":      job["total"],
        "processed":       job["processed"],
        "progress":        f"{job['progress']}%",
        "results_so_far":  job.get("results", []),
    }

    if job["status"] == "done":
        response.update({
            "total_valid":   job["total_valid"],
            "total_invalid": job["total_invalid"],
            "elapsed_ms":    job["elapsed_ms"],
            "download_url":  f"/job-download/{job_id}",
        })

    if job["status"] == "error":
        response["error"] = job.get("error", "Unknown error")

    return JSONResponse(content=response)


# ═══════════════════════════════════════════════════════════════
# Endpoint 3 — Download final result CSV
# ═══════════════════════════════════════════════════════════════

@app.get(
    "/job-download/{job_id}",
    summary="Download validated CSV result",
    description="Download the final CSV with 4 appended result columns. Only available when job status is `done`.",
    tags=["Batch"],
)
async def job_download(job_id: str):
    job = _JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")
    if job["status"] in ("queued", "processing"):
        raise HTTPException(
            status_code=202,
            detail=f"Job still running — {job['progress']}% done. Try again shortly.",
        )
    if job["status"] == "error":
        raise HTTPException(status_code=500, detail=f"Job failed: {job.get('error')}")

    return FileResponse(
        path=job["out_path"],
        media_type="text/csv",
        filename=job["out_name"],
        headers={
            "X-Total-Rows":    str(job["total_rows"]),
            "X-Total-Valid":   str(job["total_valid"]),
            "X-Total-Invalid": str(job["total_invalid"]),
            "X-Processing-Ms": str(job["elapsed_ms"]),
        },
    )

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)

