"""
Production-Grade Multi-Layer Facial Validation & Image Screening System
========================================================================
Architecture : MTCNN + MediaPipe FaceDetection + MediaPipe FaceMesh
               + OpenCV SolvePnP + Weighted Scoring
Validation Layers (Priority Order):
  1. No Human Detection
  2. Irrelevant Image Detection
  3. Multiple Person Detection  (max 1 face allowed)
  4. Dark Goggle Detection      (5-signal voting, 4/5 required — clear/prescription glasses pass)
  5. Closed Eyes Detection      (EAR method)
  6. Pose Classification        (Frontal / Side)

"""

import os
import io
import cv2
import math
import time
import logging
import numpy as np
import torch
import uvicorn
import mediapipe as mp

from datetime import datetime
from PIL import Image
from facenet_pytorch import MTCNN
from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.responses import JSONResponse, FileResponse
from typing import Optional

# ============================================================
# Logging
# ============================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger("FaceValidationEngine")

# ============================================================
# App
# ============================================================
app = FastAPI(
    title="Production Face Validation Engine",
    description=(
        "Multi-layer facial validation pipeline: "
        "pose · eyes · sunglasses · liveness"
    ),
    version="2.2.0"
)

# ============================================================
# Output Folders
# ============================================================
BASE_OUTPUT      = "detection_outputs"
CATEGORY_FOLDERS = {
    "front":           os.path.join(BASE_OUTPUT, "front"),
    "side":            os.path.join(BASE_OUTPUT, "side"),
    "mixed":           os.path.join(BASE_OUTPUT, "mixed"),
    "eyes_closed":     os.path.join(BASE_OUTPUT, "eyes_closed"),
    "multiple_person": os.path.join(BASE_OUTPUT, "multiple_person"),
    "sunglasses":      os.path.join(BASE_OUTPUT, "sunglasses"),
    "no_human":        os.path.join(BASE_OUTPUT, "no_human"),
    "irrelevant":      os.path.join(BASE_OUTPUT, "irrelevant"),
}
for _folder in CATEGORY_FOLDERS.values():
    os.makedirs(_folder, exist_ok=True)

# ============================================================
# Device
# ============================================================
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
logger.info(f"Running on device: {device}")

# ============================================================
# Model Initialization  — all global, never recreated per request
# ============================================================
mtcnn = MTCNN(
    image_size=160,
    margin=0,
    min_face_size=20,
    thresholds=[0.6, 0.7, 0.7],
    factor=0.709,
    post_process=True,
    keep_all=True,
    device=device
)

mp_face_detection = mp.solutions.face_detection
mp_face_mesh      = mp.solutions.face_mesh

# [F3] Initialized once globally
face_detector = mp_face_detection.FaceDetection(
    model_selection=1,
    min_detection_confidence=0.70
)

face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=10,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)


# [F8] Release on shutdown
@app.on_event("shutdown")
async def _on_shutdown():
    face_detector.close()
    face_mesh.close()
    logger.info("MediaPipe resources released.")


# ============================================================
# Constants & Thresholds
# ============================================================

# SolvePnP 3-D reference model
MODEL_POINTS_3D = np.array([
    (0.0,    0.0,    0.0),
    (0.0,   -330.0, -65.0),
    (-225.0, 170.0, -135.0),
    (225.0,  170.0, -135.0),
    (-150.0,-150.0, -125.0),
    (150.0, -150.0, -125.0),
], dtype=np.float64)

# Pose landmarks
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

# EAR landmarks
LEFT_EYE_EAR  = [362, 385, 387, 263, 373, 380]
RIGHT_EYE_EAR = [33,  160, 158, 133, 153, 144]

# [F2] Sunglasses eye-polygon landmarks
SG_LEFT_EYE_POLY  = [33,  133, 160, 159, 158, 144, 145, 153]
SG_RIGHT_EYE_POLY = [362, 263, 387, 386, 385, 380, 381, 373]

# Detection
MTCNN_CONFIDENCE_THRESHOLD = 0.90
NO_HUMAN_CONFIDENCE_MIN    = 0.80
MIN_FACE_SIZE_PX           = 60

# Pose
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
EAR_CLOSED_THRESHOLD = 0.20
EAR_MIN_RESOLUTION   = 80

# Dark Goggle / Sunglasses detection — strict thresholds
# Goal: detect only opaque/very-dark lenses (goggles, sunglasses).
# Regular clear/prescription glasses must NOT trigger.
#
# Signal 1 — Brightness  : dark lens ← mean grey < 50   (clear glass ~120-200)
# Signal 2 — Contrast    : uniform dark ← std < 18      (clear glass shows iris ~30-60)
# Signal 3 — HSV-V mean  : dark lens ← V < 55           (clear glass ~130-220)
# Signal 4 — Dark coverage: % of lens ROI pixels below brightness threshold ≥ 70%
#             (eliminates lightly tinted / semi-transparent lenses)
# Signal 5 — Iris visible: if iris/pupil is detectable the lens is clear (NOT goggles)
#             detected when contrast > 25 AND brightness > 55
#
# Must win 4 of 5 signals (raised from 3/4).
SG_BRIGHTNESS_MAX    = 50    # mean grey: dark goggles ≤ 50,  clear glass >> 50
SG_CONTRAST_MAX      = 18    # std grey : dark goggles ≤ 18,  iris texture >> 18
SG_HSV_V_MAX         = 55    # HSV-V    : dark goggles ≤ 55,  clear glass  >> 55
SG_DARK_COVERAGE_MIN = 0.70  # ≥ 70 % of eye-ROI pixels must be dark
SG_IRIS_CONTRAST_MIN = 25    # if contrast > 25 & brightness > 55 → iris visible → NOT goggles
SG_IRIS_BRIGHT_MIN   = 55
SG_VOTE_THRESHOLD    = 4     # needs 4 / 5 to confirm dark goggles

# Drawing
LINE_COLOR = (255, 255, 0)
FONT       = cv2.FONT_HERSHEY_PLAIN
FONT_SCALE = 1
FONT_THICK = 2

# HTTP status codes per category
CATEGORY_STATUS_MAP = {
    "front":           200,
    "side":            400,
    "mixed":           400,
    "eyes_closed":     422,
    "sunglasses":      423,
    "multiple_person": 409,
    "no_human":        404,
    "irrelevant":      415,
}


# ============================================================
# MODULE 1 — No Human / Irrelevant  (MTCNN)
# ============================================================
def detect_no_human(bbox_, prob_) -> dict:
    """Returns is_human=False when MTCNN finds nothing or confidence is low."""
    if bbox_ is None or len(bbox_) == 0:
        return {
            "is_human":       False,
            "reason":         "No face bounding box detected by MTCNN",
            "max_confidence": 0.0,
        }

    valid_probs = [float(p) for p in prob_ if p is not None]
    if not valid_probs:
        return {
            "is_human":       False,
            "reason":         "All detections returned null confidence",
            "max_confidence": 0.0,
        }

    max_conf = max(valid_probs)
    if max_conf < NO_HUMAN_CONFIDENCE_MIN:
        return {
            "is_human":       False,
            "reason":         f"Confidence {max_conf:.3f} below {NO_HUMAN_CONFIDENCE_MIN}",
            "max_confidence": round(max_conf, 4),
        }

    return {
        "is_human":       True,
        "reason":         "Human face detected",
        "max_confidence": round(max_conf, 4),
    }


def detect_irrelevant_image(bbox_, prob_, frame_cv2: np.ndarray) -> dict:
    """Secondary relevance signal: skin-tone HSV check when MTCNN finds nothing."""
    if bbox_ is None or len(bbox_) == 0:
        hsv       = cv2.cvtColor(frame_cv2, cv2.COLOR_BGR2HSV)
        skin_mask = cv2.inRange(
            hsv,
            np.array([0,  20,  70], np.uint8),
            np.array([20, 255, 255], np.uint8),
        )
        skin_ratio = np.sum(skin_mask > 0) / (frame_cv2.shape[0] * frame_cv2.shape[1])
        if skin_ratio < 0.03:
            return {"is_irrelevant": True,  "reason": "No face and no skin-tone regions"}
        return {"is_irrelevant": True,  "reason": "No face bounding box detected"}

    return {"is_irrelevant": False, "reason": "Face-like structure detected"}


# ============================================================
# MODULE 2 — Multiple Person  (MediaPipe FaceDetection)
# ============================================================
def detect_persons_and_size(frame_cv2: np.ndarray) -> dict:
    """
    Uses the globally initialised face_detector. [F3]
    Returns category="multiple_person" when more than one face found. [F4]
    """
    rgb    = cv2.cvtColor(frame_cv2, cv2.COLOR_BGR2RGB)
    result = face_detector.process(rgb)         # [F3] no re-creation

    if not result.detections:
        return {
            "valid":       True,
            "total_faces": 0,
            "category":    "ok",
            "reason":      "No detections — deferred to MTCNN",
            "face_ratio":  0.0,
        }

    total_faces = len(result.detections)

    if total_faces > 1:
        return {                                 # [F4] consistent name
            "valid":       False,
            "total_faces": total_faces,
            "category":    "multiple_person",
            "reason":      f"{total_faces} persons detected. Only one allowed.",
            "face_ratio":  0.0,
        }

    return {
        "valid":       True,
        "total_faces": 1,
        "category":    "ok",
        "reason":      "Single face detected.",
        "face_ratio":  1.0,
    }


# ============================================================
# MODULE 3 — Dark Goggle / Sunglasses Detection (5-signal, 4/5 vote)
# ============================================================
def detect_sunglasses(frame_cv2: np.ndarray, face_landmarks) -> dict:
    """
    Detects ONLY opaque dark goggles / sunglasses.
    Regular clear or prescription glasses must NOT trigger.

    5 signals — needs SG_VOTE_THRESHOLD (4/5) to confirm.

    S1 — Brightness    : mean grey of eye ROI < SG_BRIGHTNESS_MAX (50)
                         Clear glass: iris/sclera visible → much brighter (120-200)
                         Dark goggles: blocked lens → very dark (< 50)

    S2 — Contrast      : std of eye ROI < SG_CONTRAST_MAX (18)
                         Clear glass: iris texture, pupil give high std (30-60)
                         Dark goggles: uniform blackness → near-zero std

    S3 — HSV-V channel : mean V < SG_HSV_V_MAX (55)
                         Measures perceived lightness; dark lenses score very low

    S4 — Dark coverage : fraction of pixels with grey < 60 must be >= SG_DARK_COVERAGE_MIN (0.70)
                         Ensures the WHOLE lens is dark, not just the frame border
                         (eliminates regular glasses whose frame is dark but interior is clear)

    S5 — Iris guard    : if contrast > SG_IRIS_CONTRAST_MIN AND brightness > SG_IRIS_BRIGHT_MIN
                         → iris/pupil is visible through the lens → NOT dark goggles
                         This single signal can VETO a detection even if S1-S4 all pass.
    """
    if face_landmarks is None:
        return {"detected": False, "reason": "No landmarks", "votes": 0, "details": {}}

    h, w = frame_cv2.shape[:2]
    lm   = face_landmarks.landmark

    def _roi(indices: list) -> np.ndarray:
        """Extract a bounding-box crop of the eye polygon landmarks."""
        xs  = [int(lm[i].x * w) for i in indices]
        ys  = [int(lm[i].y * h) for i in indices]
        x1, y1 = max(0, min(xs) - 4), max(0, min(ys) - 4)
        x2, y2 = min(w, max(xs) + 4), min(h, max(ys) + 4)
        return frame_cv2[y1:y2, x1:x2]

    details = {}

    for side, roi in [("left",  _roi(SG_LEFT_EYE_POLY)),
                      ("right", _roi(SG_RIGHT_EYE_POLY))]:
        if roi.size == 0:
            # Can't measure → assume clear (safe default, no false positive)
            details[f"{side}_brightness"]    = 255.0
            details[f"{side}_contrast"]      = 255.0
            details[f"{side}_v_channel"]     = 255.0
            details[f"{side}_dark_coverage"] = 0.0
            continue

        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        hsv  = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        details[f"{side}_brightness"]    = round(float(np.mean(gray)), 2)
        details[f"{side}_contrast"]      = round(float(np.std(gray)),  2)
        details[f"{side}_v_channel"]     = round(float(np.mean(hsv[:, :, 2])), 2)
        # Fraction of pixels that are truly dark (below pixel value 60)
        dark_pixels = float(np.sum(gray < 60))
        details[f"{side}_dark_coverage"] = round(dark_pixels / (gray.size + 1e-8), 4)

    l_b  = details.get("left_brightness",    255)
    r_b  = details.get("right_brightness",   255)
    l_c  = details.get("left_contrast",      255)
    r_c  = details.get("right_contrast",     255)
    l_v  = details.get("left_v_channel",     255)
    r_v  = details.get("right_v_channel",    255)
    l_dc = details.get("left_dark_coverage",   0)
    r_dc = details.get("right_dark_coverage",  0)

    votes = 0

    # S1 — Very low brightness in both eyes
    if l_b < SG_BRIGHTNESS_MAX and r_b < SG_BRIGHTNESS_MAX:
        votes += 1
        details["s1_brightness"] = f"dark_goggle (L={l_b}, R={r_b})"
    else:
        details["s1_brightness"] = f"clear (L={l_b}, R={r_b})"

    # S2 — Very low contrast (uniform dark, no iris texture)
    if l_c < SG_CONTRAST_MAX and r_c < SG_CONTRAST_MAX:
        votes += 1
        details["s2_contrast"] = f"dark_goggle (L={l_c}, R={r_c})"
    else:
        details["s2_contrast"] = f"clear (L={l_c}, R={r_c})"

    # S3 — Low HSV Value channel
    if l_v < SG_HSV_V_MAX and r_v < SG_HSV_V_MAX:
        votes += 1
        details["s3_hsv_v"] = f"dark_goggle (L={l_v}, R={r_v})"
    else:
        details["s3_hsv_v"] = f"clear (L={l_v}, R={r_v})"

    # S4 — Dark pixels cover most of the lens area (not just the frame)
    if l_dc >= SG_DARK_COVERAGE_MIN and r_dc >= SG_DARK_COVERAGE_MIN:
        votes += 1
        details["s4_dark_coverage"] = f"dark_goggle (L={l_dc:.2f}, R={r_dc:.2f})"
    else:
        details["s4_dark_coverage"] = f"clear (L={l_dc:.2f}, R={r_dc:.2f})"

    # S5 — Iris guard: iris/pupil visible → NOT dark goggles (veto signal)
    # If EITHER eye shows visible iris texture, subtract a vote (can go negative)
    iris_visible_left  = (l_c > SG_IRIS_CONTRAST_MIN and l_b > SG_IRIS_BRIGHT_MIN)
    iris_visible_right = (r_c > SG_IRIS_CONTRAST_MIN and r_b > SG_IRIS_BRIGHT_MIN)
    if iris_visible_left or iris_visible_right:
        votes -= 1   # Active veto: drags below threshold even if S1-S4 all pass
        details["s5_iris_guard"] = (
            f"iris_visible — veto applied "
            f"(L_visible={iris_visible_left}, R_visible={iris_visible_right})"
        )
    else:
        votes += 1   # Counts as a positive signal when no iris is visible
        details["s5_iris_guard"] = "no_iris_visible — supports dark_goggle"

    detected         = votes >= SG_VOTE_THRESHOLD
    details["votes"] = votes
    reason           = (
        f"Dark goggles detected ({votes}/5 signals)"
        if detected else
        f"No dark goggles ({votes}/5 signals)"
    )

    return {"detected": detected, "reason": reason, "details": details}


# ============================================================
# MODULE 4 — Closed Eyes  (EAR)
# ============================================================
def _compute_ear(landmarks_2d: list, eye_idx: list) -> float:
    """EAR — close to 0 = closed, ~0.25 = open."""
    try:
        pts = [landmarks_2d[i] for i in eye_idx]
        v1  = np.linalg.norm(np.array(pts[1]) - np.array(pts[5]))
        v2  = np.linalg.norm(np.array(pts[2]) - np.array(pts[4]))
        h   = np.linalg.norm(np.array(pts[0]) - np.array(pts[3]))
        return round(float((v1 + v2) / (2.0 * h)), 4) if h > 0 else 0.3
    except Exception:
        return 0.3


def detect_closed_eyes(face_landmarks, img_w: int, img_h: int, face_bbox) -> dict:
    """Only runs when face height >= EAR_MIN_RESOLUTION pixels."""
    result = {
        "eyes_closed": False,
        "left_ear":    None,
        "right_ear":   None,
        "avg_ear":     None,
        "reason":      "Not checked",
        "checked":     False,
    }

    if face_landmarks is None:
        result["reason"] = "MediaPipe landmarks unavailable"
        return result

    if face_bbox is not None:
        x1, y1, x2, y2 = face_bbox
        face_h = abs(y2 - y1)
        if face_h < EAR_MIN_RESOLUTION:
            result["reason"] = f"Face too small ({face_h:.0f}px < {EAR_MIN_RESOLUTION}px)"
            return result

    lm           = face_landmarks.landmark
    landmarks_2d = [(lm[i].x * img_w, lm[i].y * img_h) for i in range(len(lm))]
    left_ear     = _compute_ear(landmarks_2d, LEFT_EYE_EAR)
    right_ear    = _compute_ear(landmarks_2d, RIGHT_EYE_EAR)
    avg_ear      = round((left_ear + right_ear) / 2.0, 4)

    result.update({"left_ear": left_ear, "right_ear": right_ear,
                   "avg_ear": avg_ear, "checked": True})

    if avg_ear < EAR_CLOSED_THRESHOLD:
        result["eyes_closed"] = True
        result["reason"]      = f"EAR {avg_ear:.3f} < threshold {EAR_CLOSED_THRESHOLD}"
    else:
        result["reason"] = f"EAR {avg_ear:.3f} — eyes open"

    return result


# ============================================================
# MODULE 5 — Pose  (MTCNN angle + MediaPipe yaw + solvePnP)
# ============================================================
def _np_angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    ba  = a - b;  bc = c - b
    cos = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    return float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))


def _get_mediapipe_angles(face_landmarks, img_w: int, img_h: int) -> Optional[dict]:
    """Yaw / Pitch / Roll from solvePnP."""
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

        fl  = float(img_w)
        cm  = np.array([[fl, 0, img_w/2], [0, fl, img_h/2], [0, 0, 1]], dtype=np.float64)
        ok, rvec, _ = cv2.solvePnP(
            MODEL_POINTS_3D, pts, cm, np.zeros((4, 1)),
            flags=cv2.SOLVEPNP_ITERATIVE
        )
        if not ok:
            return None

        rm, _ = cv2.Rodrigues(rvec)
        sy    = math.sqrt(rm[0, 0]**2 + rm[1, 0]**2)
        if sy > 1e-6:
            pitch = math.atan2( rm[2, 1], rm[2, 2])
            yaw   = math.atan2(-rm[2, 0], sy)
            roll  = math.atan2( rm[1, 0], rm[0, 0])
        else:
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


def _get_nose_visibility_ratio(face_landmarks, img_w: int, img_h: int) -> float:
    """Nose-width / face-width — low value = side profile."""
    try:
        lm = face_landmarks.landmark
        nw = abs(lm[NOSE_RIGHT].x - lm[NOSE_LEFT].x)   * img_w
        fw = abs(lm[RIGHT_CHEEK].x - lm[LEFT_CHEEK].x) * img_w
        return round(nw / fw, 4) if fw > 0 else 1.0
    except Exception:
        return 1.0


def classify_pose_strong(
    angR: float, angL: float,
    yaw_data: Optional[dict],
    nose_ratio: float,
) -> tuple:
    """
    Weighted 4-signal scoring (total max ±6.5):
      S1 MTCNN angle   ×1.0
      S2 Asymmetry     ×1.5
      S3 MP yaw        ×2.5
      S4 Nose ratio    ×1.5
    Score > 0 → Frontal, ≤ 0 → Side.
    Slightly-turned faces earn partial credit.
    """
    total_score = 0.0
    signals     = {}

    # S1
    ri, li = int(angR), int(angL)
    if MTCNN_FRONTAL_R_MIN <= ri <= MTCNN_FRONTAL_R_MAX and \
       MTCNN_FRONTAL_L_MIN <= li <= MTCNN_FRONTAL_L_MAX:
        s1 = 1.0;  signals["mtcnn_angle"] = "Frontal"
    elif 25 <= ri <= 68 and 25 <= li <= 70:
        s1 = 0.3;  signals["mtcnn_angle"] = "Slightly Side"
    else:
        s1 = -1.0; signals["mtcnn_angle"] = "Side"
    total_score += s1 * 1.0

    # S2
    asym = abs(angR - angL)
    if asym <= ASYM_FRONTAL_MAX:
        s2 = 1.0;  signals["asymmetry"] = f"Frontal ({asym:.1f}°)"
    elif asym <= ASYM_SLIGHTLY_MAX:
        s2 = 0.3;  signals["asymmetry"] = f"Slightly Side ({asym:.1f}°)"
    else:
        s2 = -1.0; signals["asymmetry"] = f"Side ({asym:.1f}°)"
    total_score += s2 * 1.5

    # S3
    if yaw_data is not None:
        yaw_abs = abs(yaw_data["yaw"])
        signals.update(yaw_data)
        if yaw_abs <= YAW_FRONTAL_MAX:
            s3 = 1.0;  signals["mediapipe_yaw"] = f"Frontal ({yaw_abs}°)"
        elif yaw_abs <= YAW_SLIGHTLY_SIDE_MAX:
            decay = 1.0 - ((yaw_abs - YAW_FRONTAL_MAX) /
                            (YAW_SLIGHTLY_SIDE_MAX - YAW_FRONTAL_MAX))
            s3 = max(0.1, decay * 0.8)
            signals["mediapipe_yaw"] = f"Slightly Side — Accepted ({yaw_abs}°)"
        else:
            s3 = -1.0; signals["mediapipe_yaw"] = f"Full Side ({yaw_abs}°)"
    else:
        s3 = s1 * 0.5
        signals["mediapipe_yaw"] = "unavailable"
    total_score += s3 * 2.5

    # S4
    signals["nose_visibility_ratio"] = nose_ratio
    if nose_ratio >= 0.18:
        s4 = 1.0;  signals["nose_visibility"] = f"Frontal ({nose_ratio})"
    elif nose_ratio >= NOSE_VISIBILITY_MIN:
        s4 = 0.2;  signals["nose_visibility"] = f"Slightly Side ({nose_ratio})"
    else:
        s4 = -1.0; signals["nose_visibility"] = f"Full Side ({nose_ratio})"
    total_score += s4 * 1.5

    final                  = "Frontal" if total_score > 0 else "Side"
    signals["total_score"] = round(total_score, 3)
    signals["max_score"]   = 6.5
    signals["confidence"]  = f"{round(abs(total_score) / 6.5 * 100, 1)}%"
    return final, signals


# ============================================================
# MODULE 6 — MTCNN ↔ MediaPipe Landmark Matching
# ============================================================
def _match_landmarks_to_faces(
    bbox_list: list,
    mp_faces: list,
    img_w: int,
    img_h: int,
) -> list:
    """Returns one mp_face (or None) per MTCNN bbox, by nearest nose-tip."""
    matched = []
    for bbox in bbox_list:
        if bbox is None:
            matched.append(None)
            continue
        bx1, by1, bx2, by2 = bbox
        b_cx = (bx1 + bx2) / 2.0
        b_cy = (by1 + by2) / 2.0
        best_mp, best_dist = None, float("inf")
        for mp_face in mp_faces:
            lm   = mp_face.landmark
            dist = math.sqrt(
                (lm[NOSE_TIP].x * img_w - b_cx) ** 2 +
                (lm[NOSE_TIP].y * img_h - b_cy) ** 2
            )
            if dist < best_dist:
                best_dist = dist
                best_mp   = mp_face
        matched.append(best_mp)
    return matched


# ============================================================
# CORE PIPELINE
# ============================================================
def run_full_validation(
    pil_image: Image.Image,
    frame_cv2: np.ndarray,
) -> dict:
    """
    Priority:
      no_human → irrelevant → multiple_person
      → sunglasses → eyes_closed → pose
    """
    img_h, img_w = frame_cv2.shape[:2]
    t0           = time.time()

    # ── 1. MTCNN ─────────────────────────────────────────────
    bbox_, prob_, mtcnn_lms_ = mtcnn.detect(pil_image, landmarks=True)

    # ── 2. No Human / Irrelevant ─────────────────────────────
    no_human = detect_no_human(bbox_, prob_ if prob_ is not None else [])
    if not no_human["is_human"]:
        irr      = detect_irrelevant_image(bbox_, prob_, frame_cv2)
        category = "irrelevant" if irr["is_irrelevant"] else "no_human"
        return {
            "category":      category,
            "total_faces":   0,
            "frontal_count": 0,
            "side_count":    0,
            "faces":         [],
            "validation":    {"no_human": no_human, "irrelevant": irr},
            "processing_ms": round((time.time() - t0) * 1000, 2),
        }

    # ── 3. Multiple Person  (MediaPipe FaceDetection) ─────────
    person_check = detect_persons_and_size(frame_cv2)
    if not person_check["valid"]:
        return {
            "category":      person_check["category"],   # [F4] "multiple_person"
            "total_faces":   person_check["total_faces"],
            "frontal_count": 0,
            "side_count":    0,
            "faces":         [],
            "validation":    {"no_human": no_human, "person_check": person_check},
            "processing_ms": round((time.time() - t0) * 1000, 2),
        }

    # ── 4. Valid MTCNN indices ────────────────────────────────
    valid_indices = []
    for i, p in enumerate(prob_):
        if p is None or float(p) < MTCNN_CONFIDENCE_THRESHOLD:
            continue
        face_w = abs(float(bbox_[i][2]) - float(bbox_[i][0]))
        face_h = abs(float(bbox_[i][3]) - float(bbox_[i][1]))
        if face_w < MIN_FACE_SIZE_PX or face_h < MIN_FACE_SIZE_PX:
            logger.warning(f"Skipping ghost #{i+1}: {face_w:.0f}x{face_h:.0f}px")
            continue
        valid_indices.append(i)

    # ── 5. FaceMesh ───────────────────────────────────────────
    frame_rgb  = cv2.cvtColor(frame_cv2, cv2.COLOR_BGR2RGB)
    mp_results = face_mesh.process(frame_rgb)
    mp_faces   = mp_results.multi_face_landmarks or []

    mp_face_count    = len(mp_faces)
    mtcnn_face_count = len(valid_indices)

    # FaceMesh found nothing — treat as irrelevant
    if mtcnn_face_count > 0 and mp_face_count == 0:
        return {
            "category":      "irrelevant",
            "total_faces":   0,
            "frontal_count": 0,
            "side_count":    0,
            "faces":         [],
            "validation":    {
                "no_human": {
                    "is_human":       False,
                    "reason":         "FaceMesh failed to extract landmarks",
                    "max_confidence": 0.0,
                }
            },
            "processing_ms": round((time.time() - t0) * 1000, 2),
        }

    # FaceMesh found more faces → multiple persons  [F4]
    if mp_face_count > mtcnn_face_count or mp_face_count > 1:
        return {
            "category":      "multiple_person",
            "total_faces":   mp_face_count,
            "frontal_count": 0,
            "side_count":    0,
            "faces":         [],
            "validation":    {
                "no_human": no_human,
                "multi_person": {
                    "mtcnn_faces":     mtcnn_face_count,
                    "mediapipe_faces": mp_face_count,
                    "reason":          "Multiple faces confirmed by FaceMesh",
                },
            },
            "processing_ms": round((time.time() - t0) * 1000, 2),
        }

    valid_bboxes = [bbox_[i] for i in valid_indices]
    matched_mp   = _match_landmarks_to_faces(valid_bboxes, mp_faces, img_w, img_h)

    # ── 6. Per-Face Analysis ──────────────────────────────────
    face_results = []
    all_poses    = []

    for local_idx, global_idx in enumerate(valid_indices):
        bbox      = bbox_[global_idx]
        prob      = float(prob_[global_idx])
        mtcnn_lms = mtcnn_lms_[global_idx]
        mp_face   = matched_mp[local_idx]

        face_data = {
            "face_id":    local_idx + 1,
            "confidence": round(prob, 4),
            "bbox":       [round(float(v), 2) for v in bbox],
        }

        # Angles
        angR = _np_angle(mtcnn_lms[0], mtcnn_lms[1], mtcnn_lms[2])
        angL = _np_angle(mtcnn_lms[1], mtcnn_lms[0], mtcnn_lms[2])
        face_data["angle_right_eye"] = round(angR, 2)
        face_data["angle_left_eye"]  = round(angL, 2)

        # Landmarks
        face_data["landmarks"] = {
            "left_eye":    [float(mtcnn_lms[0][0]), float(mtcnn_lms[0][1])],
            "right_eye":   [float(mtcnn_lms[1][0]), float(mtcnn_lms[1][1])],
            "nose":        [float(mtcnn_lms[2][0]), float(mtcnn_lms[2][1])],
            "left_mouth":  [float(mtcnn_lms[3][0]), float(mtcnn_lms[3][1])],
            "right_mouth": [float(mtcnn_lms[4][0]), float(mtcnn_lms[4][1])],
        }

        # Sunglasses  [F2]
        face_data["sunglasses"] = detect_sunglasses(frame_cv2, mp_face)

        # Closed eyes
        face_data["eyes"] = detect_closed_eyes(mp_face, img_w, img_h, bbox)

        # Pose
        yaw_data   = _get_mediapipe_angles(mp_face, img_w, img_h)      if mp_face else None
        nose_ratio = _get_nose_visibility_ratio(mp_face, img_w, img_h) if mp_face else 1.0
        pose, pose_signals = classify_pose_strong(angR, angL, yaw_data, nose_ratio)

        face_data["pose"]         = pose
        face_data["pose_signals"] = pose_signals
        all_poses.append(pose)
        face_results.append(face_data)

    # ── 7. Priority Decision ──────────────────────────────────

    # [F5] Guard: nothing passed confidence threshold
    if not face_results:
        return {
            "category":      "no_human",
            "total_faces":   0,
            "frontal_count": 0,
            "side_count":    0,
            "faces":         [],
            "validation":    {"no_human": no_human, "person_check": person_check},
            "processing_ms": round((time.time() - t0) * 1000, 2),
        }

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


# ============================================================
# ANNOTATION
# ============================================================
def draw_annotations(frame: np.ndarray, result: dict) -> np.ndarray:
    """Bounding boxes, labels, landmarks, summary banner."""
    category  = result["category"]
    faces     = result["faces"]
    color_map = {
        "front":           (0,   255, 0),
        "side":            (0,   0,   255),
        "mixed":           (255, 165, 0),
        "eyes_closed":     (0,   255, 255),
        "multiple_person": (255, 0,   255),
        "sunglasses":      (255, 0,   255),
        "no_human":        (128, 128, 128),
        "irrelevant":      (128, 128, 128),
    }
    box_color = color_map.get(category, (200, 200, 200))

    for face in faces:
        bbox = face.get("bbox")
        pose = face.get("pose", "Unknown")
        eyes = face.get("eyes", {})
        sigs = face.get("pose_signals", {})
        lms  = face.get("landmarks", {})
        sg   = face.get("sunglasses", {})

        if bbox:
            x1, y1, x2, y2 = [int(v) for v in bbox]
            cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)

            cv2.putText(frame,
                        f"Conf:{face.get('confidence', 0):.2f}",
                        (x1, y1 - 75), FONT, FONT_SCALE, (200, 200, 200), 1, cv2.LINE_AA)

            if sg.get("detected"):
                cv2.putText(frame, "SUNGLASSES",
                            (x1, y1 - 60), FONT, FONT_SCALE,
                            (255, 0, 255), FONT_THICK, cv2.LINE_AA)

            cv2.putText(frame,
                        f"{pose} | Score:{sigs.get('total_score','?')} "
                        f"| {sigs.get('confidence','?')}",
                        (x1, y1 - 45), FONT, FONT_SCALE, box_color, FONT_THICK, cv2.LINE_AA)

            if "yaw" in sigs:
                cv2.putText(frame,
                            f"Yaw:{sigs['yaw']} Pitch:{sigs['pitch']} Roll:{sigs['roll']}",
                            (x1, y1 - 30), FONT, FONT_SCALE, (200, 200, 200), 1, cv2.LINE_AA)

            ear_text  = "EYES CLOSED" if eyes.get("eyes_closed") else f"EAR:{eyes.get('avg_ear','?')}"
            ear_color = (0, 255, 255) if eyes.get("eyes_closed") else (150, 255, 150)
            cv2.putText(frame, ear_text, (x1, y1 - 15),
                        FONT, FONT_SCALE, ear_color, 1, cv2.LINE_AA)

        for key in ["left_eye", "right_eye", "nose", "left_mouth", "right_mouth"]:
            pt = lms.get(key)
            if pt:
                cv2.circle(frame, (int(pt[0]), int(pt[1])), 4, (0, 255, 255), -1)

        try:
            le = lms["left_eye"]; re = lms["right_eye"]; ns = lms["nose"]
            cv2.line(frame, (int(le[0]), int(le[1])), (int(re[0]), int(re[1])), LINE_COLOR, 2)
            cv2.line(frame, (int(le[0]), int(le[1])), (int(ns[0]), int(ns[1])), LINE_COLOR, 2)
            cv2.line(frame, (int(re[0]), int(re[1])), (int(ns[0]), int(ns[1])), LINE_COLOR, 2)
        except Exception:
            pass

    # Banner
    cv2.rectangle(frame, (0, 0), (frame.shape[1], 55), (30, 30, 30), -1)
    cv2.putText(
        frame,
        (f"Category: {category.upper()} | "
         f"Faces:{result['total_faces']} | "
         f"F:{result['frontal_count']} S:{result['side_count']} | "
         f"{result['processing_ms']}ms"),
        (10, 35), FONT, FONT_SCALE, (255, 255, 255), FONT_THICK, cv2.LINE_AA,
    )
    return frame


# ============================================================
# API — /validate-face
# ============================================================
@app.post("/validate-face", summary="Full multi-layer face validation")
async def validate_face(
    file: UploadFile = File(...),
    save: bool       = Query(False, description="Save annotated image to disk"),
):
    """
    Validation priority:
    No Human → Irrelevant → Multiple Person → Sunglasses → Eyes Closed → Pose
    """
    if file.content_type not in ["image/jpeg", "image/png", "image/jpg"]:
        raise HTTPException(status_code=400, detail="Only JPEG/PNG supported")

    try:
        contents = await file.read()
        try:
            pil_image = Image.open(io.BytesIO(contents))
            if pil_image.mode != "RGB":
                pil_image = pil_image.convert("RGB")
        finally:
            await file.close()

        cv2_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        result    = run_full_validation(pil_image, cv2_image)
        category  = result["category"]

        output_path = None
        filename    = None

        if save:
            annotated   = draw_annotations(cv2_image.copy(), result)
            timestamp   = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:19]
            filename    = f"{category}_{timestamp}.jpg"
            output_path = os.path.join(
                CATEGORY_FOLDERS.get(category, BASE_OUTPUT), filename
            )
            cv2.imwrite(output_path, annotated)
            logger.info(f"Saved: {filename} → {category}")

        result["output_image"] = output_path
        result["filename"]     = filename
        result["saved"]        = save
        result["success"]      = (category == "front")

        status_code = CATEGORY_STATUS_MAP.get(category, 400)
        logger.info(
            f"Done | category={category} | status={status_code} | "
            f"faces={result['total_faces']} | {result['processing_ms']}ms"
        )
        return JSONResponse(status_code=status_code, content=result)

    except Exception as e:
        logger.error(f"Processing error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================
# API — /get-image
# ============================================================
@app.get("/get-image", summary="Retrieve a saved output image")
async def get_image(
    filename: str           = Query(...),
    category: Optional[str] = Query(None),
):
    folder = CATEGORY_FOLDERS.get(category, BASE_OUTPUT) if category else BASE_OUTPUT
    path   = os.path.join(folder, filename)

    if not os.path.exists(path):
        for f in CATEGORY_FOLDERS.values():
            candidate = os.path.join(f, filename)
            if os.path.exists(candidate):
                path = candidate
                break

    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail=f"'{filename}' not found")

    return FileResponse(path, media_type="image/jpeg")


# ============================================================
# API — /get-latest                                      [F7]
# ============================================================
@app.get("/get-latest", summary="Get the latest image from a category")
async def get_latest(
    category: str = Query(
        "front",
        description=(                               # [F7] complete list
            "Category: front | side | mixed | "
            "eyes_closed | multiple_person | sunglasses | "
            "no_human | irrelevant"
        ),
    )
):
    folder = CATEGORY_FOLDERS.get(category)
    if not folder:
        raise HTTPException(status_code=400, detail=f"Unknown category: {category}")

    files = sorted(f for f in os.listdir(folder) if f.endswith(".jpg"))
    if not files:
        raise HTTPException(status_code=404, detail=f"No images in '{category}'")

    return FileResponse(os.path.join(folder, files[-1]), media_type="image/jpeg")


# ============================================================
# API — /list-outputs
# ============================================================
@app.get("/list-outputs", summary="List all saved output images by category")
async def list_outputs():
    output = {}
    total  = 0
    for cat, folder in CATEGORY_FOLDERS.items():
        files       = sorted(os.listdir(folder), reverse=True)
        output[cat] = {"count": len(files), "files": files}
        total      += len(files)
    return JSONResponse(content={"success": True, "total": total, "categories": output})


# ============================================================
# API — /health
# ============================================================
@app.get("/health", summary="Health check")
async def health():
    return {
        "status":     "ok",
        "device":     str(device),
        "gpu":        torch.cuda.is_available(),
        "version":    "2.2.0",
        "categories": list(CATEGORY_FOLDERS.keys()),
    }


# ============================================================
# API — /validate-folder  (batch)
# ============================================================
@app.post("/validate-folder", summary="Batch-validate all images in a folder")
async def validate_folder(
    folder_path: str = Query(..., description="Absolute or relative folder path"),
    save: bool       = Query(False, description="Save annotated images to disk"),
):
    if not os.path.exists(folder_path):
        raise HTTPException(status_code=404, detail="Folder does not exist")
    if not os.path.isdir(folder_path):
        raise HTTPException(status_code=400, detail="Path is not a folder")

    supported = (".jpg", ".jpeg", ".png")
    images    = [f for f in os.listdir(folder_path) if f.lower().endswith(supported)]
    if not images:
        raise HTTPException(status_code=404, detail="No supported images found")

    batch_results   = []
    total_processed = 0
    t0              = time.time()

    for img_name in sorted(images):
        img_path = os.path.join(folder_path, img_name)
        try:
            pil_image = Image.open(img_path)
            if pil_image.mode != "RGB":
                pil_image = pil_image.convert("RGB")
            cv2_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
            result    = run_full_validation(pil_image, cv2_image)
            category  = result["category"]
            saved_as  = None

            if save:
                annotated   = draw_annotations(cv2_image.copy(), result)
                timestamp   = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:19]
                filename    = f"{category}_{timestamp}.jpg"
                out_path    = os.path.join(CATEGORY_FOLDERS.get(category, BASE_OUTPUT), filename)
                if not cv2.imwrite(out_path, annotated):
                    raise RuntimeError("cv2.imwrite failed")
                saved_as = filename

            batch_results.append({
                "original_file":  img_name,
                "category":       category,
                "faces_detected": result["total_faces"],
                "saved_as":       saved_as,
            })
            total_processed += 1

        except Exception as e:
            logger.error(f"Batch error — {img_name}: {e}")
            batch_results.append({"original_file": img_name, "error": str(e)})

    return JSONResponse(content={
        "success":            True,
        "folder_path":        os.path.abspath(folder_path),
        "total_images_found": len(images),
        "total_processed":    total_processed,
        "processing_time_ms": round((time.time() - t0) * 1000, 2),
        "results":            batch_results,
    })


# ============================================================
# Entry point
# ============================================================
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
