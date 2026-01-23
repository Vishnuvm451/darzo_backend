from fastapi import APIRouter, File, UploadFile, HTTPException, Form
import numpy as np
import cv2
import face_recognition
import asyncio
from functools import partial
from firebase_admin import firestore
from firebase import get_db
import logging

logger = logging.getLogger(__name__)

router = APIRouter()

MAX_IMAGE_SIZE = 2 * 1024 * 1024  # 2MB

# -------------------------------------------------
# FACE VECTOR EXTRACTION
# -------------------------------------------------
def extract_face_vector(rgb_img):
    """
    Extracts a normalized 128D face vector from image.
    Returns None if face extraction fails.
    """
    try:
        # HOG model is faster for mobile images
        boxes = face_recognition.face_locations(rgb_img, model="hog")
        
        logger.info(f"🧠 Face detection: {len(boxes)} face(s) found")

        if not boxes:
            logger.warning("❌ No face detected in image")
            return None

        # ✅ FIX #1: Select the LARGEST face (not just first)
        boxes = sorted(
            boxes,
            key=lambda b: (b[2] - b[0]) * (b[1] - b[3]),
            reverse=True
        )

        encodings = face_recognition.face_encodings(rgb_img, [boxes[0]])

        if not encodings:
            logger.warning("❌ Could not create face encoding")
            return None

        vec = encodings[0]

        # Normalize vector (CRITICAL)
        norm = np.linalg.norm(vec)
        if norm == 0:
            logger.warning("❌ Vector norm is zero")
            return None

        vec = vec / norm
        logger.info(f"✅ Face vector generated (length: {len(vec)})")
        return vec.tolist()

    except Exception as e:
        logger.error(f"❌ Face extraction error: {str(e)}", exc_info=True)
        return None


# -------------------------------------------------
# REGISTER FACE ENDPOINT
# -------------------------------------------------
@router.post("/register")
async def register_face(
    admission_no: str = Form(...),
    auth_uid: str = Form(...),
    image: UploadFile = File(...)
):
    logger.info(f"🔥 /face/register CALLED for admission_no: {admission_no}")
    
    db = get_db()

    # -------------------------------------------------
    # 1. VALIDATE STUDENT EXISTS
    # -------------------------------------------------
    try:
        student_ref = db.collection("student").document(admission_no)
        student_doc = student_ref.get()

        if not student_doc.exists:
            logger.error(f"❌ Student not found: {admission_no}")
            raise HTTPException(status_code=404, detail="Student not found")

        student = student_doc.to_dict()
        logger.info(f"✅ Student found: {admission_no}")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Database error: {str(e)}")
        raise HTTPException(status_code=500, detail="Database error")

    # -------------------------------------------------
    # 2. VALIDATE AUTH_UID
    # -------------------------------------------------
    if student.get("authUid") != auth_uid:
        logger.warning(f"⚠️ Auth mismatch for {admission_no}")
        raise HTTPException(status_code=403, detail="Unauthorized access")

    # -------------------------------------------------
    # 3. CHECK IF FACE ALREADY REGISTERED
    # -------------------------------------------------
    if student.get("face_enabled") is True:
        logger.warning(f"⚠️ Face already registered for {admission_no}")
        raise HTTPException(status_code=409, detail="Face already registered")

    # -------------------------------------------------
    # 4. READ AND VALIDATE IMAGE
    # -------------------------------------------------
    # ✅ FIX #2: Validate image content type
    if image.content_type not in ["image/jpeg", "image/png"]:
        raise HTTPException(
            status_code=400,
            detail="Only JPG and PNG images are supported"
        )

    contents = await image.read()
    logger.info(f"📸 Image size: {len(contents)} bytes")

    if not contents:
        raise HTTPException(status_code=400, detail="Empty image file")

    if len(contents) > MAX_IMAGE_SIZE:
        raise HTTPException(
            status_code=413,
            detail="Image too large. Maximum 2MB allowed."
        )

    np_arr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    if img is None:
        raise HTTPException(status_code=400, detail="Invalid image format")

    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Resize for performance
    h, w = rgb.shape[:2]
    if w > 800:
        scale = 800 / w
        rgb = cv2.resize(rgb, (int(w * scale), int(h * scale)))
        logger.info(f"📏 Image resized: {w} → {int(w * scale)} pixels")

    # -------------------------------------------------
    # 5. EXTRACT FACE VECTOR (NON-BLOCKING)
    # -------------------------------------------------
    try:
        # ✅ FIX #3: Use get_running_loop (safe for modern Python)
        loop = asyncio.get_running_loop()
        face_vector = await loop.run_in_executor(
            None,
            partial(extract_face_vector, rgb)
        )
    except Exception as e:
        logger.error(f"❌ Async execution error: {str(e)}")
        raise HTTPException(status_code=500, detail="Face processing failed")

    if face_vector is None:
        raise HTTPException(
            status_code=400,
            detail="Face not detected clearly. Ensure good lighting and face is visible."
        )

    logger.info(f"🧠 Vector extracted: {len(face_vector)} dimensions")

    # -------------------------------------------------
    # 6. STORE FACE VECTOR
    # -------------------------------------------------
    try:
        db.collection("face_data").document(admission_no).set({
            "admissionNo": admission_no,
            "authUid": auth_uid,
            "vector": face_vector,
            "updatedAt": firestore.SERVER_TIMESTAMP
        })
        logger.info(f"✅ Face vector stored in face_data/{admission_no}")

    except Exception as e:
        logger.error(f"❌ Database write error: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to store face vector")

    # -------------------------------------------------
    # 7. UPDATE STUDENT RECORD
    # -------------------------------------------------
    try:
        student_ref.update({
            "face_enabled": True,
            "face_registered_at": firestore.SERVER_TIMESTAMP
        })
        logger.info("✅ Student record updated")

    except Exception as e:
        logger.error(f"❌ Failed to update student record: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to update student record")

    # -------------------------------------------------
    # 8. SUCCESS RESPONSE
    # -------------------------------------------------
    return {
        "success": True,
        "message": "Face registered successfully",
        "admission_no": admission_no,
        "vector_dimensions": len(face_vector),
        "status": "ready_for_login"
    }
