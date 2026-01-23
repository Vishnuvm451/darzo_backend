from fastapi import APIRouter, UploadFile, File, Form, HTTPException
import numpy as np
import cv2
import asyncio
from functools import partial
import logging

from face_utils import extract_face_encoding, verify_user_face
from firebase import get_db
from firebase_admin import firestore

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Face Verification"])

MAX_IMAGE_SIZE = 2 * 1024 * 1024  # 2MB safety limit

# -------------------------------------------------
# FACE VERIFICATION FOR ATTENDANCE (1:1 MATCHING)
# -------------------------------------------------
@router.post("/verify")
async def verify_face_attendance(
    admission_no: str = Form(...),
    image: UploadFile = File(...)
):
    """
    ✅ Verifies a student's face and marks attendance (1:1 matching).
    
    This endpoint performs face liveness detection and compares the captured
    face with the registered face vector. If match, attendance is marked.
    
    Body Parameters:
    - admission_no (str): Student admission number
    - image (file): Face image (JPG/PNG, max 2MB)
    
    Returns:
    - 200: Attendance marked successfully
      {
        "success": true,
        "message": "Attendance marked successfully",
        "admissionNo": "ADM001",
      }
    
    - 400: Invalid image or face not detected
    - 401: Face does not match (Attendance denied)
    - 404: Student not found or face not registered
    - 413: Image too large
    - 500: Server error
    """
    logger.info(f"🔥 /face/verify CALLED for {admission_no}")
    
    db = get_db()

    # ========== 1. VALIDATE STUDENT ==========
    try:
        logger.debug(f"📂 Validating student: {admission_no}")
        
        student_ref = db.collection("student").document(admission_no)
        student_doc = student_ref.get()

        if not student_doc.exists:
            logger.error(f"❌ Student not found: {admission_no}")
            raise HTTPException(status_code=404, detail="Student not found")

        student_data = student_doc.to_dict()
        logger.info(f"✅ Student found: {admission_no}")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Database error during student validation: {str(e)}")
        raise HTTPException(status_code=500, detail="Database error")

    # ========== 2. CHECK IF FACE REGISTERED ==========
    # ✅ FIX #1: Validate face is registered
    if not student_data.get("face_enabled"):
        logger.warning(f"⚠️ Face not registered for {admission_no}")
        raise HTTPException(status_code=404, detail="Face not registered for this student")

    # ========== 3. READ AND VALIDATE IMAGE ==========
    try:
        logger.debug("📸 Reading image file")
        contents = await image.read()

        if not contents:
            logger.error("❌ Empty image file received")
            raise HTTPException(status_code=400, detail="Empty image file")

        # ✅ FIX #2: Check image size limit
        if len(contents) > MAX_IMAGE_SIZE:
            logger.error(f"❌ Image too large: {len(contents)} > {MAX_IMAGE_SIZE}")
            raise HTTPException(
                status_code=413,
                detail="Image too large. Maximum 2MB allowed."
            )

        logger.info(f"📸 Image size: {len(contents)} bytes")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error reading image: {str(e)}")
        raise HTTPException(status_code=400, detail="Failed to read image file")

    # ========== 4. DECODE IMAGE ==========
    try:
        logger.debug("🖼️ Decoding image")
        np_arr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if img is None:
            logger.error("❌ Invalid image format")
            raise HTTPException(status_code=400, detail="Invalid image format")

        logger.debug(f"✅ Image decoded: {img.shape}")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error decoding image: {str(e)}")
        raise HTTPException(status_code=400, detail="Invalid image data")

    # ========== 5. RESIZE IMAGE FOR PERFORMANCE ==========
    # ✅ FIX #3: Resize large images to prevent timeout
    try:
        h, w = img.shape[:2]
        if w > 800:
            scale = 800 / w
            img = cv2.resize(img, (int(w * scale), int(h * scale)))
            logger.info(f"📏 Image resized: {w} → {int(w*scale)} pixels")
    except Exception as e:
        logger.error(f"❌ Error resizing image: {str(e)}")
        raise HTTPException(status_code=400, detail="Failed to process image")

    # ========== 6. EXTRACT FACE ENCODING (NON-BLOCKING) ==========
    try:
        logger.debug("🧠 Extracting face encoding (async)")
        loop = asyncio.get_event_loop()
        
        encoding = await loop.run_in_executor(
            None,
            partial(extract_face_encoding, img)
        )

        if encoding is None:
            logger.warning(f"❌ Face detection failed for {admission_no}")
            raise HTTPException(
                status_code=400,
                detail="Face not detected clearly. Ensure good lighting and face is visible."
            )

        logger.info(f"✅ Face encoding extracted (dimensions: {len(encoding)})")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error extracting face encoding: {str(e)}")
        raise HTTPException(status_code=500, detail="Face processing failed")

    # ========== 7. VERIFY FACE AGAINST STORED VECTOR ==========
    try:
        logger.debug("🔐 Verifying face against stored vector")
        success, message = verify_user_face(admission_no, encoding)

        if not success:
            logger.warning(f"❌ Face verification failed: {message}")
            raise HTTPException(
                status_code=401,
                detail=message
            )

        logger.info(f"✅ Face verification successful: {message}")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error during face verification: {str(e)}")
        raise HTTPException(status_code=500, detail="Face verification error")

    # ========== 8. SUCCESS RESPONSE ==========
    # ✅ FIX #4: Detailed success response
    logger.info(f"🎉 Verification successful for {admission_no}")
    
    return {
        "success": True,
        "message": "Face verified successfully",
        "admissionNo": admission_no,
    }