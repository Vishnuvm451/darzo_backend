from fastapi import APIRouter, UploadFile, File, Form, HTTPException
import numpy as np
import cv2
import asyncio

from face_utils import extract_face_encoding, verify_user_face

router = APIRouter(tags=["Face Verification"])

MAX_IMAGE_SIZE = 2 * 1024 * 1024  # 2MB

# -------------------------------------------------
# VERIFY FACE (1:1 MATCHING)
# -------------------------------------------------
@router.post("/verify")
async def verify_face(
    admission_no: str = Form(...),
    image: UploadFile = File(...)
):
    """
    Verifies a student's face using stored vector (1:1).
    Used during attendance marking.
    """

    print("🔍 /face/verify CALLED:", admission_no)

    # --------------------------------------------------
    # 1. READ IMAGE
    # --------------------------------------------------
    contents = await image.read()

    if not contents:
        raise HTTPException(status_code=400, detail="Empty image file")

    print("📸 IMAGE SIZE:", len(contents))

    if len(contents) > MAX_IMAGE_SIZE:
        raise HTTPException(
            status_code=413,
            detail="Image too large"
        )

    img = cv2.imdecode(np.frombuffer(contents, np.uint8), cv2.IMREAD_COLOR)

    if img is None:
        raise HTTPException(
            status_code=400,
            detail="Invalid image format"
        )

    # Resize for performance (Render-safe)
    h, w, _ = img.shape
    if w > 800:
        scale = 800 / w
        img = cv2.resize(img, (int(w * scale), int(h * scale)))

    # --------------------------------------------------
    # 2. EXTRACT FACE ENCODING (SAFE THREAD)
    # --------------------------------------------------
    try:
        encoding = await asyncio.to_thread(extract_face_encoding, img)
    except Exception as e:
        print("❌ FACE ENCODING ERROR:", e)
        raise HTTPException(
            status_code=500,
            detail="Face encoding failed"
        )

    if encoding is None:
        raise HTTPException(
            status_code=400,
            detail="No face detected or unclear image"
        )

    print("🧠 LIVE VECTOR EXTRACTED")

    # --------------------------------------------------
    # 3. VERIFY AGAINST STORED VECTOR
    # --------------------------------------------------
    success, message = verify_user_face(admission_no, encoding)

    print("🔐 VERIFY RESULT:", success, "|", message)

    if not success:
        raise HTTPException(
            status_code=401,
            detail=message
        )

    # --------------------------------------------------
    # 4. SUCCESS
    # --------------------------------------------------
    print("✅ FACE VERIFIED:", admission_no)

    return {
        "success": True,
        "admissionNo": admission_no,
        "message": "Face verified successfully"
    }
