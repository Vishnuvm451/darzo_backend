from fastapi import APIRouter, UploadFile, File, Form, HTTPException
import numpy as np
import cv2
import asyncio
from functools import partial

from face_utils import extract_face_encoding, verify_user_face

router = APIRouter(tags=["Face Verification"])

MAX_IMAGE_SIZE = 2 * 1024 * 1024  # 2MB safety limit

# -------------------------------------------------
# VERIFY FACE (1:1 MATCHING)
# -------------------------------------------------
@router.post("/verify")
async def verify_face(
    admission_no: str = Form(...),   # ✅ received from Flutter
    image: UploadFile = File(...)
):
    """
    Verifies a student's face using 1:1 comparison.
    """

    try:
        # --------------------------------------------------
        # 1. READ IMAGE
        # --------------------------------------------------
        contents = await image.read()

        if not contents:
            raise HTTPException(status_code=400, detail="Empty image file")

        if len(contents) > MAX_IMAGE_SIZE:
            raise HTTPException(
                status_code=413,
                detail="Image too large"
            )

        np_arr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if img is None:
            raise HTTPException(
                status_code=400,
                detail="Invalid image format"
            )

        # Resize for speed (VERY IMPORTANT on Render)
        h, w, _ = img.shape
        if w > 800:
            scale = 800 / w
            img = cv2.resize(img, (int(w * scale), int(h * scale)))

        # --------------------------------------------------
        # 2. EXTRACT FACE ENCODING (NON-BLOCKING)
        # --------------------------------------------------
        loop = asyncio.get_event_loop()
        encoding = await loop.run_in_executor(
            None,
            partial(extract_face_encoding, img)
        )

        if encoding is None:
            raise HTTPException(
                status_code=400,
                detail="No face detected or image too blurry"
            )

        # --------------------------------------------------
        # 3. VERIFY AGAINST STORED VECTOR (1:1)
        # --------------------------------------------------
        success, message = verify_user_face(admission_no, encoding)

        if not success:
            raise HTTPException(
                status_code=401,
                detail=message
            )

        # --------------------------------------------------
        # 4. SUCCESS
        # --------------------------------------------------
        return {
            "success": True,
            "admissionNo": admission_no,
            "message": "Face verified successfully"
        }

    except HTTPException:
        # Let FastAPI return clean error responses
        raise

    except Exception as e:
        print("❌ CRITICAL ERROR IN /face/verify:", e)
        raise HTTPException(
            status_code=500,
            detail="Internal server error during verification"
        )
