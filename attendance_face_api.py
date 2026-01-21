from fastapi import APIRouter, UploadFile, File, Form, HTTPException
import numpy as np
import cv2
import asyncio
from functools import partial

from face_utils import extract_face_encoding, verify_user_face

router = APIRouter(tags=["Face Attendance Verification"])

# -------------------------------------------------
# FACE ATTENDANCE VERIFY (1:1 MATCH)
# -------------------------------------------------
@router.post("/verify")
async def verify_face(
    admission_no: str = Form(...),
    image: UploadFile = File(...)
):
    """
    Attendance-time face verification (1:1).
    Uses the SAME encoding + verification pipeline
    as face registration.
    """

    try:
        # -------------------------------------------------
        # 1. READ IMAGE
        # -------------------------------------------------
        contents = await image.read()

        if not contents:
            raise HTTPException(
                status_code=400,
                detail="Empty image file"
            )

        np_arr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if img is None:
            raise HTTPException(
                status_code=400,
                detail="Invalid image format"
            )

        # -------------------------------------------------
        # 2. EXTRACT FACE ENCODING (NON-BLOCKING)
        # -------------------------------------------------
        loop = asyncio.get_event_loop()

        encoding = await loop.run_in_executor(
            None,
            partial(extract_face_encoding, img)
        )

        if encoding is None:
            raise HTTPException(
                status_code=400,
                detail="No face detected or unclear image"
            )

        # -------------------------------------------------
        # 3. VERIFY FACE (1:1 MATCH)
        # -------------------------------------------------
        success, message = verify_user_face(admission_no, encoding)

        if not success:
            raise HTTPException(
                status_code=401,
                detail=message
            )

        # -------------------------------------------------
        # 4. SUCCESS
        # -------------------------------------------------
        return {
            "success": True,
            "admissionNo": admission_no,
            "message": "Attendance face verified"
        }

    except HTTPException:
        # Re-raise known errors
        raise

    except Exception as e:
        print(f"[FACE ATTENDANCE ERROR] {e}")
        raise HTTPException(
            status_code=500,
            detail="Internal verification error"
        )
