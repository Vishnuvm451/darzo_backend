from fastapi import APIRouter, UploadFile, File, Form, HTTPException
import numpy as np
import cv2
from face_utils import extract_face_encoding, verify_user_face

router = APIRouter(tags=["Face Attendance Verification"])

@router.post("/verify")
async def verify_face(
    admission_no: str = Form(...),
    image: UploadFile = File(...)
):
    """
    Attendance-time face verification (1:1).
    Uses SAME pipeline as registration verification.
    """
    try:
        # 1. Decode image
        contents = await image.read()
        np_arr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if img is None:
            raise HTTPException(status_code=400, detail="Invalid image")

        # 2. Extract normalized encoding
        encoding = extract_face_encoding(img)
        if encoding is None:
            raise HTTPException(
                status_code=400,
                detail="No face detected or unclear image"
            )

        # 3. Verify using SAME logic
        success, message = verify_user_face(admission_no, encoding)

        if not success:
            raise HTTPException(status_code=401, detail=message)

        # 4. Success
        return {
            "success": True,
            "admissionNo": admission_no,
            "message": "Attendance face verified"
        }

    except HTTPException:
        raise
    except Exception as e:
        print(f"[FACE ATTENDANCE ERROR] {e}")
        raise HTTPException(
            status_code=500,
            detail="Internal verification error"
        )
