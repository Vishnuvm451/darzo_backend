from fastapi import APIRouter, UploadFile, File, Form, HTTPException
import numpy as np
import cv2
from face_utils import extract_face_encoding, verify_user_face

router = APIRouter(tags=["Face Verification"])

@router.post("/verify")
async def verify_face(
    admission_no: str = Form(...), # ✅ Now receives admission_no from Flutter
    image: UploadFile = File(...)
):
    """
    Verifies a specific student's face using 1:1 matching.
    """
    try:
        # --------------------------------------------------
        # 1. READ IMAGE
        # --------------------------------------------------
        contents = await image.read()
        np_arr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if img is None:
            return {
                "success": False,
                "message": "Invalid image format"
            }

        # --------------------------------------------------
        # 2. EXTRACT ENCODING (THE "QUERY" FACE)
        # --------------------------------------------------
        # This converts the photo into a 128-d vector
        encoding = extract_face_encoding(img)

        if encoding is None:
            return {
                "success": False,
                "message": "No face detected or image too blurry"
            }

        # --------------------------------------------------
        # 3. 1:1 VERIFICATION (THE "BRAIN" STEP)
        # --------------------------------------------------
        # This function fetches the vector from Firestore and compares it
        success, message = verify_user_face(admission_no, encoding)

        if not success:
            return {
                "success": False,
                "message": message
            }

        # --------------------------------------------------
        # 4. SUCCESS
        # --------------------------------------------------
        return {
            "success": True,
            "admissionNo": admission_no,
            "message": "Face verified successfully"
        }

    except Exception as e:
        print(f"CRITICAL ERROR IN /VERIFY: {e}")
        return {
            "success": False,
            "message": "Internal server error during verification"
        }
