from fastapi import APIRouter, UploadFile, File, Form, HTTPException
import numpy as np
import cv2
from datetime import datetime

from firebase import get_db
from face_utils import (
    load_known_faces,
    extract_face_encoding,
    compare_faces
)

router = APIRouter(prefix="/attendance", tags=["Attendance"])

@router.post("/mark")
async def mark_attendance_by_face(
    student_uid: str = Form(...),
    class_id: str = Form(...),
    session_type: str = Form(...),
    image: UploadFile = File(...)
):
    """
    1. Verifies Face
    2. Checks if Face matches Student UID
    3. Marks Attendance in Firestore
    """
    db = get_db()

    # --------------------------------------------------
    # 1. READ IMAGE
    # --------------------------------------------------
    contents = await image.read()
    np_arr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    if img is None:
        raise HTTPException(status_code=400, detail="Invalid image")

    # --------------------------------------------------
    # 2. FACE RECOGNITION
    # --------------------------------------------------
    # Reload faces to get latest data
    known_encodings, known_admissions = load_known_faces()
    
    if not known_encodings:
        raise HTTPException(status_code=400, detail="No registered faces in system")

    encoding = extract_face_encoding(img)
    if encoding is None:
        raise HTTPException(status_code=400, detail="No face detected")

    matched_admission = compare_faces(encoding, known_encodings, known_admissions)

    if matched_admission is None:
        raise HTTPException(status_code=400, detail="Face not recognized")

    # --------------------------------------------------
    # 3. VERIFY USER IDENTITY (Security Check)
    # --------------------------------------------------
    # Ensure the face belongs to the user currently logged in
    student_ref = db.collection("student").document(matched_admission)
    student_doc = student_ref.get()

    if not student_doc.exists:
        raise HTTPException(status_code=404, detail="Student record not found")

    if student_doc.to_dict().get("authUid") != student_uid:
        raise HTTPException(status_code=403, detail="Face does not match logged-in user")

    # --------------------------------------------------
    # 4. MARK ATTENDANCE IN FIRESTORE
    # --------------------------------------------------
    # Generate Session ID: classId_YYYY-MM-DD_sessionType
    now = datetime.now()
    date_str = now.strftime("%Y-%m-%d")
    session_id = f"{class_id}_{date_str}_{session_type}"

    # Check if session exists (Optional: can remove if you want to allow anytime)
    session_ref = db.collection("attendance_session").document(session_id)
    session_snap = session_ref.get()

    if not session_snap.exists or not session_snap.to_dict().get("isActive", False):
         raise HTTPException(status_code=400, detail="Attendance session is not active")

    # Mark the student present
    attendance_ref = db.collection("attendance").document(session_id).collection("student").document(matched_admission)
    
    attendance_ref.set({
        "studentId": matched_admission,
        "status": "present",
        "method": "face",
        "markedAt": firestore.SERVER_TIMESTAMP if 'firestore' in globals() else datetime.now(),
        "confidence": "high"
    }, merge=True)

    return {
        "success": True,
        "message": "Attendance marked successfully",
        "admissionNo": matched_admission
    }