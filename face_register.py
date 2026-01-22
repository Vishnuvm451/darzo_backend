from fastapi import APIRouter, File, UploadFile, HTTPException, Form
import numpy as np
import cv2
import face_recognition
import asyncio
from firebase_admin import firestore
from firebase import get_db

router = APIRouter()

MAX_IMAGE_SIZE = 2 * 1024 * 1024  # 2MB

# -------------------------------------------------
# FACE VECTOR
# -------------------------------------------------
def extract_face_vector(rgb_img):
    boxes = face_recognition.face_locations(rgb_img, model="hog")

    if len(boxes) != 1:
        raise ValueError("Exactly one face required")

    encodings = face_recognition.face_encodings(rgb_img, boxes)
    if not encodings:
        raise ValueError("Face encoding failed")

    vec = encodings[0]
    vec = vec / np.linalg.norm(vec)
    return vec.tolist()

# -------------------------------------------------
# REGISTER FACE
# -------------------------------------------------
@router.post("/register")
async def register_face(
    admission_no: str = Form(...),
    auth_uid: str = Form(...),
    image: UploadFile = File(...)
):
    print("🔥 /face/register CALLED:", admission_no)
    db = get_db()

    student_ref = db.collection("student").document(admission_no)
    student_doc = student_ref.get()

    if not student_doc.exists:
        raise HTTPException(404, "Student not found")

    student = student_doc.to_dict()
    if student.get("authUid") != auth_uid:
        raise HTTPException(403, "Unauthorized")

    if student.get("face_enabled") is True:
        raise HTTPException(409, "Face already registered")

    contents = await image.read()
    print("📸 IMAGE SIZE:", len(contents))

    if not contents or len(contents) > MAX_IMAGE_SIZE:
        raise HTTPException(400, "Invalid image")

    img = cv2.imdecode(np.frombuffer(contents, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(400, "Invalid image data")

    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    try:
        face_vector = await asyncio.to_thread(extract_face_vector, rgb)
    except ValueError as e:
        raise HTTPException(400, str(e))
    except Exception as e:
        print("❌ FACE ERROR:", e)
        raise HTTPException(500, "Face processing failed")

    print("🧠 VECTOR LENGTH:", len(face_vector))

    # ---------------- STORE VECTOR ----------------
    db.collection("face_data").document(admission_no).set({
        "admissionNo": admission_no,
        "authUid": auth_uid,
        "vector": face_vector,
        "updatedAt": firestore.SERVER_TIMESTAMP
    })

    print("✅ FACE VECTOR STORED")

    student_ref.update({
        "face_enabled": True,
        "face_registered_at": firestore.SERVER_TIMESTAMP
    })

    print("✅ STUDENT UPDATED")

    return {"success": True}
