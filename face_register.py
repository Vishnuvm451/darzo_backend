from fastapi import APIRouter, File, UploadFile, HTTPException, Form
import numpy as np
import cv2
import face_recognition
from firebase_admin import firestore
from firebase import get_db

router = APIRouter(tags=["Face Registration"])

@router.post("/register")
async def register_face(
    admission_no: str = Form(...),
    auth_uid: str = Form(...),
    image: UploadFile = File(...)
):
    db = get_db()

    # 1. Validate student
    student_ref = db.collection("student").document(admission_no)
    student_doc = student_ref.get()

    if not student_doc.exists:
        raise HTTPException(status_code=404, detail="Student not found")

    student_data = student_doc.to_dict()

    if student_data.get("authUid") != auth_uid:
        raise HTTPException(status_code=403, detail="Unauthorized")

    if student_data.get("face_enabled") is True:
        raise HTTPException(status_code=400, detail="Face already registered")

    # 2. Decode image
    contents = await image.read()
    np_arr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    if img is None:
        raise HTTPException(status_code=400, detail="Invalid image")

    # 3. Vectorization
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    boxes = face_recognition.face_locations(rgb_img, model="hog")

    if len(boxes) == 0:
        raise HTTPException(status_code=400, detail="No face detected")

    if len(boxes) > 1:
        raise HTTPException(status_code=400, detail="Multiple faces detected")

    top, right, bottom, left = boxes[0]
    if (bottom - top) < 80 or (right - left) < 80:
        raise HTTPException(
            status_code=400,
            detail="Face too far from camera"
        )

    encodings = face_recognition.face_encodings(
        rgb_img,
        known_face_locations=boxes
    )

    if not encodings:
        raise HTTPException(status_code=400, detail="Encoding failed")

    # Normalize embedding
    embedding = encodings[0]
    embedding = embedding / np.linalg.norm(embedding)
    face_vector = embedding.tolist()

    # 4. Store vector
    db.collection("face_data").document(admission_no).set({
        "admissionNo": admission_no,
        "authUid": auth_uid,
        "vector": face_vector,
        "updatedAt": firestore.SERVER_TIMESTAMP
    })

    # 5. Update student
    student_ref.update({
        "face_enabled": True,
        "face_registered_at": firestore.SERVER_TIMESTAMP
    })

    return {
        "success": True,
        "message": "Face registered successfully"
    }
