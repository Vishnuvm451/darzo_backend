from fastapi import APIRouter, File, UploadFile, HTTPException, Form
import numpy as np
import cv2
import face_recognition
import asyncio
from functools import partial
from firebase_admin import firestore
from firebase import get_db

router = APIRouter(tags=["Face Registration"])

MAX_IMAGE_SIZE = 2 * 1024 * 1024  # 2MB

# -------------------------------------------------
# FACE VECTOR EXTRACTION (MOBILE-SAFE)
# -------------------------------------------------
def extract_face_vector(rgb_img):
    try:
        boxes = face_recognition.face_locations(rgb_img, model="hog")
        print("🧠 FACE BOXES:", boxes)

        if not boxes:
            return None  # ❗ DO NOT CRASH

        encodings = face_recognition.face_encodings(
            rgb_img,
            known_face_locations=boxes
        )

        if not encodings:
            return None

        embedding = encodings[0]

        # Normalize (IMPORTANT)
        norm = np.linalg.norm(embedding)
        if norm == 0:
            return None

        embedding = embedding / norm
        print("🧠 VECTOR GENERATED")

        return embedding.tolist()

    except Exception as e:
        print("❌ VECTOR EXTRACTION ERROR:", e)
        return None


# -------------------------------------------------
# REGISTER FACE
# -------------------------------------------------
@router.post("/register")
async def register_face(
    admission_no: str = Form(...),
    auth_uid: str = Form(...),
    image: UploadFile = File(...)
):
    print("🔥 /face/register CALLED")
    print("🔥 Writing face vector for:", admission_no)

    db = get_db()

    # -------------------------------------------------
    # 1. VALIDATE STUDENT
    # -------------------------------------------------
    student_ref = db.collection("student").document(admission_no)
    student_doc = student_ref.get()

    if not student_doc.exists:
        raise HTTPException(status_code=404, detail="Student not found")

    student_data = student_doc.to_dict()

    if student_data.get("authUid") != auth_uid:
        raise HTTPException(status_code=403, detail="Unauthorized access")

    if student_data.get("face_enabled") is True:
        raise HTTPException(status_code=409, detail="Face already registered")

    # -------------------------------------------------
    # 2. READ IMAGE
    # -------------------------------------------------
    contents = await image.read()
    print("📸 IMAGE SIZE:", len(contents))

    if not contents:
        raise HTTPException(status_code=400, detail="Empty image file")

    if len(contents) > MAX_IMAGE_SIZE:
        raise HTTPException(
            status_code=413,
            detail="Image too large. Use camera capture."
        )

    np_arr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    if img is None:
        raise HTTPException(status_code=400, detail="Invalid image data")

    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Resize for performance
    h, w, _ = rgb_img.shape
    if w > 800:
        scale = 800 / w
        rgb_img = cv2.resize(
            rgb_img,
            (int(w * scale), int(h * scale))
        )

    # -------------------------------------------------
    # 3. FACE PROCESSING (ASYNC SAFE)
    # -------------------------------------------------
    loop = asyncio.get_event_loop()
    face_vector = await loop.run_in_executor(
        None,
        partial(extract_face_vector, rgb_img)
    )

    if face_vector is None:
        raise HTTPException(
            status_code=400,
            detail="Face not detected clearly. Try again."
        )

    # -------------------------------------------------
    # 4. STORE FACE VECTOR
    # -------------------------------------------------
    db.collection("face_data").document(admission_no).set({
        "admissionNo": admission_no,
        "authUid": auth_uid,
        "vector": face_vector,
        "updatedAt": firestore.SERVER_TIMESTAMP
    })

    print("✅ FACE VECTOR STORED IN FIRESTORE")

    # -------------------------------------------------
    # 5. UPDATE STUDENT STATUS
    # -------------------------------------------------
    student_ref.update({
        "face_enabled": True,
        "face_registered_at": firestore.SERVER_TIMESTAMP
    })

    return {
        "success": True,
        "message": "Face registered successfully"
    }
