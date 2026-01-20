from fastapi import APIRouter, File, UploadFile, HTTPException, Form
import numpy as np
import cv2
import face_recognition  # 🚀 REQUIRED for Vectorization
from datetime import datetime
from firebase import get_db

# =====================================================
# ROUTER
# =====================================================
router = APIRouter(tags=["Face Registration"])

# =====================================================
# FACE REGISTRATION API
# =====================================================
@router.post("/register")
async def register_face(
    admission_no: str = Form(...), 
    auth_uid: str = Form(...),
    image: UploadFile = File(...)
):
    """
    1. Validates Student
    2. Detects Face & Generates Vector (128-d)
    3. Saves Vector to Firestore 'face_data' collection
    4. Updates 'student' collection flag
    """

    db = get_db()

    # --------------------------------------------------
    # 1. VALIDATE STUDENT RECORD
    # --------------------------------------------------
    student_ref = db.collection("student").document(admission_no)
    student_doc = student_ref.get()

    if not student_doc.exists:
        raise HTTPException(status_code=404, detail="Student not found")

    student_data = student_doc.to_dict()

    # Security Check: Ensure the user registering is the owner of the account
    if student_data.get("authUid") != auth_uid:
        raise HTTPException(status_code=403, detail="Auth UID mismatch. Unauthorized.")

    # Prevent re-registration (Optional: remove if you want to allow overwrites)
    if student_data.get("face_enabled") is True:
        raise HTTPException(status_code=400, detail="Face already registered.")

    # --------------------------------------------------
    # 2. READ & DECODE IMAGE
    # --------------------------------------------------
    try:
        contents = await image.read()
        np_arr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if img is None:
            raise HTTPException(status_code=400, detail="Invalid or corrupt image file")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Image processing error: {str(e)}")

    # --------------------------------------------------
    # 3. VECTORIZATION (THE "BRAIN" STEP) 🧠
    # --------------------------------------------------
    # Convert BGR (OpenCV format) to RGB (face_recognition format)
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Detect face locations first (HOG algorithm is faster)
    boxes = face_recognition.face_locations(rgb_img, model="hog")

    if len(boxes) == 0:
        raise HTTPException(status_code=400, detail="No face detected. Please try again.")
    
    if len(boxes) > 1:
        raise HTTPException(status_code=400, detail="Multiple faces detected. Only one person allowed.")

    # Generate the 128-dimension encoding (Vector)
    # This returns a list of arrays, we take the first one [0]
    encodings = face_recognition.face_encodings(rgb_img, known_face_locations=boxes)
    
    if not encodings:
        raise HTTPException(status_code=400, detail="Could not encode face features.")

    # Convert numpy array to standard Python list for Firestore (JSON) compatibility
    face_vector = encodings[0].tolist()

    # --------------------------------------------------
    # 4. SAVE VECTOR TO FIRESTORE ('face_data')
    # --------------------------------------------------
    # This automatically creates the 'face_data' collection if it doesn't exist
    face_data_ref = db.collection("face_data").document(admission_no)
    
    face_data_ref.set({
        "admissionNo": admission_no,
        "authUid": auth_uid,
        "vector": face_vector,  # <--- Storing the math here
        "updatedAt": datetime.now()
    })

    # --------------------------------------------------
    # 5. UPDATE STUDENT FLAG
    # --------------------------------------------------
    student_ref.update({
        "face_enabled": True,
        "face_registered_at": datetime.now()
    })

    # --------------------------------------------------
    # 6. RESPONSE
    # --------------------------------------------------
    return {
        "success": True,
        "admissionNo": admission_no,
        "message": "Face vector stored and registration complete"
    }
