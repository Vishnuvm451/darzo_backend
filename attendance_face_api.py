from fastapi import APIRouter, UploadFile, File, Form, HTTPException
import numpy as np
import cv2
import face_recognition
from firebase import get_db

router = APIRouter(tags=["Face Verification"])

# =====================================================
# FACE VERIFICATION API (1:1 MATCHING)
# =====================================================
@router.post("/verify")
async def verify_face(
    admission_no: str = Form(...),  # <--- Matches Flutter request
    image: UploadFile = File(...)
):
    """
    1. Fetches stored vector for THIS student (1:1 Verification)
    2. Compares with uploaded face
    3. Returns Success/Fail (Flutter handles the marking)
    """
    db = get_db()

    # --------------------------------------------------
    # 1. FETCH STORED VECTOR (The "Truth")
    # --------------------------------------------------
    # We look up the face data directly using Admission No.
    # This is O(1) speed (Instant).
    face_ref = db.collection("face_data").document(admission_no)
    face_doc = face_ref.get()

    if not face_doc.exists:
        raise HTTPException(status_code=404, detail="Face data not found. Please register first.")

    stored_data = face_doc.to_dict()
    stored_vector = stored_data.get("vector")

    if not stored_vector:
        raise HTTPException(status_code=500, detail="Stored face data is corrupt.")

    # --------------------------------------------------
    # 2. PROCESS INCOMING IMAGE (The "Query")
    # --------------------------------------------------
    try:
        contents = await image.read()
        np_arr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if img is None:
            raise HTTPException(status_code=400, detail="Invalid image file")
            
        # Convert to RGB (face_recognition requirement)
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Detect Face
        # 'hog' is faster and sufficient for 1:1 matching
        boxes = face_recognition.face_locations(rgb_img, model="hog")
        
        if len(boxes) == 0:
            raise HTTPException(status_code=400, detail="No face detected in photo.")
        
        if len(boxes) > 1:
            raise HTTPException(status_code=400, detail="Multiple faces detected. One person only.")

        # Encode Face
        encodings = face_recognition.face_encodings(rgb_img, known_face_locations=boxes)
        if not encodings:
            raise HTTPException(status_code=400, detail="Could not encode face.")
            
        new_vector = encodings[0]

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Image processing error: {str(e)}")

    # --------------------------------------------------
    # 3. COMPARE VECTORS (The Match)
    # --------------------------------------------------
    # Euclidean distance: Lower is better.
    # 0.0 = Perfect match. > 0.6 = Different person.
    distance = face_recognition.face_distance([stored_vector], new_vector)[0]
    
    # Strict threshold for security (0.5 is tighter than standard 0.6)
    THRESHOLD = 0.5 
    is_match = distance <= THRESHOLD

    if not is_match:
        raise HTTPException(status_code=401, detail="Face mismatch. Verification failed.")

    # --------------------------------------------------
    # 4. SUCCESS RESPONSE
    # --------------------------------------------------
    # We return 200 OK. The Flutter app will see this and 
    # immediately trigger the attendance marking transaction.
    return {
        "success": True,
        "admissionNo": admission_no,
        "match_score": float(1 - distance),
        "message": "Face verified successfully"
    }
