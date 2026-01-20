import face_recognition
import cv2
import numpy as np
from firebase import get_db

# -------------------------------------------------
# EXTRACT FACE ENCODING
# -------------------------------------------------
def extract_face_encoding(image):
    """
    Converts an image into a 128-d mathematical vector.
    """
    if image is None:
        return None

    # Convert BGR (OpenCV) to RGB (Face Recognition requirement)
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # HOG is fast on CPUs. Locations returns [(top, right, bottom, left)]
    locations = face_recognition.face_locations(rgb, model="hog")

    if len(locations) != 1:
        # Fails if 0 faces or 2+ faces are present for security/clarity
        return None

    encodings = face_recognition.face_encodings(rgb, locations)
    
    return np.array(encodings[0]) if encodings else None


# -------------------------------------------------
# COMPARE FACE ENCODINGS (1:1 MATCHING)
# -------------------------------------------------
def verify_user_face(admission_no, current_encoding, tolerance=0.5):
    """
    Fetches the specific student vector and compares it with the current scan.
    """
    db = get_db()
    
    # Ensure admission_no is a string for the document ID
    target_id = str(admission_no)
    
    # 1. Fetch only the relevant document from Firestore
    face_ref = db.collection("face_data").document(target_id)
    doc = face_ref.get()
    
    if not doc.exists:
        return False, "No face registered for this student"

    data = doc.to_dict()
    if "vector" not in data:
        return False, "Stored face data is incomplete/corrupt"

    # 2. Convert stored list back to numpy array for calculation
    stored_vector = np.array(data.get("vector"))

    # 3. Calculate Euclidean Distance (0.0 is a perfect match)
    # 
    distance = face_recognition.face_distance([stored_vector], current_encoding)[0]
    
    # Lower distance = Higher confidence
    is_match = distance <= tolerance
    
    if is_match:
        return True, "Match found"
    else:
        # We return a confidence score (1 - distance) for debugging purposes
        confidence = max(0, 1 - distance)
        return False, f"Face mismatch (Confidence: {confidence:.2%})"
