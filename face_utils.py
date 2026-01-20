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
    Used during Verification.
    """
    if image is None:
        return None

    # Convert BGR (OpenCV) to RGB (Face Recognition)
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Use HOG model for speed on CPU-based servers (Render/Heroku)
    locations = face_recognition.face_locations(rgb, model="hog")

    if len(locations) != 1:
        return None

    encodings = face_recognition.face_encodings(rgb, locations)
    
    # Return as a numpy array for math comparison
    return np.array(encodings[0]) if encodings else None


# -------------------------------------------------
# COMPARE FACE ENCODINGS (1:1 MATCHING)
# -------------------------------------------------
def verify_user_face(admission_no, current_encoding, tolerance=0.5):
    """
    Fetches the SPECIFIC vector from Firestore and compares it.
    Returns True if it matches, False otherwise.
    """
    db = get_db()
    
    # 1. Fetch only the relevant document
    face_ref = db.collection("face_data").document(admission_no)
    doc = face_ref.get()
    
    if not doc.exists:
        return False, "No face registered for this student"

    # 2. Get the stored vector (List -> Numpy Array)
    stored_vector = np.array(doc.to_dict().get("vector"))

    # 3. Calculate Euclidean Distance
    # Lower distance means more similar
    distance = face_recognition.face_distance([stored_vector], current_encoding)[0]
    
    is_match = distance <= tolerance
    
    if is_match:
        return True, "Match found"
    else:
        return False, f"Face mismatch (Score: {1-distance:.2f})"
