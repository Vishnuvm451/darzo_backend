import face_recognition
import cv2
import numpy as np
from firebase import get_db

# -------------------------------------------------
# EXTRACT FACE ENCODING
# -------------------------------------------------
def extract_face_encoding(image):
    """
    Converts an image into a normalized 128-d face vector.
    """
    if image is None:
        return None

    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    locations = face_recognition.face_locations(rgb, model="hog")
    if len(locations) != 1:
        return None

    encodings = face_recognition.face_encodings(rgb, locations)
    if not encodings:
        return None

    encoding = encodings[0]

    # 🔥 NORMALIZE (MUST MATCH REGISTRATION)
    encoding = encoding / np.linalg.norm(encoding)
    return encoding


# -------------------------------------------------
# VERIFY USER FACE (1:1 MATCH)
# -------------------------------------------------
def verify_user_face(admission_no, current_encoding, tolerance=0.6):
    db = get_db()

    target_id = str(admission_no)
    doc = db.collection("face_data").document(target_id).get()

    if not doc.exists:
        return False, "No face registered for this student"

    data = doc.to_dict()
    if "vector" not in data:
        return False, "Stored face data corrupted"

    stored_vector = np.array(data["vector"])

    # 🔐 SAFE EUCLIDEAN DISTANCE (NORMALIZED VECTORS)
    distance = np.linalg.norm(stored_vector - current_encoding)

    if distance <= tolerance:
        return True, "Face verified"
    else:
        confidence = max(0.0, 1.0 - distance)
        return False, f"Face mismatch (confidence: {confidence:.2%})"
