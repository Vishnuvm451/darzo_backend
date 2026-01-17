import os
import pickle
import face_recognition
import cv2
import numpy as np

# -------------------------------------------------
# LOCAL FACE DATABASE
# -------------------------------------------------
FACES_DB = "faces_db.pkl"


# -------------------------------------------------
# LOAD KNOWN FACES
# -------------------------------------------------
def load_known_faces():
    """
    Loads saved face encodings and admission numbers
    """
    if not os.path.exists(FACES_DB):
        return [], []

    with open(FACES_DB, "rb") as f:
        data = pickle.load(f)

    return data.get("encodings", []), data.get("admissions", [])


# -------------------------------------------------
# REGISTER FACE ENCODING
# -------------------------------------------------
def register_face_encoding(image, admission_no):
    """
    Extract and store face encoding for a student
    """

    # 🛑 Safety check
    if image is None:
        return False, "Invalid image"

    # Convert BGR → RGB (safe for dlib)
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Detect face locations
    locations = face_recognition.face_locations(rgb)

    if len(locations) != 1:
        return False, "Exactly one face must be visible"

    # Extract encoding
    encodings = face_recognition.face_encodings(rgb, locations)

    if not encodings:
        return False, "Face encoding failed"

    encoding = np.array(encodings[0])

    # Load or create DB
    if os.path.exists(FACES_DB):
        with open(FACES_DB, "rb") as f:
            data = pickle.load(f)
    else:
        data = {"encodings": [], "admissions": []}

    # Prevent duplicate registration
    if admission_no in data["admissions"]:
        return False, "Face already registered"

    # Save encoding
    data["encodings"].append(encoding)
    data["admissions"].append(admission_no)

    with open(FACES_DB, "wb") as f:
        pickle.dump(data, f)

    return True, "Face registered successfully"


# -------------------------------------------------
# EXTRACT FACE ENCODING
# -------------------------------------------------
def extract_face_encoding(image):
    """
    Extract face encoding from image
    """

    if image is None:
        return None

    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    locations = face_recognition.face_locations(rgb)

    if len(locations) != 1:
        return None

    encodings = face_recognition.face_encodings(rgb, locations)
    return np.array(encodings[0]) if encodings else None


# -------------------------------------------------
# COMPARE FACE ENCODINGS
# -------------------------------------------------
def compare_faces(
    unknown_encoding,
    known_encodings,
    admissions,
    tolerance=0.45
):
    """
    Compare unknown face with known faces
    Returns admission number if matched
    """

    if unknown_encoding is None or not known_encodings:
        return None

    results = face_recognition.compare_faces(
        known_encodings,
        unknown_encoding,
        tolerance=tolerance
    )

    for i, matched in enumerate(results):
        if matched:
            return admissions[i]

    return None
