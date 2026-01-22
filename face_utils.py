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
    Returns None if face is invalid.
    """

    if image is None:
        print("❌ extract_face_encoding: image is None")
        return None

    try:
        # Convert to RGB
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # 🔥 Resize for consistency (MATCHES REGISTRATION)
        h, w, _ = rgb.shape
        if w > 800:
            scale = 800 / w
            rgb = cv2.resize(rgb, (int(w * scale), int(h * scale)))

        # Detect faces (HOG = fast + mobile-safe)
        locations = face_recognition.face_locations(rgb, model="hog")

        # STRICT: exactly one face
        if len(locations) != 1:
            print("❌ extract_face_encoding: face count =", len(locations))
            return None

        encodings = face_recognition.face_encodings(
            rgb,
            known_face_locations=locations
        )

        if not encodings:
            print("❌ extract_face_encoding: encoding failed")
            return None

        encoding = encodings[0]

        # 🔥 NORMALIZE (CRITICAL)
        norm = np.linalg.norm(encoding)
        if norm == 0:
            print("❌ extract_face_encoding: zero norm")
            return None

        encoding = encoding / norm
        print("🧠 Face encoding extracted")

        return encoding

    except Exception as e:
        print("❌ ERROR IN extract_face_encoding:", e)
        return None


# -------------------------------------------------
# VERIFY USER FACE (1:1 MATCH)
# -------------------------------------------------
def verify_user_face(admission_no, current_encoding, tolerance=0.6):
    """
    Compares live face vector with stored vector (1:1).
    Uses Euclidean distance on normalized vectors.
    """

    try:
        db = get_db()
        target_id = str(admission_no)

        doc = db.collection("face_data").document(target_id).get()

        if not doc.exists:
            print("❌ verify_user_face: no face data for", target_id)
            return False, "No face registered for this student"

        data = doc.to_dict()
        if not data or "vector" not in data:
            print("❌ verify_user_face: corrupted face data")
            return False, "Stored face data corrupted"

        stored_vector = np.array(data["vector"], dtype=np.float32)
        current_vector = np.array(current_encoding, dtype=np.float32)

        # Shape safety
        if stored_vector.shape != current_vector.shape:
            print("❌ verify_user_face: vector shape mismatch")
            return False, "Face vector mismatch"

        # 🔐 EUCLIDEAN DISTANCE
        distance = np.linalg.norm(stored_vector - current_vector)

        print(f"🔐 Face distance for {admission_no}: {distance:.4f}")

        if distance <= tolerance:
            return True, "Face verified"

        confidence = max(0.0, 1.0 - distance)
        return False, f"Face mismatch (confidence: {confidence:.2%})"

    except Exception as e:
        print("❌ ERROR IN verify_user_face:", e)
        return False, "Face verification failed"
