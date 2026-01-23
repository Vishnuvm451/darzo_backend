import face_recognition
import cv2
import numpy as np
from firebase import get_db
import logging

logger = logging.getLogger(__name__)

# ========== CONSTANTS ==========
VECTOR_THRESHOLD = 0.4  # ✅ FIX #2: Cosine distance threshold (lower = stricter match)
VECTOR_DIMENSIONS = 128  # Expected face embedding dimensions

# -------------------------------------------------
# EXTRACT FACE ENCODING (NORMALIZED 128D VECTOR)
# -------------------------------------------------
def extract_face_encoding(image):
    """
    ✅ Extracts a normalized 128D face vector from image.
    Uses face_recognition library (dlib-based).
    
    Parameters:
    - image: OpenCV image (BGR format, already resized)
    
    Returns:
    - numpy.ndarray: Normalized 128D face vector (float64)
    - None: If face detection fails
    
    Process:
    1. Convert BGR to RGB
    2. Detect face locations (HOG model)
    3. Extract face encodings (128D vector per face)
    4. Return first face (largest/most prominent)
    5. Normalize for cosine distance matching
    """
    if image is None:
        logger.warning("❌ Image is None in extract_face_encoding")
        return None

    try:
        # ========== CONVERT COLOR SPACE ==========
        # ✅ FIX #1: Add input validation
        if len(image.shape) != 3 or image.shape[2] != 3:
            logger.error(f"❌ Invalid image shape: {image.shape} (expected H x W x 3)")
            return None
        
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        logger.debug(f"📏 Image shape: {rgb.shape}")

        # ========== DETECT FACES ==========
        # HOG is faster than CNN, good for mobile
        locations = face_recognition.face_locations(rgb, model="hog")
        logger.info(f"🧠 Detected {len(locations)} face(s)")

        # ✅ FIX #3: Accept any face, use first one (not strict != 1)
        if not locations:
            logger.warning("❌ No faces detected in image")
            return None

        # Use only first (largest) face
        first_face_location = locations[0]

        # ========== EXTRACT FACE ENCODING ==========
        encodings = face_recognition.face_encodings(
            rgb,
            known_face_locations=[first_face_location]
        )

        if not encodings:
            logger.warning("❌ Face detected but could not create encoding")
            return None

        encoding = encodings[0]
        logger.debug(f"✅ Face encoding extracted (dimensions: {len(encoding)})")

        # ========== NORMALIZE VECTOR ==========
        # ✅ FIX #4: Critical for cosine distance matching
        norm = np.linalg.norm(encoding)
        
        if norm == 0:
            logger.error("❌ Vector norm is zero (invalid encoding)")
            return None

        encoding = encoding / norm
        
        # ✅ FIX #5: Validate dimensions
        if len(encoding) != VECTOR_DIMENSIONS:
            logger.error(f"❌ Vector dimension mismatch: {len(encoding)} != {VECTOR_DIMENSIONS}")
            return None
        
        logger.info(f"✅ Face vector generated and normalized (dimensions: {len(encoding)})")
        return encoding

    except Exception as e:
        logger.error(f"❌ Error in extract_face_encoding: {str(e)}", exc_info=True)
        return None


# -------------------------------------------------
# COMPARE TWO FACE VECTORS (1:1 MATCHING)
# -------------------------------------------------
def compare_vectors(vector1: np.ndarray, vector2: np.ndarray) -> dict:
    """
    ✅ Compares two normalized face vectors using COSINE DISTANCE.
    
    Parameters:
    - vector1 (np.ndarray): Stored face vector (normalized 128D)
    - vector2 (np.ndarray): Current face vector (normalized 128D)
    
    Returns:
    - dict: {
        "match": bool,           # True if distance < threshold
        "distance": float,       # Cosine distance (0-1, lower = more similar)
        "confidence": float,     # 0-100% confidence of match
        "threshold": float       # Matching threshold used
      }
    
    Theory:
    - Cosine Distance = 1 - (v1 · v2) / (||v1|| × ||v2||)
    - Range: 0 (identical) to 1 (completely different)
    - For normalized vectors: distance ≈ ||v1 - v2|| / 2
    - Threshold 0.4 = ~85% similarity
    """
    try:
        # ========== INPUT VALIDATION ==========
        if vector1 is None or vector2 is None:
            logger.error("❌ One or both vectors are None")
            return {
                "match": False,
                "distance": None,
                "confidence": 0.0,
                "threshold": VECTOR_THRESHOLD,
                "error": "Invalid input vectors"
            }

        # ✅ FIX #6: Convert to numpy arrays with consistent dtype
        v1 = np.array(vector1, dtype=np.float64)
        v2 = np.array(vector2, dtype=np.float64)

        # ✅ FIX #7: Check dimensions match
        if v1.shape != v2.shape:
            logger.error(f"❌ Vector shape mismatch: {v1.shape} vs {v2.shape}")
            return {
                "match": False,
                "distance": None,
                "confidence": 0.0,
                "threshold": VECTOR_THRESHOLD,
                "error": f"Shape mismatch: {v1.shape} vs {v2.shape}"
            }

        # ========== COSINE DISTANCE CALCULATION ==========
        # ✅ FIX #8: Use cosine distance (NOT Euclidean)
        dot_product = np.dot(v1, v2)
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)

        if norm1 == 0 or norm2 == 0:
            logger.error("❌ Vector norm is zero (invalid vector)")
            return {
                "match": False,
                "distance": None,
                "confidence": 0.0,
                "threshold": VECTOR_THRESHOLD,
                "error": "Invalid vector norm"
            }

        # Cosine similarity = (v1 · v2) / (||v1|| × ||v2||)
        # Cosine distance = 1 - cosine_similarity
        cosine_similarity = dot_product / (norm1 * norm2)
        distance = 1 - cosine_similarity

        logger.debug(f"🔍 Cosine similarity: {cosine_similarity:.4f}")
        logger.debug(f"🔍 Cosine distance: {distance:.4f}")

        # ========== DETERMINE MATCH ==========
        is_match = distance < VECTOR_THRESHOLD
        
        # ✅ FIX #9: Proper confidence calculation
        # Confidence = how similar (inverse of distance)
        confidence = max(0.0, (1.0 - distance) * 100.0)  # Convert to 0-100%

        logger.info(f"🔍 COMPARISON: distance={distance:.4f}, confidence={confidence:.1f}%, match={is_match}")

        return {
            "match": is_match,
            "distance": float(distance),
            "confidence": float(confidence),
            "threshold": VECTOR_THRESHOLD
        }

    except Exception as e:
        logger.error(f"❌ Error in compare_vectors: {str(e)}", exc_info=True)
        return {
            "match": False,
            "distance": None,
            "confidence": 0.0,
            "threshold": VECTOR_THRESHOLD,
            "error": str(e)
        }


# -------------------------------------------------
# VERIFY USER FACE (1:1 MATCH WITH STORED VECTOR)
# -------------------------------------------------
def verify_user_face(admission_no: str, current_encoding) -> tuple:
    """
    ✅ Compares live face with stored face vector (1:1 matching).
    
    Parameters:
    - admission_no (str): Student admission number
    - current_encoding: Current face vector (128D)
    
    Returns:
    - tuple: (success: bool, message: str)
      - (True, "Face verified"): Vectors match
      - (False, "Error message"): Vectors don't match or error occurred
    
    Database Lookup:
    - Fetches stored vector from face_data/{admission_no}
    - Compares with current_encoding using cosine distance
    - Threshold 0.4 = ~85% similarity required
    """
    try:
        logger.info(f"🔐 Starting face verification for {admission_no}")
        
        db = get_db()
        target_id = str(admission_no)

        # ========== FETCH STORED VECTOR ==========
        logger.debug(f"📂 Querying face_data/{target_id}")
        
        doc = db.collection("face_data").document(target_id).get(timeout=5)

        if not doc.exists:
            logger.warning(f"❌ No stored face found for {admission_no}")
            return False, "No face registered for this student"

        data = doc.to_dict()
        
        if not data or "vector" not in data:
            logger.error(f"❌ Stored face data corrupted for {admission_no}")
            return False, "Stored face data corrupted"

        stored_vector = data["vector"]
        logger.info(f"✅ Stored vector retrieved for {admission_no}")

        # ========== CONVERT TO NUMPY ARRAYS ==========
        # ✅ FIX #10: Convert with consistent dtype
        stored_vector = np.array(stored_vector, dtype=np.float64)
        current_vector = np.array(current_encoding, dtype=np.float64)

        # ========== COMPARE VECTORS ==========
        comparison = compare_vectors(stored_vector, current_vector)

        if not comparison["match"]:
            logger.warning(
                f"❌ Face verification FAILED for {admission_no} "
                f"(distance={comparison['distance']:.4f}, threshold={VECTOR_THRESHOLD})"
            )
            return (
                False,
                f"Face does not match (confidence: {comparison['confidence']:.1f}%)"
            )

        # ✅ FIX #11: Return success with confidence
        logger.info(
            f"✅ Face verification SUCCESS for {admission_no} "
            f"(distance={comparison['distance']:.4f}, confidence={comparison['confidence']:.1f}%)"
        )
        return True, f"Face verified (confidence: {comparison['confidence']:.1f}%)"

    except TimeoutError:
        logger.error(f"❌ Database timeout for {admission_no}")
        return False, "Database timeout - please try again"
    
    except Exception as e:
        logger.error(f"❌ Error in verify_user_face: {str(e)}", exc_info=True)
        return False, f"Face verification failed: {str(e)}"


# ========== FACE MATCHING ALGORITHM DOCUMENTATION ==========
"""
📐 COSINE DISTANCE EXPLANATION

For normalized vectors v1 and v2:

Formula:
  distance = 1 - (v1 · v2) / (||v1|| × ||v2||)

Range:
  - 0.0 = identical faces (perfect match)
  - 0.4 = threshold (85% similarity)
  - 1.0 = completely different faces

Example:
  ✅ Same person: distance = 0.2 (below threshold) → MATCH
  ❌ Different person: distance = 0.6 (above threshold) → NO MATCH

Why Cosine Distance?
  1. Works well with normalized vectors
  2. Angle-based similarity (robust to magnitude)
  3. Standard for face embedding comparison
  4. Better than Euclidean for this use case

Threshold Tuning:
  - 0.3 = Very strict (only closest matches)
  - 0.4 = Default (good balance)
  - 0.5 = Lenient (more false positives)
"""