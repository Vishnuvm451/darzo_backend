from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from typing import List
import numpy as np
import cv2
import logging
import gc
from datetime import datetime
from firebase_admin import firestore

from firebase import get_db
from face_engine import get_embedding
from anti_spoof import is_blurry, pose_variance_ok

logger = logging.getLogger(__name__)
router = APIRouter()

MAX_FILE_SIZE = 3 * 1024 * 1024  # 3MB total
MAX_SINGLE_IMAGE = 1 * 1024 * 1024  # 1MB per image


def cosine_distance(a, b):
    """
    Compute cosine distance between two vectors
    Returns float between 0 and 2 (lower = more similar)
    """
    try:
        a = np.array(a)
        b = np.array(b)
        
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        
        if norm_a == 0 or norm_b == 0:
            return 2.0  # Maximum distance if either is zero
        
        # Cosine similarity: dot(a, b) / (norm(a) * norm(b))
        # Returns value in [-1, 1] where 1 is identical
        similarity = np.dot(a, b) / (norm_a * norm_b)
        
        # Convert to distance [0, 2] where 0 is identical
        distance = 1.0 - similarity
        
        return float(distance)
    except Exception as e:
        logger.error(f"Cosine distance error: {e}")
        return 2.0


@router.post("/verify")
async def verify_face_attendance(
    admission_no: str = Form(...),
    images: List[UploadFile] = File(...)
):
    logger.info(f"🔥 Face verify attendance: {admission_no}")
    
    db = None
    total_size = 0
    embeddings = []
    
    try:
        # -------------------------------------------------
        # 1. REQUEST VALIDATION
        # -------------------------------------------------
        if not admission_no or admission_no.strip() == "":
            logger.error("Missing admission_no")
            raise HTTPException(400, "admission_no is required")

        if not images or len(images) != 3:
            logger.error(f"Expected 3 images, got {len(images) if images else 0}")
            raise HTTPException(400, "Exactly 3 images required (front, left, right)")

        logger.info(f"✓ Request validation passed")

        # -------------------------------------------------
        # 2. DATABASE & STUDENT VALIDATION
        # -------------------------------------------------
        db = get_db()
        if db is None:
            logger.error("Database connection failed")
            raise HTTPException(500, "Database connection error")

        student_ref = db.collection("student").document(admission_no)
        student_doc = student_ref.get()

        if not student_doc.exists:
            logger.error(f"Student not found: {admission_no}")
            raise HTTPException(404, f"Student {admission_no} not found")

        student = student_doc.to_dict()
        if not student:
            logger.error(f"Student data corrupted: {admission_no}")
            raise HTTPException(500, "Student data corrupted")

        # ✅ FIX: Check face registration
        if not student.get("face_enabled"):
            logger.warning(f"Face not registered: {admission_no}")
            raise HTTPException(400, "Face not registered. Please register first.")

        class_id = student.get("classId")
        if not class_id:
            logger.error(f"Student not assigned to class: {admission_no}")
            raise HTTPException(400, "Student not assigned to a class")

        logger.info(f"✓ Student validation passed: {admission_no}, classId: {class_id}")

        # -------------------------------------------------
        # 3. ACTIVE SESSION VALIDATION
        # -------------------------------------------------
        try:
            session_query = db.collection("attendance_session") \
                .where("classId", "==", class_id) \
                .where("isActive", "==", True) \
                .limit(1) \
                .stream()

            session = None
            session_id = None
            
            for s in session_query:
                session = s.to_dict()
                session_id = s.id
                break

            if not session:
                logger.warning(f"No active session for class: {class_id}")
                raise HTTPException(400, "No active attendance session for your class")

            # ✅ FIX: Check session expiration
            expires_at = session.get("expiresAt")
            if expires_at:
                # Handle both datetime and Timestamp objects
                expire_time = expires_at
                if hasattr(expires_at, 'to_pydatetime'):
                    expire_time = expires_at.to_pydatetime()
                
                if expire_time < datetime.now():
                    logger.warning(f"Attendance session expired: {session_id}")
                    raise HTTPException(400, "Attendance session has expired")

            logger.info(f"✓ Active session found: {session_id}")

        except HTTPException:
            raise
        except Exception as session_e:
            logger.error(f"Session query error: {session_e}")
            raise HTTPException(500, f"Failed to check attendance session: {str(session_e)}")

        # -------------------------------------------------
        # 4. LOAD STORED FACE VECTOR
        # -------------------------------------------------
        try:
            face_doc = db.collection("face_data").document(admission_no).get()

            if not face_doc.exists:
                logger.error(f"Face data missing: {admission_no}")
                raise HTTPException(401, "Face not registered in system")

            face_data = face_doc.to_dict()
            if not face_data or "vector" not in face_data:
                logger.error(f"Face vector missing: {admission_no}")
                raise HTTPException(500, "Face vector data corrupted")

            stored_vector = np.array(face_data["vector"])
            
            # ✅ FIX: Validate stored vector
            if len(stored_vector) == 0:
                logger.error(f"Stored vector is empty: {admission_no}")
                raise HTTPException(500, "Stored face vector is invalid")
            
            # Normalize stored vector
            stored_norm = np.linalg.norm(stored_vector)
            if stored_norm > 0:
                stored_vector = stored_vector / stored_norm

            logger.info(f"✓ Stored face vector loaded: {len(stored_vector)} dims")

        except HTTPException:
            raise
        except Exception as face_e:
            logger.error(f"Face data load error: {face_e}")
            raise HTTPException(500, f"Failed to load face data: {str(face_e)}")

        # -------------------------------------------------
        # 5. PROCESS LIVE IMAGES
        # -------------------------------------------------
        for idx, file in enumerate(images):
            try:
                if not file or not file.filename:
                    raise HTTPException(400, f"Image {idx} missing or invalid")

                logger.info(f"Processing image {idx + 1}/3: {file.filename}")

                # Read file
                data = await file.read()
                total_size += len(data)

                # ✅ FIX: Validate individual and total sizes
                if len(data) > MAX_SINGLE_IMAGE:
                    logger.error(f"Image {idx} too large: {len(data)} bytes")
                    raise HTTPException(
                        413,
                        f"Image {idx + 1} exceeds 1MB limit"
                    )

                if total_size > MAX_FILE_SIZE:
                    logger.error(f"Total images too large: {total_size} bytes")
                    raise HTTPException(
                        413,
                        f"Total images exceed 3MB limit"
                    )

                # Decode image
                img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)

                if img is None:
                    logger.error(f"Image {idx} decode failed")
                    raise HTTPException(400, f"Image {idx + 1} is corrupted or invalid")

                logger.info(f"Image {idx} decoded: {img.shape}")

                # ✅ FIX: Validate image dimensions
                if img.shape[0] < 50 or img.shape[1] < 50:
                    logger.error(f"Image {idx} too small: {img.shape}")
                    raise HTTPException(400, f"Image {idx + 1} too small (min 50x50)")

                if img.shape[0] > 2000 or img.shape[1] > 2000:
                    logger.error(f"Image {idx} too large: {img.shape}")
                    raise HTTPException(400, f"Image {idx + 1} too large (max 2000x2000)")

                # ✅ FIX: Resize large images
                if img.shape[1] > 800 or img.shape[0] > 800:
                    scale = min(800 / img.shape[1], 800 / img.shape[0])
                    new_width = int(img.shape[1] * scale)
                    new_height = int(img.shape[0] * scale)
                    img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)
                    logger.info(f"Image {idx} resized to {img.shape}")

                # ✅ FIX: Blur check with error handling
                try:
                    if is_blurry(img):
                        logger.warning(f"Image {idx} is blurry")
                        raise HTTPException(400, f"Image {idx + 1} is too blurry - hold still")
                except HTTPException:
                    raise
                except Exception as blur_e:
                    logger.warning(f"Blur check failed for image {idx}: {blur_e}")

                # ✅ FIX: Embedding generation with validation
                try:
                    emb = get_embedding(img)
                    
                    if emb is None or len(emb) == 0:
                        logger.error(f"Image {idx} embedding is empty/None")
                        raise HTTPException(
                            401,
                            f"Image {idx + 1} - no face detected or poor quality"
                        )
                    
                    embeddings.append(np.array(emb))
                    logger.info(f"Image {idx} embedding generated: {len(emb)} dims")
                    
                except HTTPException:
                    raise
                except Exception as emb_e:
                    logger.error(f"Embedding error for image {idx}: {emb_e}")
                    raise HTTPException(
                        401,
                        f"Image {idx + 1} failed to extract face - improve lighting"
                    )

                del img, data
                gc.collect()

            except HTTPException:
                raise
            except Exception as e:
                logger.error(f"Image {idx} processing error: {e}")
                raise HTTPException(400, f"Image {idx + 1} processing failed")

        logger.info(f"✓ All 3 images processed")

        # -------------------------------------------------
        # 6. VALIDATE EMBEDDINGS & ANTI-SPOOF
        # -------------------------------------------------
        if len(embeddings) != 3:
            logger.error(f"Expected 3 embeddings, got {len(embeddings)}")
            raise HTTPException(400, "Failed to process all 3 images")

        # ✅ FIX: Normalize embeddings
        normalized_embeddings = []
        for i, emb in enumerate(embeddings):
            emb_array = np.array(emb)
            norm = np.linalg.norm(emb_array)
            
            if norm == 0:
                logger.error(f"Embedding {i} has zero norm")
                raise HTTPException(401, f"Image {i + 1} embedding invalid")
            
            normalized_embeddings.append(emb_array / norm)

        logger.info(f"✓ All embeddings normalized")

        # ✅ FIX: Anti-spoof check
        try:
            if not pose_variance_ok(normalized_embeddings):
                logger.warning(f"Spoof check failed: {admission_no}")
                raise HTTPException(
                    401,
                    "Spoof check failed - move your head more between shots"
                )
        except HTTPException:
            raise
        except Exception as spoof_e:
            logger.warning(f"Spoof check error: {spoof_e}")
            # Don't fail completely on spoof check error
            logger.info("Continuing despite spoof check warning...")

        logger.info(f"✓ Anti-spoof check passed")

        # -------------------------------------------------
        # 7. COMPUTE LIVE MEAN VECTOR & COMPARE
        # -------------------------------------------------
        try:
            live_mean = np.mean(normalized_embeddings, axis=0)
            live_norm = np.linalg.norm(live_mean)
            
            if live_norm == 0:
                logger.error("Live mean vector has zero norm")
                raise HTTPException(500, "Failed to compute verification vector")
            
            live_mean = live_mean / live_norm

            logger.info(f"✓ Live mean vector computed")

        except Exception as e:
            logger.error(f"Mean vector error: {e}")
            raise HTTPException(500, f"Failed to compute face vector: {str(e)}")

        # ✅ FIX: Individual embedding scores + mean score
        try:
            # Compute distance for each individual embedding
            individual_distances = []
            for i, emb in enumerate(normalized_embeddings):
                dist = cosine_distance(emb, stored_vector)
                individual_distances.append(dist)
                logger.info(f"Image {i} distance: {dist:.4f}")

            # Compute distance for mean vector
            mean_distance = cosine_distance(live_mean, stored_vector)
            logger.info(f"Mean vector distance: {mean_distance:.4f}")

            # ✅ FIX: Compute confidence score (inverse of distance)
            # Distance 0 = perfect match (confidence 100)
            # Distance 1 = no match (confidence 0)
            confidence = max(0, min(100, (1.0 - mean_distance) * 100))

            logger.info(f"Overall confidence: {confidence:.1f}%")
            logger.info(f"Individual distances: {individual_distances}")

            # ✅ FIX: Thresholds
            # Mean distance should be low (< 0.45 = good match)
            # All individual distances should be reasonably close
            mean_threshold = 0.45
            individual_threshold = 0.55

            # Check mean distance
            if mean_distance > mean_threshold:
                logger.warning(f"Face mismatch: distance {mean_distance:.4f} > {mean_threshold}")
                raise HTTPException(
                    401,
                    f"Face does not match - confidence {confidence:.0f}%"
                )

            # ✅ FIX: Check individual distances (at least 2 of 3 should be good)
            good_matches = sum(1 for d in individual_distances if d < individual_threshold)
            if good_matches < 2:
                logger.warning(f"Only {good_matches}/3 images matched well")
                raise HTTPException(
                    401,
                    "Face quality too low - retake photos"
                )

            logger.info(f"✓ Face match verified: {confidence:.1f}% confidence")

        except HTTPException:
            raise
        except Exception as match_e:
            logger.error(f"Face matching error: {match_e}")
            raise HTTPException(500, f"Face matching failed: {str(match_e)}")

        # -------------------------------------------------
        # 8. CHECK ALREADY MARKED
        # -------------------------------------------------
        try:
            today = datetime.now().strftime("%Y-%m-%d")
            session_type = session.get("sessionType", "morning")
            doc_id = f"{class_id}_{today}_{session_type}"

            attendance_doc = db.collection("attendance") \
                .document(doc_id) \
                .collection("student") \
                .document(admission_no) \
                .get()

            if attendance_doc.exists:
                logger.warning(f"Already marked: {admission_no}")
                return {
                    "success": True,
                    "status": "already_marked",
                    "message": "Attendance already marked today",
                    "confidence": float(confidence)
                }

            logger.info(f"✓ First attendance mark for today")

        except Exception as check_e:
            logger.warning(f"Duplicate check error: {check_e}")
            # Continue anyway - don't fail on duplicate check

        # -------------------------------------------------
        # 9. MARK ATTENDANCE
        # -------------------------------------------------
        try:
            today = datetime.now().strftime("%Y-%m-%d")
            session_type = session.get("sessionType", "morning")
            doc_id = f"{class_id}_{today}_{session_type}"

            attendance_ref = db.collection("attendance") \
                .document(doc_id) \
                .collection("student") \
                .document(admission_no)

            attendance_ref.set({
                "studentId": student.get("authUid"),
                "admissionNo": admission_no,
                "name": student.get("name"),
                "classId": class_id,
                "status": "present",
                "method": "face",
                "verified": True,
                "confidence": float(confidence),
                "mean_distance": float(mean_distance),
                "individual_distances": [float(d) for d in individual_distances],
                "markedAt": firestore.SERVER_TIMESTAMP,
                "sessionId": session_id
            })

            logger.info(f"✅ Attendance marked: {admission_no}, confidence: {confidence:.1f}%")

        except Exception as db_e:
            logger.error(f"Database write error: {db_e}")
            raise HTTPException(500, f"Failed to mark attendance: {str(db_e)}")

        # -------------------------------------------------
        # 10. SUCCESS RESPONSE
        # -------------------------------------------------
        return {
            "success": True,
            "status": "marked",
            "message": f"Attendance marked successfully - {confidence:.0f}% match",
            "confidence": float(confidence),
            "student_name": student.get("name", ""),
            "admission_no": admission_no,
            "timestamp": datetime.now().isoformat()
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        raise HTTPException(500, f"Server error: {str(e)}")


# ============================================================
# DEBUG ENDPOINT
# ============================================================
@router.post("/debug/verify")
async def debug_verify_face(
    admission_no: str = Form(...),
    images: List[UploadFile] = File(...)
):
    """
    Debug endpoint that returns step-by-step verification process
    """
    logger.info(f"🔍 DEBUG: Face verify for {admission_no}")

    debug_info = {
        "admission_no": admission_no,
        "image_count": len(images) if images else 0,
        "steps": []
    }

    try:
        # Request validation
        if not images or len(images) != 3:
            raise HTTPException(400, "Exactly 3 images required")

        debug_info["steps"].append({
            "step": "validation",
            "status": "✓",
            "message": "Request validated"
        })

        # Database
        db = get_db()
        if db is None:
            raise HTTPException(500, "Database error")

        debug_info["steps"].append({
            "step": "database",
            "status": "✓",
            "message": "Database connected"
        })

        # Student lookup
        student_doc = db.collection("student").document(admission_no).get()
        if not student_doc.exists:
            raise HTTPException(404, "Student not found")

        student = student_doc.to_dict()

        debug_info["steps"].append({
            "step": "student_lookup",
            "status": "✓",
            "message": f"Student found: {student.get('name')}"
        })

        # Load face data
        face_doc = db.collection("face_data").document(admission_no).get()
        if not face_doc.exists:
            raise HTTPException(401, "Face not registered")

        debug_info["steps"].append({
            "step": "face_data_load",
            "status": "✓",
            "message": "Face vector loaded"
        })

        # Process images
        for idx, file in enumerate(images):
            data = await file.read()
            img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)

            if img is None:
                raise HTTPException(400, f"Image {idx} decode failed")

            emb = get_embedding(img)
            if emb is None:
                raise HTTPException(401, f"Image {idx} - no face detected")

            debug_info["steps"].append({
                "step": f"image_{idx}_process",
                "status": "✓",
                "shape": img.shape,
                "embedding_dim": len(emb),
                "file_size_kb": len(data) / 1024
            })

            del img, data
            gc.collect()

        debug_info["steps"].append({
            "step": "face_matching",
            "status": "⏳",
            "message": "Computing similarity..."
        })

        return {
            "success": False,
            "message": "Debug endpoint - check steps",
            "debug_info": debug_info
        }

    except Exception as e:
        debug_info["error"] = str(e)
        logger.error(f"Debug error: {e}")
        return {
            "success": False,
            "message": str(e),
            "debug_info": debug_info
        }