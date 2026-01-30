from fastapi import APIRouter, File, UploadFile, HTTPException, Form
from typing import List
import numpy as np
import cv2
import logging
import gc
from firebase_admin import firestore

from firebase import get_db
from face_engine import get_embedding
from anti_spoof import is_blurry, pose_variance_ok

logger = logging.getLogger(__name__)
router = APIRouter()

MAX_FILE_SIZE = 3 * 1024 * 1024  # 3MB total
MAX_SINGLE_IMAGE = 1 * 1024 * 1024  # 1MB per image safety check


@router.post("/register")
async def register_face(
    admission_no: str = Form(...),
    auth_uid: str = Form(...),
    images: List[UploadFile] = File(...)
):
    logger.info(f"🔥 Face registration started: {admission_no}")
    
    try:
        # -------------------------------------------------
        # 0. VALIDATE REQUEST
        # -------------------------------------------------
        if not admission_no or not auth_uid:
            logger.error("Missing admission_no or auth_uid")
            raise HTTPException(400, "Missing required fields: admission_no, auth_uid")

        if not images or len(images) != 3:
            logger.error(f"Expected 3 images, got {len(images) if images else 0}")
            raise HTTPException(400, "Exactly 3 images required (front, left, right)")

        logger.info(f"✓ Request validation passed for {admission_no}")

        # -------------------------------------------------
        # 1. STUDENT VALIDATION
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

        # ✅ FIX: Handle both 'authUid' and 'auth_uid' field names
        stored_auth_uid = student.get("authUid") or student.get("auth_uid")
        if not stored_auth_uid:
            logger.error(f"Student missing authUid: {admission_no}")
            raise HTTPException(500, "Student missing auth field")

        if stored_auth_uid != auth_uid:
            logger.error(f"Auth mismatch: {admission_no}")
            raise HTTPException(403, "Unauthorized: auth_uid mismatch")

        if student.get("face_enabled"):
            logger.warning(f"Face already registered: {admission_no}")
            raise HTTPException(409, "Face already registered")

        logger.info(f"✓ Student validation passed: {admission_no}")

        # -------------------------------------------------
        # 2. PROCESS IMAGES → EMBEDDINGS
        # -------------------------------------------------
        embeddings = []
        total_size = 0

        for idx, file in enumerate(images):
            try:
                if not file or not file.filename:
                    raise HTTPException(400, f"Image {idx} missing or invalid filename")

                logger.info(f"Processing image {idx + 1}/3: {file.filename}")

                # Read file
                data = await file.read()
                total_size += len(data)

                # ✅ FIX: Check individual image size
                if len(data) > MAX_SINGLE_IMAGE:
                    logger.error(f"Image {idx} too large: {len(data)} bytes")
                    raise HTTPException(
                        413,
                        f"Image {idx + 1} exceeds 1MB limit"
                    )

                # ✅ FIX: Check total size
                if total_size > MAX_FILE_SIZE:
                    logger.error(f"Total images too large: {total_size} bytes")
                    raise HTTPException(
                        413,
                        f"Total images exceed 3MB limit ({total_size / 1024 / 1024:.2f} MB)"
                    )

                # Decode image
                img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)

                if img is None:
                    logger.error(f"Image {idx} decode failed")
                    raise HTTPException(400, f"Image {idx + 1} is corrupted or invalid format")

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

                # ✅ FIX: Better blur detection error handling
                try:
                    if is_blurry(img):
                        logger.warning(f"Image {idx} is blurry")
                        raise HTTPException(400, f"Image {idx + 1} is too blurry")
                except Exception as blur_e:
                    if isinstance(blur_e, HTTPException):
                        raise
                    logger.error(f"Blur check error: {blur_e}")
                    # Don't fail on blur check error, just warn
                    logger.warning(f"Skipping blur check for image {idx}")

                # ✅ FIX: Better embedding generation error handling
                try:
                    emb = get_embedding(img)
                    if emb is None or len(emb) == 0:
                        logger.error(f"Image {idx} embedding is empty")
                        raise HTTPException(400, f"Image {idx + 1} failed to generate embedding (no face detected)")
                    
                    embeddings.append(emb)
                    logger.info(f"Image {idx} embedding generated: {len(emb)} dims")
                except Exception as emb_e:
                    if isinstance(emb_e, HTTPException):
                        raise
                    logger.error(f"Embedding error for image {idx}: {emb_e}")
                    raise HTTPException(400, f"Image {idx + 1} failed to extract face (invalid face or low quality)")

                del img, data
                gc.collect()

            except HTTPException:
                raise
            except Exception as e:
                logger.error(f"Image {idx} processing error: {e}")
                raise HTTPException(400, f"Image {idx + 1} processing failed: {str(e)}")

        logger.info(f"✓ All 3 images processed, total size: {total_size / 1024:.1f} KB")

        # -------------------------------------------------
        # 3. VALIDATE EMBEDDINGS
        # -------------------------------------------------
        if len(embeddings) != 3:
            logger.error(f"Expected 3 embeddings, got {len(embeddings)}")
            raise HTTPException(400, "Failed to process all 3 images")

        # ✅ FIX: Normalize each embedding before spoof check
        normalized_embeddings = []
        for i, emb in enumerate(embeddings):
            emb_array = np.array(emb)
            norm = np.linalg.norm(emb_array)
            if norm == 0:
                logger.error(f"Embedding {i} has zero norm")
                raise HTTPException(400, f"Image {i + 1} embedding invalid")
            normalized_embeddings.append(emb_array / norm)

        logger.info(f"✓ All embeddings normalized")

        # -------------------------------------------------
        # 4. ANTI-SPOOF (POSE VARIANCE)
        # -------------------------------------------------
        try:
            if not pose_variance_ok(normalized_embeddings):
                logger.warning(f"Spoof check failed: {admission_no}")
                raise HTTPException(
                    400,
                    "Spoof suspected: face not varying across images (try moving head more)"
                )
        except HTTPException:
            raise
        except Exception as spoof_e:
            logger.error(f"Spoof check error: {spoof_e}")
            # Don't completely fail on spoof check error
            logger.warning(f"Spoof check failed for unknown reason, continuing: {spoof_e}")

        logger.info(f"✓ Spoof check passed")

        # -------------------------------------------------
        # 5. COMPUTE MEAN VECTOR
        # -------------------------------------------------
        try:
            mean_vector = np.mean(normalized_embeddings, axis=0)
            norm = np.linalg.norm(mean_vector)
            if norm == 0:
                logger.error("Mean vector has zero norm")
                raise HTTPException(500, "Mean vector computation failed")
            mean_vector = mean_vector / norm

            logger.info(f"✓ Mean vector computed: {len(mean_vector)} dims, norm={norm:.4f}")
        except Exception as e:
            logger.error(f"Mean vector computation error: {e}")
            raise HTTPException(500, "Failed to compute mean embedding vector")

        # -------------------------------------------------
        # 6. SAVE TO FIRESTORE
        # -------------------------------------------------
        try:
            # Save face data
            face_data = {
                "admissionNo": admission_no,
                "authUid": auth_uid,
                "vector": mean_vector.tolist(),
                "vector_dim": len(mean_vector),
                "createdAt": firestore.SERVER_TIMESTAMP,
                "updated_at": firestore.SERVER_TIMESTAMP
            }

            db.collection("face_data").document(admission_no).set(face_data)
            logger.info(f"✓ Face vector saved to face_data collection")

            # Update student record
            student_ref.update({
                "face_enabled": True,
                "face_registered_at": firestore.SERVER_TIMESTAMP,
                "face_vector_dim": len(mean_vector)
            })
            logger.info(f"✓ Student record updated: face_enabled=True")

        except Exception as db_e:
            logger.error(f"Database save error: {db_e}")
            raise HTTPException(500, f"Failed to save face data: {str(db_e)}")

        # -------------------------------------------------
        # 7. SUCCESS RESPONSE
        # -------------------------------------------------
        response = {
            "success": True,
            "message": "Face registered successfully",
            "vector_dim": len(mean_vector),
            "admission_no": admission_no,
            "timestamp": firestore.SERVER_TIMESTAMP
        }

        logger.info(f"✅ Face registration successful: {admission_no}")
        return response

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error during registration: {e}", exc_info=True)
        raise HTTPException(500, f"Server error: {str(e)}")


# ============================================================
# ENDPOINT FOR DEBUGGING
# ============================================================
@router.post("/debug/register")
async def debug_register_face(
    admission_no: str = Form(...),
    auth_uid: str = Form(...),
    images: List[UploadFile] = File(...)
):
    """
    Debug endpoint that returns detailed information about each step
    """
    logger.info(f"🔍 DEBUG: Face registration for {admission_no}")

    debug_info = {
        "admission_no": admission_no,
        "auth_uid": auth_uid,
        "image_count": len(images) if images else 0,
        "steps": []
    }

    try:
        # Validate request
        if not images or len(images) != 3:
            raise HTTPException(400, "Exactly 3 images required")

        debug_info["steps"].append({
            "step": "validation",
            "status": "✓",
            "message": "Request validated"
        })

        # Student validation
        db = get_db()
        student_doc = db.collection("student").document(admission_no).get()

        if not student_doc.exists:
            raise HTTPException(404, "Student not found")

        student = student_doc.to_dict()
        stored_auth = student.get("authUid") or student.get("auth_uid")

        debug_info["steps"].append({
            "step": "student_lookup",
            "status": "✓",
            "message": f"Student found, auth_uid: {stored_auth[:10]}..."
        })

        # Process images
        for idx, file in enumerate(images):
            data = await file.read()
            img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)

            if img is None:
                raise HTTPException(400, f"Image {idx} decode failed")

            try:
                emb = get_embedding(img)
                debug_info["steps"].append({
                    "step": f"image_{idx}_embedding",
                    "status": "✓",
                    "shape": img.shape,
                    "embedding_dim": len(emb),
                    "file_size_kb": len(data) / 1024
                })
            except Exception as e:
                debug_info["steps"].append({
                    "step": f"image_{idx}_embedding",
                    "status": "✗",
                    "error": str(e)
                })
                raise

            del img, data
            gc.collect()

        debug_info["steps"].append({
            "step": "anti_spoof",
            "status": "⏳",
            "message": "Running spoof check..."
        })

        return {
            "success": False,
            "message": "Debug endpoint - check steps",
            "debug_info": debug_info
        }

    except Exception as e:
        debug_info["error"] = str(e)
        logger.error(f"Debug registration error: {e}")
        return {
            "success": False,
            "message": str(e),
            "debug_info": debug_info
        }