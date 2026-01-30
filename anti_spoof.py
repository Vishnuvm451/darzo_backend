import cv2
import numpy as np
import logging
from typing import List

logger = logging.getLogger(__name__)

# =====================================================
# BLUR DETECTION (PHOTO / SCREENSHOT BLOCK)
# =====================================================
def is_blurry(img: np.ndarray, threshold: float = 40.0) -> bool:
    """
    Detect if image is blurry using Laplacian variance.
    
    Args:
        img: BGR image from OpenCV
        threshold: Blur threshold (lower = blurrier). Default 40.0 based on testing.
                  Typical: < 20 = very blurry, 20-50 = blurry, 50+ = clear
        
    Returns:
        True if blurry, False if clear
        
    Raises:
        ValueError: If image is invalid
    """
    if img is None:
        raise ValueError("Image is None")

    if not isinstance(img, np.ndarray):
        raise ValueError(f"Expected numpy.ndarray, got {type(img)}")

    if img.size == 0:
        raise ValueError("Image is empty")

    if img.shape[0] < 10 or img.shape[1] < 10:
        raise ValueError(f"Image too small: {img.shape}")

    try:
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Compute Laplacian variance
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        variance = laplacian.var()

        # ✅ FIX: Validate variance
        if np.isnan(variance) or np.isinf(variance):
            logger.warning(f"Blur variance is invalid: {variance}, treating as blurry")
            return True

        is_blur = variance < threshold

        logger.debug(f"Blur score: {variance:.2f} (threshold: {threshold}, blurry: {is_blur})")

        return is_blur

    except Exception as e:
        logger.error(f"Blur detection error: {type(e).__name__}: {e}")
        raise ValueError(f"Blur detection failed: {str(e)}") from e


# =====================================================
# POSE VARIANCE CHECK (VIDEO / STATIC IMAGE BLOCK)
# =====================================================
def pose_variance_ok(embs: List[np.ndarray], variance_threshold: float = 0.98) -> bool:
    """
    Check if embeddings have sufficient pose variance (not the same face in same pose).
    
    This prevents video playback, photo spoofing, and ensures head movement between images.
    
    Args:
        embs: List of embeddings (should be 3 normalized vectors, 128-dim each)
        variance_threshold: Similarity threshold (lower = more variation required)
                           Default 0.98: if similarity > 0.98, same pose (fails)
                                          if similarity < 0.98, different pose (passes)
        
    Returns:
        True if pose variance OK (different poses), False if spoof suspected (same pose)
        
    Raises:
        ValueError: If embeddings are invalid
    """
    if not embs:
        raise ValueError("Embeddings list is empty")

    if not isinstance(embs, (list, tuple)):
        raise ValueError(f"Expected list/tuple of embeddings, got {type(embs)}")

    # ✅ FIX: Validate embedding count
    if len(embs) != 3:
        logger.warning(f"Expected 3 embeddings, got {len(embs)}")
        if len(embs) < 2:
            raise ValueError("Need at least 2 embeddings to check variance")

    try:
        # ✅ FIX: Validate each embedding
        validated_embs = []
        for i, emb in enumerate(embs):
            if emb is None:
                raise ValueError(f"Embedding {i} is None")

            if not isinstance(emb, np.ndarray):
                logger.warning(f"Embedding {i} is {type(emb)}, converting to ndarray")
                try:
                    emb = np.array(emb)
                except Exception as e:
                    raise ValueError(f"Cannot convert embedding {i} to ndarray: {e}")

            if emb.size == 0:
                raise ValueError(f"Embedding {i} is empty")

            if len(emb.shape) > 1:
                raise ValueError(f"Embedding {i} has shape {emb.shape}, expected 1D")

            if len(emb) != 128:
                logger.warning(f"Embedding {i} has {len(emb)} dimensions, expected 128")
                # Still proceed if it's a vector
                if len(emb) < 10:
                    raise ValueError(f"Embedding {i} too small ({len(emb)} dims)")

            # ✅ FIX: Check for invalid values
            if np.isnan(emb).any() or np.isinf(emb).any():
                raise ValueError(f"Embedding {i} contains NaN or Inf values")

            validated_embs.append(np.array(emb, dtype=np.float32))

        # ✅ FIX: Normalize embeddings if not already normalized
        normalized_embs = []
        for i, emb in enumerate(validated_embs):
            norm = np.linalg.norm(emb)
            if norm == 0:
                raise ValueError(f"Embedding {i} has zero norm")
            if norm != 1.0:
                logger.debug(f"Embedding {i} norm is {norm:.6f}, normalizing")
                emb = emb / norm
            normalized_embs.append(emb)

        # Compute pairwise cosine similarities
        similarities = []
        
        for i in range(len(normalized_embs)):
            for j in range(i + 1, len(normalized_embs)):
                # Cosine similarity = dot product of normalized vectors
                sim = float(np.dot(normalized_embs[i], normalized_embs[j]))

                # ✅ FIX: Validate similarity value
                if np.isnan(sim) or np.isinf(sim):
                    logger.warning(f"Similarity[{i},{j}] is invalid: {sim}")
                    sim = 1.0  # Treat as same pose

                # Clamp to [-1, 1] range
                sim = max(-1.0, min(1.0, sim))

                similarities.append(sim)

                logger.debug(f"Similarity[{i},{j}]: {sim:.4f}")

        if not similarities:
            raise ValueError("No similarities computed")

        # Compute mean similarity
        mean_similarity = np.mean(similarities)

        # ✅ FIX: Validate mean similarity
        if np.isnan(mean_similarity) or np.isinf(mean_similarity):
            logger.error(f"Mean similarity is invalid: {mean_similarity}")
            return False

        logger.info(
            f"Pose variance: similarities={[f'{s:.4f}' for s in similarities]}, "
            f"mean={mean_similarity:.4f}, threshold={variance_threshold}, "
            f"variance_ok={mean_similarity < variance_threshold}"
        )

        # ✅ FIX: More nuanced spoof detection
        # Check if ALL similarities are too high (likely spoofing)
        all_high = all(sim > variance_threshold for sim in similarities)
        if all_high:
            logger.warning("All similarities above threshold - spoof suspected")
            return False

        # Check if ANY similarity is extremely high (> 0.99 = identical)
        any_identical = any(sim > 0.995 for sim in similarities)
        if any_identical:
            logger.warning("Found nearly identical embeddings - spoof suspected")
            return False

        # At least one pair should have good variation
        has_variance = any(sim < variance_threshold for sim in similarities)
        if not has_variance:
            logger.warning("No sufficient pose variance detected")
            return False

        return True

    except ValueError:
        raise
    except Exception as e:
        logger.error(f"Pose variance check error: {type(e).__name__}: {e}", exc_info=True)
        raise ValueError(f"Pose variance check failed: {str(e)}") from e


# =====================================================
# HELPER: Batch spoof check with detailed feedback
# =====================================================
def check_spoof_detailed(embs: List[np.ndarray]) -> dict:
    """
    Detailed spoof check with explanations for debugging.
    
    Args:
        embs: List of embeddings
        
    Returns:
        Dict with 'status' (ok/spoof_suspected), 'message', 'similarities'
    """
    try:
        if not isinstance(embs, (list, tuple)) or len(embs) == 0:
            return {
                "status": "spoof_suspected",
                "message": "Invalid embeddings",
                "similarities": []
            }

        # Normalize
        normalized_embs = []
        for emb in embs:
            if isinstance(emb, np.ndarray):
                norm = np.linalg.norm(emb)
                if norm > 0:
                    normalized_embs.append(emb / norm)

        if len(normalized_embs) < 2:
            return {
                "status": "spoof_suspected",
                "message": "Insufficient embeddings",
                "similarities": []
            }

        # Compute similarities
        similarities = []
        for i in range(len(normalized_embs)):
            for j in range(i + 1, len(normalized_embs)):
                sim = float(np.dot(normalized_embs[i], normalized_embs[j]))
                sim = max(-1.0, min(1.0, sim))
                similarities.append(sim)

        mean_sim = np.mean(similarities)
        pose_ok = pose_variance_ok(embs)

        return {
            "status": "ok" if pose_ok else "spoof_suspected",
            "message": f"Mean similarity: {mean_sim:.4f}, variance_ok: {pose_ok}",
            "similarities": similarities,
            "mean_similarity": float(mean_sim)
        }

    except Exception as e:
        logger.error(f"Detailed spoof check error: {e}")
        return {
            "status": "spoof_suspected",
            "message": f"Check failed: {str(e)}",
            "similarities": [],
            "error": str(e)
        }