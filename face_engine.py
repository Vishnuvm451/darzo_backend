import cv2
import numpy as np
import onnxruntime as ort
import os
import logging
from typing import Tuple, Optional

logger = logging.getLogger(__name__)

# =====================================================
# PATH RESOLUTION (ABSOLUTE, SAFE)
# =====================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")

MODEL_PATH = os.path.join(MODELS_DIR, "mobilefacenet.onnx")
EYE_MODEL_PATH = os.path.join(MODELS_DIR, "haarcascade_eye.xml")

# =====================================================
# GLOBAL SINGLETONS
# =====================================================
_session: Optional[ort.InferenceSession] = None
_input_name: Optional[str] = None
_eye_cascade: Optional[cv2.CascadeClassifier] = None
_model_load_error: Optional[str] = None

# =====================================================
# LOAD MODELS (ONCE, WITH SAFETY)
# =====================================================
def _load_models() -> None:
    """
    Load ONNX and cascade models once at startup.
    Sets global variables safely.
    """
    global _session, _input_name, _eye_cascade, _model_load_error

    try:
        # ---------- ONNX MODEL VALIDATION ----------
        if not os.path.exists(MODEL_PATH):
            error_msg = f"ONNX model not found: {MODEL_PATH}"
            logger.critical(f"❌ {error_msg}")
            _model_load_error = error_msg
            raise FileNotFoundError(error_msg)

        if not os.path.isfile(MODEL_PATH):
            error_msg = f"ONNX path is not a file: {MODEL_PATH}"
            logger.critical(f"❌ {error_msg}")
            _model_load_error = error_msg
            raise ValueError(error_msg)

        file_size = os.path.getsize(MODEL_PATH)
        if file_size < 1024 * 1024:  # Less than 1MB is suspicious
            error_msg = f"ONNX file too small ({file_size} bytes): {MODEL_PATH}"
            logger.critical(f"❌ {error_msg}")
            _model_load_error = error_msg
            raise ValueError(error_msg)

        logger.info(f"🔍 Loading ONNX model ({file_size / 1024 / 1024:.1f}MB): {MODEL_PATH}")

        # Load ONNX session
        try:
            _session = ort.InferenceSession(
                MODEL_PATH,
                providers=["CPUExecutionProvider"]
            )
        except Exception as onnx_e:
            error_msg = f"Failed to load ONNX model: {str(onnx_e)}"
            logger.critical(f"❌ {error_msg}")
            _model_load_error = error_msg
            raise RuntimeError(error_msg) from onnx_e

        # ✅ FIX: Validate ONNX inputs
        inputs = _session.get_inputs()
        if not inputs or len(inputs) == 0:
            error_msg = "ONNX model has no inputs"
            logger.critical(f"❌ {error_msg}")
            _model_load_error = error_msg
            raise ValueError(error_msg)

        _input_name = inputs[0].name

        # ✅ FIX: Validate ONNX outputs
        outputs = _session.get_outputs()
        if not outputs or len(outputs) == 0:
            error_msg = "ONNX model has no outputs"
            logger.critical(f"❌ {error_msg}")
            _model_load_error = error_msg
            raise ValueError(error_msg)

        logger.info(f"✅ ONNX model loaded: input={_input_name}, output_shape={outputs[0].shape}")

        # ---------- EYE CASCADE (OPTIONAL) ----------
        _eye_cascade = None
        if os.path.exists(EYE_MODEL_PATH):
            try:
                cascade = cv2.CascadeClassifier(EYE_MODEL_PATH)
                if cascade.empty():
                    logger.warning("⚠️ Eye cascade found but empty. Alignment disabled.")
                else:
                    _eye_cascade = cascade
                    logger.info("👁️ Eye cascade loaded (alignment enabled)")
            except Exception as cascade_e:
                logger.warning(f"⚠️ Eye cascade load failed: {cascade_e}. Alignment disabled.")
        else:
            logger.warning(f"⚠️ Eye cascade not found at: {EYE_MODEL_PATH}")

    except Exception as e:
        logger.critical(f"❌ Model loading critical error: {e}", exc_info=True)
        _model_load_error = str(e)
        raise


# =====================================================
# PUBLIC: GET SESSION (SAFE SINGLETON)
# =====================================================
def get_session() -> Tuple[ort.InferenceSession, str]:
    """
    Get ONNX session safely. Loads once on first call.
    
    Raises:
        RuntimeError: If model loading failed
        
    Returns:
        Tuple of (session, input_name)
    """
    global _session, _input_name, _model_load_error

    if _model_load_error:
        raise RuntimeError(f"Model loading error: {_model_load_error}")

    if _session is None or _input_name is None:
        _load_models()

    if _session is None or _input_name is None:
        raise RuntimeError("Failed to initialize ONNX session")

    return _session, _input_name


# =====================================================
# FACE ALIGNMENT (OPTIONAL, SAFE)
# =====================================================
def align_face(img: np.ndarray) -> np.ndarray:
    """
    Align face using eye detection. Falls back to original if fails.
    
    Args:
        img: BGR image from OpenCV
        
    Returns:
        Aligned image or original if alignment fails
    """
    if _eye_cascade is None:
        return img

    if img is None or img.size == 0:
        logger.warning("⚠️ Alignment: Image is None or empty")
        return img

    try:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        eyes = _eye_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(15, 15),
            maxSize=(200, 200)
        )

        if len(eyes) < 2:
            logger.debug("Alignment: Less than 2 eyes detected")
            return img

        # Pick two largest eyes
        eyes = sorted(eyes, key=lambda e: e[2], reverse=True)[:2]
        (x1, y1, w1, h1), (x2, y2, w2, h2) = eyes

        # Eye centers
        c1 = (x1 + w1 // 2, y1 + h1 // 2)
        c2 = (x2 + w2 // 2, y2 + h2 // 2)

        # Ensure left eye is c1, right eye is c2
        if c1[0] > c2[0]:
            c1, c2 = c2, c1

        # Calculate rotation angle
        angle = np.degrees(np.arctan2(c2[1] - c1[1], c2[0] - c1[0]))

        # ✅ FIX: Limit rotation to avoid distortion
        if abs(angle) > 30:
            logger.debug(f"Alignment: Angle too large ({angle:.1f}°), skipping")
            return img

        center = ((c1[0] + c2[0]) // 2, (c1[1] + c2[1]) // 2)

        # Rotation matrix
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        aligned = cv2.warpAffine(
            img,
            M,
            (img.shape[1], img.shape[0]),
            borderMode=cv2.BORDER_REPLICATE
        )

        logger.debug(f"Alignment: Rotated {angle:.1f}°")
        return aligned

    except Exception as e:
        logger.debug(f"Alignment skipped: {type(e).__name__}: {e}")
        return img


# =====================================================
# PREPROCESS (STRICT, MODEL-SAFE)
# =====================================================
def preprocess(img: np.ndarray) -> np.ndarray:
    """
    Preprocess image for MobileFaceNet (112x112 RGB normalized).
    
    Args:
        img: BGR image from OpenCV
        
    Returns:
        Preprocessed image batch (1, 3, 112, 112) in float32
        
    Raises:
        ValueError: If image is invalid
    """
    if img is None:
        raise ValueError("Image is None")

    if not isinstance(img, np.ndarray):
        raise ValueError(f"Expected numpy.ndarray, got {type(img)}")

    if img.size == 0:
        raise ValueError("Image is empty (size=0)")

    if img.dtype != np.uint8:
        logger.warning(f"Image dtype {img.dtype} is not uint8, converting")
        img = cv2.convertScaleAbs(img)

    try:
        # Resize to model input size
        if img.shape[0] != 112 or img.shape[1] != 112:
            img = cv2.resize(img, (112, 112), interpolation=cv2.INTER_LINEAR)

        # Convert BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Convert to float32
        img = img.astype(np.float32)

        # Normalize: (pixel - 127.5) / 128
        img = (img - 127.5) / 128.0

        # ✅ FIX: Validate normalization range
        if np.isnan(img).any() or np.isinf(img).any():
            raise ValueError("Normalization produced NaN or Inf values")

        if img.min() < -2.0 or img.max() > 2.0:
            logger.warning(f"Normalized image out of expected range: [{img.min():.2f}, {img.max():.2f}]")

        # Transpose to CHW format: (3, 112, 112)
        img = np.transpose(img, (2, 0, 1))

        # Add batch dimension: (1, 3, 112, 112)
        img = np.expand_dims(img, axis=0)

        return img

    except Exception as e:
        logger.error(f"Preprocessing error: {type(e).__name__}: {e}")
        raise


# =====================================================
# PUBLIC: GET FACE EMBEDDING
# =====================================================
def get_embedding(img: np.ndarray) -> np.ndarray:
    """
    Extract normalized face embedding from image.
    
    Args:
        img: BGR image from OpenCV
        
    Returns:
        Normalized embedding vector (128 dimensions)
        
    Raises:
        ValueError: If image is invalid or no embedding generated
        RuntimeError: If model inference fails
    """
    if img is None:
        raise ValueError("Image is None")

    if not isinstance(img, np.ndarray):
        raise ValueError(f"Expected numpy.ndarray, got {type(img)}")

    if img.size == 0:
        raise ValueError("Image is empty")

    try:
        # Get session
        session, input_name = get_session()

        # Align face (optional)
        aligned_img = align_face(img)

        # Preprocess
        inp = preprocess(aligned_img)

        # ✅ FIX: Validate input shape
        if inp.shape != (1, 3, 112, 112):
            raise ValueError(f"Invalid input shape after preprocessing: {inp.shape}")

        # Inference
        try:
            outputs = session.run(None, {input_name: inp})
        except Exception as inference_e:
            logger.error(f"ONNX inference failed: {inference_e}")
            raise RuntimeError(f"Model inference failed: {str(inference_e)}") from inference_e

        # ✅ FIX: Validate outputs
        if not outputs or len(outputs) == 0:
            raise ValueError("Model produced no outputs")

        emb = outputs[0]

        # ✅ FIX: Validate output shape
        if emb.ndim == 2:  # (1, 128)
            emb = emb[0]
        elif emb.ndim != 1:
            raise ValueError(f"Invalid embedding shape: {emb.shape}")

        # ✅ FIX: Validate embedding dimension
        if len(emb) != 128:
            logger.warning(f"Embedding dimension {len(emb)} != 128")
            # Still try to use it if it's a vector
            if len(emb) == 0:
                raise ValueError("Embedding is empty")

        emb = np.array(emb, dtype=np.float32)

        # ✅ FIX: Check for invalid values
        if np.isnan(emb).any() or np.isinf(emb).any():
            raise ValueError("Embedding contains NaN or Inf values")

        # Normalize
        norm = np.linalg.norm(emb)
        if norm == 0:
            raise ValueError("Embedding has zero norm (cannot normalize)")

        if np.isnan(norm) or np.isinf(norm):
            raise ValueError(f"Embedding norm is invalid: {norm}")

        normalized_emb = emb / norm

        # ✅ FIX: Verify normalization
        final_norm = np.linalg.norm(normalized_emb)
        if abs(final_norm - 1.0) > 0.01:  # Should be ~1.0
            logger.warning(f"Normalized embedding norm is {final_norm:.6f}, expected 1.0")

        return normalized_emb

    except (ValueError, RuntimeError):
        raise
    except Exception as e:
        logger.error(f"Embedding extraction error: {type(e).__name__}: {e}", exc_info=True)
        raise RuntimeError(f"Failed to extract embedding: {str(e)}") from e