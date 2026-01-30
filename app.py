from face_engine import get_session
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
import logging
import os
import sys

from firebase import init_firebase
from face_register import router as face_register_router
from face_verify import router as face_verify_router

# =====================================================
# LOGGING SETUP
# =====================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.StreamHandler(sys.stderr)  # Ensure errors go to stderr
    ]
)
logger = logging.getLogger("DARZO-BACKEND")

logger.info("=" * 80)
logger.info("🔥 STARTING DARZO BACKEND v1.2.3 🔥")
logger.info("=" * 80)

# =====================================================
# STARTUP VALIDATION
# =====================================================
def validate_startup():
    """
    Validate all critical dependencies before starting app.
    Exits with error if any critical dependency fails.
    """
    errors = []
    warnings = []

    # ✅ FIX: Check Python version
    if sys.version_info < (3, 8):
        errors.append(f"Python {sys.version_info.major}.{sys.version_info.minor} < 3.8 required")

    # ✅ FIX: Check environment variables
    required_env = [
        "FIREBASE_CREDENTIALS",
        "FIREBASE_PROJECT_ID"
    ]
    for env_var in required_env:
        if not os.getenv(env_var):
            warnings.append(f"Missing environment variable: {env_var}")

    # ✅ FIX: Check critical file existence
    models_dir = os.path.join(os.path.dirname(__file__), "models")
    if not os.path.exists(models_dir):
        errors.append(f"Models directory missing: {models_dir}")
    else:
        model_file = os.path.join(models_dir, "mobilefacenet.onnx")
        if not os.path.exists(model_file):
            errors.append(f"ONNX model file missing: {model_file}")

    # Log warnings
    if warnings:
        for warning in warnings:
            logger.warning(f"⚠️ {warning}")

    # Exit if critical errors
    if errors:
        for error in errors:
            logger.critical(f"❌ {error}")
        logger.critical("❌ Startup validation failed. Exiting.")
        sys.exit(1)

    logger.info("✅ Startup validation passed")


# =====================================================
# LIFESPAN (STARTUP / SHUTDOWN)
# =====================================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI lifespan context manager.
    Handles startup and shutdown logic.
    """
    startup_errors = []

    try:
        # 🔹 Firebase Initialization
        logger.info("📱 Initializing Firebase...")
        try:
            init_firebase()
            logger.info("✅ Firebase initialized successfully")
        except Exception as firebase_e:
            error_msg = f"Firebase init failed: {firebase_e}"
            logger.error(f"❌ {error_msg}")
            startup_errors.append(error_msg)

        # 🔹 ONNX Model Loading
        logger.info("🧠 Checking ONNX face model...")
        try:
            session, input_name = get_session()
            logger.info(f"✅ ONNX face model ready (input: {input_name})")
        except Exception as onnx_e:
            error_msg = f"ONNX model load failed: {onnx_e}"
            logger.error(f"❌ {error_msg}")
            startup_errors.append(error_msg)

        # 🔹 Check for critical startup errors
        if startup_errors:
            logger.critical("=" * 80)
            logger.critical("❌ STARTUP FAILED - CRITICAL DEPENDENCIES MISSING")
            logger.critical("=" * 80)
            for error in startup_errors:
                logger.critical(f"   • {error}")
            logger.critical("=" * 80)
            # Don't exit here - let the request handlers catch it
            # This allows health checks to work even with missing models

        logger.info("=" * 80)
        logger.info("✅ DARZO BACKEND READY")
        logger.info("=" * 80)

    except Exception as startup_e:
        logger.critical(f"❌ Unexpected startup error: {startup_e}", exc_info=True)
        startup_errors.append(str(startup_e))

    # Yield control to the app
    yield

    # Shutdown
    logger.info("=" * 80)
    logger.info("🛑 DARZO BACKEND SHUTTING DOWN")
    logger.info("=" * 80)

    # ✅ FIX: Cleanup resources
    try:
        # Close ONNX session if needed
        logger.info("🧠 Cleaning up ONNX resources...")
        # ONNX sessions are cleaned up automatically
        logger.info("✅ ONNX cleanup complete")
    except Exception as cleanup_e:
        logger.warning(f"⚠️ Cleanup error: {cleanup_e}")

    logger.info("✅ Shutdown complete")


# =====================================================
# APP INITIALIZATION
# =====================================================
try:
    # Validate before creating app
    validate_startup()

    app = FastAPI(
        title="DARZO Face Recognition API",
        description="Secure biometric authentication & smart attendance system",
        version="1.2.3",
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        lifespan=lifespan,
        # ✅ FIX: Add exception handlers to FastAPI
        exception_handlers={}
    )

except Exception as app_init_e:
    logger.critical(f"❌ App initialization failed: {app_init_e}", exc_info=True)
    sys.exit(1)

# =====================================================
# CORS CONFIGURATION
# =====================================================
try:
    CORS_ORIGINS = os.getenv("CORS_ORIGINS", "*").split(",")
    # Clean up origins
    CORS_ORIGINS = [origin.strip() for origin in CORS_ORIGINS if origin.strip()]

    if not CORS_ORIGINS:
        CORS_ORIGINS = ["*"]
        logger.warning("⚠️ Using wildcard CORS (*) - not recommended for production")

    logger.info(f"🔒 CORS origins: {CORS_ORIGINS}")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=CORS_ORIGINS,
        allow_credentials=True,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["*"],
        max_age=600,  # ✅ FIX: Add cache time
    )

except Exception as cors_e:
    logger.error(f"⚠️ CORS setup error: {cors_e}")
    # Continue anyway with default settings


# =====================================================
# ROUTE REGISTRATION
# =====================================================
try:
    logger.info("📋 Registering routers...")
    
    app.include_router(
        face_register_router,
        prefix="/face",
        tags=["Face Registration"]
    )
    logger.info("✅ Face registration router loaded")

    app.include_router(
        face_verify_router,
        prefix="/face",
        tags=["Face Verification"]
    )
    logger.info("✅ Face verification router loaded")

except Exception as router_e:
    logger.error(f"❌ Router registration failed: {router_e}", exc_info=True)
    sys.exit(1)


# =====================================================
# ENDPOINTS
# =====================================================

@app.get("/", tags=["Health"])
def root():
    """
    Root endpoint - service information.
    """
    return {
        "status": "online",
        "service": "DARZO Biometric API",
        "version": "1.2.3",
        "docs": "/docs",
        "database": "Firebase Firestore",
        "timestamp": __import__('datetime').datetime.now().isoformat()
    }


@app.get("/health", tags=["Health"])
def health():
    """
    Health check endpoint.
    Returns basic status without checking dependencies.
    """
    return {
        "status": "ok",
        "message": "DARZO backend running",
        "timestamp": __import__('datetime').datetime.now().isoformat()
    }


@app.get("/health/deep", tags=["Health"])
def deep_health():
    """
    Deep health check - validates critical dependencies.
    """
    health_status = {
        "status": "ok",
        "checks": {}
    }

    # Check ONNX
    try:
        get_session()
        health_status["checks"]["onnx_model"] = "✅ ok"
    except Exception as e:
        health_status["checks"]["onnx_model"] = f"❌ error: {str(e)[:100]}"
        health_status["status"] = "degraded"

    # Check Firebase
    try:
        from firebase import get_db
        db = get_db()
        if db is None:
            health_status["checks"]["firebase"] = "⚠️ not initialized"
            health_status["status"] = "degraded"
        else:
            health_status["checks"]["firebase"] = "✅ ok"
    except Exception as e:
        health_status["checks"]["firebase"] = f"❌ error: {str(e)[:100]}"
        health_status["status"] = "degraded"

    return health_status


# =====================================================
# EXCEPTION HANDLERS
# =====================================================

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """
    Handle HTTP exceptions with proper logging and response format.
    """
    logger.warning(
        f"⚠️ HTTP {exc.status_code} | "
        f"{request.method} {request.url.path} | "
        f"{exc.detail}"
    )
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "success": False,
            "error": exc.detail,
            "status_code": exc.status_code,
            "timestamp": __import__('datetime').datetime.now().isoformat()
        }
    )


@app.exception_handler(ValueError)
async def value_error_handler(request: Request, exc: ValueError):
    """
    Handle ValueError (validation errors) with 400 Bad Request.
    """
    logger.warning(
        f"⚠️ Validation error | "
        f"{request.method} {request.url.path} | "
        f"{str(exc)[:200]}"
    )
    return JSONResponse(
        status_code=400,
        content={
            "success": False,
            "error": f"Validation error: {str(exc)}",
            "status_code": 400,
            "timestamp": __import__('datetime').datetime.now().isoformat()
        }
    )


@app.exception_handler(RuntimeError)
async def runtime_error_handler(request: Request, exc: RuntimeError):
    """
    Handle RuntimeError (mostly from model inference).
    """
    logger.error(
        f"❌ Runtime error | "
        f"{request.method} {request.url.path} | "
        f"{str(exc)[:200]}"
    )
    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "error": f"Server error: {str(exc)}",
            "status_code": 500,
            "timestamp": __import__('datetime').datetime.now().isoformat()
        }
    )


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """
    Handle all uncaught exceptions.
    """
    exc_type = type(exc).__name__
    exc_msg = str(exc)
    
    logger.error(
        f"❌ UNHANDLED {exc_type} | "
        f"{request.method} {request.url.path} | "
        f"{exc_msg[:200]}",
        exc_info=True
    )
    
    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "error": "Internal server error",
            "detail": exc_msg[:200] if exc_msg else "Unknown error",
            "type": exc_type,
            "status_code": 500,
            "timestamp": __import__('datetime').datetime.now().isoformat()
        }
    )


# =====================================================
# STARTUP MESSAGE
# =====================================================
if __name__ == "__main__":
    logger.info("Run with: uvicorn app:app --host 0.0.0.0 --port 8000 --workers 1")