from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
import logging

from firebase import init_firebase
from face_register import router as face_register_router
from face_verify import router as face_verify_router

# ========== LOGGING SETUP ==========
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

logger.info("🔥🔥🔥 STARTING DARZO BACKEND v1.2.2 🔥🔥🔥")

# ========== LIFESPAN ==========
@asynccontextmanager
async def lifespan(app: FastAPI):
    # STARTUP
    logger.info("📱 Initializing Firebase...")
    init_firebase()
    logger.info("✅ Firebase initialized successfully")
    yield
    # SHUTDOWN
    logger.info("🛑 API shutting down")

# ========== APP INIT ==========
app = FastAPI(
    title="DARZO Face Recognition API",
    description="Secure biometric authentication & attendance system",
    version="1.2.2",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    lifespan=lifespan
)

# ========== CORS CONFIG ==========
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # ⚠️ Tighten for production: ["https://yourfrontend.com"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ========== ROUTERS ==========
app.include_router(face_register_router, prefix="/face", tags=["Face Registration"])
app.include_router(face_verify_router, prefix="/face", tags=["Face Verification"])

# ========== ROOT ENDPOINT ==========
@app.get("/", tags=["Health"])
def root():
    return {
        "status": "online",
        "service": "DARZO Biometric API",
        "version": "1.2.2",
        "docs": "/docs",
        "database": "Firebase Firestore"
    }

# ========== HEALTH CHECK ==========
@app.get("/health", tags=["Health"])
def health():
    return {
        "status": "ok",
        "message": "API is healthy and ready"
    }

# ========== HTTP EXCEPTION HANDLER ==========
# ✅ FIX: Proper HTTPException handling for validation errors
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    logger.warning(f"⚠️ HTTP {exc.status_code}: {exc.detail}")
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "success": False,
            "error": exc.detail,
            "status_code": exc.status_code
        }
    )

# ========== GLOBAL EXCEPTION HANDLER ==========
# ✅ FIX: Improved global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    logger.error(f"❌ Unhandled exception: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "error": "Internal server error",
            "detail": str(exc) if str(exc) else "Unknown error"
        }
    )

# ========== RUN ==========
# Development:
#   uvicorn app:app --host 0.0.0.0 --port 8000 --reload
#
# Production (Docker):
#   uvicorn app:app --host 0.0.0.0 --port 8000 
#