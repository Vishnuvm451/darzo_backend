from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from firebase import init_firebase
from face_register import router as face_register_router
from face_verify import router as face_verify_router

# -------------------------------------------------
# LIFESPAN (STARTUP / SHUTDOWN)
# -------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Initialize Firebase once
    init_firebase()
    yield
    # Shutdown logic (optional)

# -------------------------------------------------
# APP INIT
# -------------------------------------------------
app = FastAPI(
    title="DARZO Face Recognition API",
    description="Backend for Flutter Smart Attendance System",
    version="1.1.0",
    lifespan=lifespan,
)

# -------------------------------------------------
# CORS (ALLOW FLUTTER APP)
# -------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------------------------------
# ROUTES
# -------------------------------------------------
# Creates:
# POST /face/register
# POST /face/verify
app.include_router(face_register_router, prefix="/face")
app.include_router(face_verify_router, prefix="/face")

# -------------------------------------------------
# ROOT
# -------------------------------------------------
@app.get("/", tags=["Health"])
def root():
    return {
        "status": "online",
        "service": "DARZO Biometric API",
        "version": "1.1.0",
    }

# -------------------------------------------------
# HEALTH CHECK (IMPORTANT FOR RENDER + FLUTTER)
# -------------------------------------------------
@app.get("/health", tags=["Health"])
def health():
    return {
        "status": "ok",
        "service": "DARZO Biometric API",
    }
