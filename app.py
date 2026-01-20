from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from firebase import init_firebase
from face_register import router as face_register_router
from face_verify import router as face_verify_router

# -------------------------------------------------
# LIFESPAN (STARTUP/SHUTDOWN)
# -------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Initialize Firebase
    init_firebase()
    yield
    # Shutdown logic (if any) goes here

# -------------------------------------------------
# APP INIT
# -------------------------------------------------
app = FastAPI(
    title="DARZO Face Recognition API",
    description="Backend for Flutter Smart Attendance System",
    version="1.1.0",
    lifespan=lifespan
)

# -------------------------------------------------
# CORS (ALLOW FLUTTER APP)
# -------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # For production, replace with your actual Flutter web URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------------------------------
# ROUTES
# -------------------------------------------------
# This creates:
# 1. POST /face/register
# 2. POST /face/verify
app.include_router(face_register_router, prefix="/face")
app.include_router(face_verify_router, prefix="/face")

# -------------------------------------------------
# ROOT & HEALTH CHECK
# -------------------------------------------------
@app.get("/", tags=["Health"])
def root():
    return {
        "status": "online",
        "service": "DARZO Biometric API",
        "version": "1.1.0"
    }
