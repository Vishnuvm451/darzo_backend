from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from firebase import init_firebase
from face_register import router as face_register_router
from face_verify import router as face_verify_router

# -------------------------------------------------
# APP INIT
# -------------------------------------------------
app = FastAPI(
    title="DARZO Face Recognition API",
    description="Face registration & verification for attendance",
    version="1.0.0"
)

# -------------------------------------------------
# CORS (ALLOW FLUTTER APP)
# -------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # ⚠️ restrict later
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------------------------------
# INIT FIREBASE
# -------------------------------------------------
init_firebase()

# -------------------------------------------------
# ROUTES
# -------------------------------------------------
# Face Registration → /face/register
# Face Verification → /face/verify
app.include_router(face_register_router, prefix="/face")
app.include_router(face_verify_router, prefix="/face")

# -------------------------------------------------
# HEALTH CHECK
# -------------------------------------------------
@app.get("/")
def root():
    return {
        "status": "ok",
        "message": "DARZO Face Recognition API running"
    }
