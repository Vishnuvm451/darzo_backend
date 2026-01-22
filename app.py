from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from firebase import init_firebase
from face_register import router as face_register_router
from face_verify import router as face_verify_router

print("🔥🔥🔥 RUNNING NEW BACKEND CODE v1.2.1 🔥🔥🔥")

# -------------------------------------------------
# LIFESPAN
# -------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    init_firebase()  # 🔥 ONLY ONCE
    print("🔥 Firebase initialized (lifespan)")
    yield
    print("🛑 API shutting down")

# -------------------------------------------------
# APP
# -------------------------------------------------
app = FastAPI(
    title="DARZO Face Recognition API",
    description="Backend for Flutter Smart Attendance System",
    version="1.2.1",
    lifespan=lifespan
)

# -------------------------------------------------
# CORS
# -------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------------------------------
# ROUTERS
# -------------------------------------------------
app.include_router(face_register_router, prefix="/face", tags=["Face Register"])
app.include_router(face_verify_router, prefix="/face", tags=["Face Verify"])

# -------------------------------------------------
# ROOT
# -------------------------------------------------
@app.get("/", tags=["Health"])
def root():
    return {
        "status": "online",
        "service": "DARZO Biometric API",
        "version": "1.2.1"
    }

# -------------------------------------------------
# HEALTH
# -------------------------------------------------
@app.get("/health", tags=["Health"])
def health():
    return {"status": "ok"}
