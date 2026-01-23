import firebase_admin
from firebase_admin import credentials, firestore
import os
import logging

logger = logging.getLogger(__name__)

# ========== GLOBAL DB OBJECT ==========
db = None


# ========== INIT FIREBASE ==========
def init_firebase():
    """
    ✅ Initializes Firebase app and Firestore client.
    ✅ Works on Render (env vars) and Local (serviceAccountKey.json)
    """

    global db

    # -------------------------------------------------
    # ALREADY INITIALIZED
    # -------------------------------------------------
    if firebase_admin._apps:
        logger.info("ℹ️ Firebase already initialized")
        db = firestore.client()
        return

    logger.info("🔥 Initializing Firebase Admin SDK...")

    try:
        # -------------------------------------------------
        # RENDER / PRODUCTION (ENV VARS)
        # -------------------------------------------------
        project_id = os.getenv("FIREBASE_PROJECT_ID")
        client_email = os.getenv("FIREBASE_CLIENT_EMAIL")
        private_key = os.getenv("FIREBASE_PRIVATE_KEY")

        if project_id and client_email and private_key:
            logger.info("✅ Using Firebase credentials from environment variables (Render)")

            # 🔥 VERY IMPORTANT: fix escaped newlines
            private_key = private_key.replace("\\n", "\n")

            cred = credentials.Certificate({
                "type": "service_account",
                "project_id": project_id,
                "client_email": client_email,
                "private_key": private_key,
                "token_uri": "https://oauth2.googleapis.com/token"
            })

        # -------------------------------------------------
        # LOCAL DEVELOPMENT (JSON FILE)
        # -------------------------------------------------
        else:
            cred_path = "serviceAccountKey.json"

            if not os.path.exists(cred_path):
                raise RuntimeError(
                    "Firebase credentials not found.\n"
                    "Set environment variables OR provide serviceAccountKey.json"
                )

            logger.info("✅ Using Firebase credentials from local file")
            cred = credentials.Certificate(cred_path)

        # -------------------------------------------------
        # INITIALIZE FIREBASE
        # -------------------------------------------------
        firebase_admin.initialize_app(cred)
        logger.info("✅ Firebase app initialized")

        db = firestore.client()
        logger.info(f"🔥 Firestore project connected: {db._client.project}")


        # -------------------------------------------------
        # VERIFY CONNECTION
        # -------------------------------------------------
        try:
            list(db.collection("_health_check").limit(1).stream())
            logger.info(f"🔥 Firestore connected successfully")
        except Exception as e:
            logger.warning(f"⚠️ Firestore test warning: {e}")

    except Exception as e:
        logger.error("❌ Firebase initialization failed", exc_info=True)
        raise RuntimeError(f"Firebase init failed: {str(e)}")


# ========== GET DB ==========
def get_db():
    global db

    if db is None:
        init_firebase()

    if db is None:
        raise RuntimeError("Firestore DB not initialized")

    return db



# ========== FIREBASE SCHEMA DOCUMENTATION ==========
"""
📋 FIRESTORE COLLECTIONS STRUCTURE

┌─────────────────────────────────────────────────────────┐
│ Collection: student                                     │
├─────────────────────────────────────────────────────────┤
│ Document: {admission_no}                                │
│                                                         │
│ Fields:                                                 │
│   - admissionNo (string): Student admission number      │
│   - authUid (string): Firebase Auth UID                │
│   - name (string): Student full name                    │
│   - email (string): Student email                       │
│   - face_enabled (boolean): Face registered? (true/false)
│   - face_registered_at (timestamp): Registration time  │
│   - last_attendance (timestamp): Last attendance time   │
│   - attendance_count (integer): Total attendance marks  │
│                                                         │
│ Example:                                                │
│ {                                                       │
│   "admissionNo": "ADM001",                             │
│   "authUid": "firebase_uid_xyz",                       │
│   "name": "John Doe",                                  │
│   "email": "john@example.com",                         │
│   "face_enabled": true,                                │
│   "face_registered_at": timestamp,                     │
│   "last_attendance": timestamp,                        │
│   "attendance_count": 15                               │
│ }                                                       │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│ Collection: face_data                                   │
├─────────────────────────────────────────────────────────┤
│ Document: {admission_no}                                │
│                                                         │
│ Fields:                                                 │
│   - admissionNo (string): Student admission number      │
│   - authUid (string): Firebase Auth UID                │
│   - vector (array): 128D normalized face vector         │
│   - updatedAt (timestamp): Last update time             │
│                                                         │
│ Example:                                                │
│ {                                                       │
│   "admissionNo": "ADM001",                             │
│   "authUid": "firebase_uid_xyz",                       │
│   "vector": [0.123, -0.456, 0.789, ...],              │
│   "updatedAt": timestamp                               │
│ }                                                       │
│                                                         │
│ Note: Vector = [128 float values]                       │
│       This is the normalized face embedding             │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│ Collection: attendance                                  │
├─────────────────────────────────────────────────────────┤
│ Document: auto-generated                                │
│                                                         │
│ Fields:                                                 │
│   - admissionNo (string): Student admission number      │
│   - authUid (string): Firebase Auth UID                │
│   - timestamp (timestamp): When attendance was marked   │
│   - status (string): "present" or "absent"             │
│   - verification_method (string): "face" or "manual"   │
│   - vector_distance (float): Face match distance (0-1) │
│                                                         │
│ Example:                                                │
│ {                                                       │
│   "admissionNo": "ADM001",                             │
│   "authUid": "firebase_uid_xyz",                       │
│   "timestamp": timestamp,                              │
│   "status": "present",                                 │
│   "verification_method": "face",                       │
│   "vector_distance": 0.35                              │
│ }                                                       │
└─────────────────────────────────────────────────────────┘
"""