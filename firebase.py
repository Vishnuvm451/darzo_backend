import firebase_admin
from firebase_admin import credentials, firestore
import os
import json
import logging

logger = logging.getLogger(__name__)

# ========== GLOBAL DB OBJECT ==========
db = None

# ========== INIT FIREBASE ==========
def init_firebase():
    """
    ✅ Initializes Firebase app and Firestore client.
    Supports both environment variable (Render) and local file (development).
    
    Environment Variable:
    - FIREBASE_CREDENTIALS: JSON string of service account key
    
    Local File (Development):
    - serviceAccountKey.json (in project root)
    
    Raises:
    - RuntimeError: If Firebase initialization fails
    """
    global db

    # ========== CHECK IF ALREADY INITIALIZED ==========
    if firebase_admin._apps:
        logger.info("ℹ️ Firebase already initialized, reusing existing app")
        db = firestore.client()
        return

    logger.info("📱 Starting Firebase initialization...")

    try:
        # ========== TRY ENVIRONMENT VARIABLE (RENDER) ==========
        creds_json = os.getenv("FIREBASE_CREDENTIALS")
        
        if creds_json:
            logger.info("✅ Using Firebase credentials from environment variable (RENDER)")
            try:
                cred_dict = json.loads(creds_json)
                cred = credentials.Certificate(cred_dict)
            except json.JSONDecodeError as e:
                logger.error(f"❌ Invalid JSON in FIREBASE_CREDENTIALS: {str(e)}")
                raise RuntimeError(f"Invalid Firebase credentials JSON: {str(e)}")
        
        # ========== FALLBACK TO LOCAL FILE (DEVELOPMENT) ==========
        else:
            cred_path = "serviceAccountKey.json"
            
            if not os.path.exists(cred_path):
                logger.error(f"❌ Credentials file not found: {cred_path}")
                logger.error("   Set FIREBASE_CREDENTIALS environment variable or place serviceAccountKey.json in project root")
                raise RuntimeError(
                    f"Firebase credentials not found. "
                    f"Either set FIREBASE_CREDENTIALS env variable or place {cred_path} in project root"
                )
            
            logger.info(f"✅ Using Firebase credentials from file: {cred_path}")
            cred = credentials.Certificate(cred_path)

        # ========== INITIALIZE FIREBASE APP ==========
        firebase_admin.initialize_app(cred)
        logger.info("✅ Firebase app initialized")

        # ========== GET FIRESTORE CLIENT ==========
        db = firestore.client()
        logger.info("✅ Firestore client created")

        # ========== TEST CONNECTION ==========
        try:
            # Try to query a collection (non-blocking test)
            list(db.collection("_health_check").limit(1).stream())
            logger.info("✅ Firestore connection verified - API is ready!")
        except Exception as e:
            logger.warning(f"⚠️ Firestore connection test warning: {str(e)}")
            # Don't fail here - Firestore might be accessible even if test fails

    except RuntimeError:
        raise
    
    except Exception as e:
        logger.error(f"❌ Critical error during Firebase initialization: {str(e)}", exc_info=True)
        raise RuntimeError(f"Firebase initialization failed: {str(e)}") from e


# ========== GET DATABASE CLIENT ==========
def get_db():
    """
    ✅ Returns the Firestore client.
    Auto-initializes if not already initialized.
    
    Returns:
    - firestore.client(): Firestore database client
    
    Raises:
    - RuntimeError: If Firebase initialization fails
    """
    global db
    
    if db is None:
        logger.warning("⚠️ Firebase not initialized yet, calling init_firebase()...")
        init_firebase()
    
    if db is None:
        logger.error("❌ Failed to initialize Firebase database")
        raise RuntimeError("Firebase database initialization failed")
    
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