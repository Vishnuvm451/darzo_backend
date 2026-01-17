import firebase_admin
from firebase_admin import credentials, firestore
# ❌ REMOVED: from firebase_admin import storage

# -------------------------------------------------
# GLOBAL OBJECTS
# -------------------------------------------------
db = None
# ❌ REMOVED: bucket = None

# -------------------------------------------------
# INIT FIREBASE (RUN ONCE)
# -------------------------------------------------
def init_firebase():
    global db

    if firebase_admin._apps:
        return  # Already initialized

    # 🔐 Service account key (same folder as app.py)
    cred = credentials.Certificate("serviceAccountKey.json")

    # ✅ Initialize ONLY with Credential (No Storage Bucket)
    firebase_admin.initialize_app(cred)

    db = firestore.client()
    
    # ❌ REMOVED: bucket = storage.bucket()

    print("✅ Firebase initialized successfully (Firestore Only)")


# -------------------------------------------------
# SAFE ACCESSORS
# -------------------------------------------------
def get_db():
    if db is None:
        raise RuntimeError("❌ Firestore not initialized. Call init_firebase() first.")
    return db

# ❌ REMOVED: get_bucket() function