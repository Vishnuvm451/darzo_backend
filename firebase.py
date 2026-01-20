import firebase_admin
from firebase_admin import credentials, firestore

# -------------------------------------------------
# GLOBAL DB OBJECT
# -------------------------------------------------
db = None

# -------------------------------------------------
# INIT FIREBASE (RUN ONCE)
# -------------------------------------------------
def init_firebase():
    global db

    # Prevent re-initialization error
    if firebase_admin._apps:
        if db is None:
            db = firestore.client()
        return

    try:
        # Service account key must be in the same folder as app.py
        cred = credentials.Certificate("serviceAccountKey.json")

        # Initialize Firebase with the service account
        firebase_admin.initialize_app(cred)

        # Initialize Firestore Client
        db = firestore.client()
        
        print("✅ Firebase initialized successfully (Firestore Client Active)")
    except Exception as e:
        print(f"❌ Critical Error during Firebase Init: {e}")
        raise e

# -------------------------------------------------
# SAFE ACCESSOR
# -------------------------------------------------
def get_db():
    global db
    if db is None:
        # Auto-initialize if someone forgot to call init_firebase()
        init_firebase()
    return db
