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
    Initializes Firebase Admin SDK and Firestore client.

    Supports:
    - Render / Production (env vars)
    - Local development (serviceAccountKey.json)

    Required ENV VARS (production):
    - FIREBASE_PROJECT_ID
    - FIREBASE_CLIENT_EMAIL
    - FIREBASE_PRIVATE_KEY   (with \\n literals)
    """

    global db

    # -------------------------------------------------
    # ALREADY INITIALIZED
    # -------------------------------------------------
    if firebase_admin._apps:
        logger.info("ℹ️ Firebase already initialized, reusing app")
        db = firestore.client()
        return

    logger.info("🔥 Initializing Firebase Admin SDK...")

    try:
        # -------------------------------------------------
        # PRODUCTION / RENDER (ENV VARS)
        # -------------------------------------------------
        project_id = os.getenv("FIREBASE_PROJECT_ID")
        client_email = os.getenv("FIREBASE_CLIENT_EMAIL")
        private_key = os.getenv("FIREBASE_PRIVATE_KEY")

        if project_id and client_email and private_key:
            logger.info("✅ Using Firebase credentials from environment variables")

            # Convert escaped newlines
            private_key = private_key.replace("\\n", "\n")

            if not private_key.startswith("-----BEGIN"):
                raise ValueError("Invalid private key format")

            cred = credentials.Certificate({
                "type": "service_account",
                "project_id": project_id,
                "private_key": private_key,
                "client_email": client_email,
                "token_uri": "https://oauth2.googleapis.com/token",
            })

        # -------------------------------------------------
        # LOCAL DEVELOPMENT (JSON FILE)
        # -------------------------------------------------
        else:
            cred_path = "serviceAccountKey.json"

            if not os.path.exists(cred_path):
                raise RuntimeError(
                    "Firebase credentials not found.\n"
                    "Set env vars or provide serviceAccountKey.json"
                )

            logger.info(f"✅ Using Firebase credentials from file: {cred_path}")
            cred = credentials.Certificate(cred_path)

        # -------------------------------------------------
        # INITIALIZE FIREBASE
        # -------------------------------------------------
        firebase_admin.initialize_app(cred)
        logger.info("✅ Firebase admin app initialized")

        db = firestore.client()
        logger.info(f"✅ Firestore client created (project: {db.project})")

        # -------------------------------------------------
        # VERIFY CONNECTION (NON-FATAL)
        # -------------------------------------------------
        try:
            list(db.collection("_health_check").limit(1).stream())
            logger.info("✅ Firestore connection verified")
        except Exception as e:
            logger.warning(f"⚠️ Firestore health check warning: {e}")

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