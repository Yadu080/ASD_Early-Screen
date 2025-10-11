from backend.core.key_manager import load_key
from backend.core.inference import InferenceService
import os
FERNET_KEY_PATH="backend/artifacts/fernet.key"
MODEL_ENC="backend/artifacts/model_v1.pkl.enc"
SCALER_ENC="backend/artifacts/scaler_v1.pkl.enc"
key = load_key(FERNET_KEY_PATH)
print("✅ Key loaded:", bool(key))
svc = InferenceService(MODEL_ENC, SCALER_ENC, key, model_version="v1")
print("✅ Model initialized, feature count:", len(svc.feature_names))
