# backend/core/model_store.py
import tempfile
import os
import joblib
from cryptography.fernet import Fernet

def decrypt_to_temp(enc_path: str, key_bytes: bytes) -> str:
    if not os.path.exists(enc_path):
        raise FileNotFoundError(f"Encrypted artifact not found: {enc_path}")
    with open(enc_path, "rb") as ef:
        data = ef.read()
    f = Fernet(key_bytes)
    dec = f.decrypt(data)
    tmp = tempfile.NamedTemporaryFile(delete=False)
    tmp.write(dec)
    tmp.flush()
    tmp.close()
    os.chmod(tmp.name, 0o600)
    return tmp.name

def load_model_and_scaler(enc_model_path: str, enc_scaler_path: str, key_bytes: bytes):
    mtmp = decrypt_to_temp(enc_model_path, key_bytes)
    stmp = decrypt_to_temp(enc_scaler_path, key_bytes)
    try:
        art = joblib.load(mtmp)
        if isinstance(art, dict) and 'model' in art and 'scaler' in art:
            model = art['model']
            scaler = art['scaler']
        else:
            model = art
            scaler = joblib.load(stmp)
    finally:
        try: os.remove(mtmp)
        except Exception: pass
        try: os.remove(stmp)
        except Exception: pass
    return model, scaler
