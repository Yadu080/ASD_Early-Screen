# backend/core/key_manager.py
from cryptography.fernet import Fernet
import os

def gen_key(path: str) -> bytes:
    key = Fernet.generate_key()
    with open(path, "wb") as f:
        f.write(key)
    os.chmod(path, 0o600)
    return key

def load_key(path: str) -> bytes:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Fernet key not found at {path}")
    with open(path, "rb") as f:
        return f.read()
