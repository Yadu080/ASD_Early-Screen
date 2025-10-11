# scripts/train_and_encrypt.py (fixed)
import os, json, joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from cryptography.fernet import Fernet

BASE = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
DATA_DIR = os.path.join(BASE, "data", "processed")
ART_DIR = os.path.join(BASE, "backend", "artifacts")
os.makedirs(ART_DIR, exist_ok=True)

meta_path = os.path.join(DATA_DIR, "meta.json")
if not os.path.exists(meta_path):
    raise SystemExit("meta.json missing. Run scripts/data_prep_uci.py first.")
meta = json.load(open(meta_path))
csv_path = meta["output_file"]
label_col = meta["label_col"]

df = pd.read_csv(csv_path)
if label_col not in df.columns:
    raise SystemExit(f"Label column '{label_col}' not in processed CSV.")

X = df.drop(columns=[label_col])
y = df[label_col].astype(int)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

scaler = StandardScaler().fit(X_train)
Xtr = scaler.transform(X_train)

clf = LogisticRegression(max_iter=1000, random_state=42)
clf.fit(Xtr, y_train)
acc = clf.score(scaler.transform(X_test), y_test)
print(f"✅ Trained model. Test accuracy: {acc:.3f}")

# Save model & scaler SEPARATELY
model_file = os.path.join(ART_DIR, "model_v1.pkl")
scaler_file = os.path.join(ART_DIR, "scaler_v1.pkl")
joblib.dump(clf, model_file)
joblib.dump(scaler, scaler_file)

# Generate or reuse encryption key
key_path = os.path.join(ART_DIR, "fernet.key")
if os.path.exists(key_path):
    key = open(key_path, "rb").read()
    print("🔑 Using existing key:", key_path)
else:
    key = Fernet.generate_key()
    with open(key_path, "wb") as kf:
        kf.write(key)
    os.chmod(key_path, 0o600)
    print("🔐 Generated key:", key_path)

# Encrypt artifacts
f = Fernet(key)
def encrypt_file(in_path):
    with open(in_path, "rb") as inf:
        data = inf.read()
    enc = f.encrypt(data)
    out_path = in_path + ".enc"
    with open(out_path, "wb") as outf:
        outf.write(enc)
    os.chmod(out_path, 0o600)
    print("Encrypted:", out_path)

encrypt_file(model_file)
encrypt_file(scaler_file)

# Cleanup unencrypted files
os.remove(model_file)
os.remove(scaler_file)

print("\n✅ Done! Encrypted artifacts in:", ART_DIR)
