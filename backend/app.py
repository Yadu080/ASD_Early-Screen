# backend/app.py (top portion — replace existing content from file top through init_services decorator)

import sys, os, time, json

# Make sure project root is on sys.path so "from backend.core..." works
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from flask import Flask, request, jsonify, abort
from backend.core.key_manager import load_key
from backend.core.inference import InferenceService
from backend.core.fairness import load_fairness
from backend.core.audit import log_inference
import pandas as pd

ARTIFACT_DIR = os.environ.get("ARTIFACT_DIR", "backend/artifacts")
FERNET_KEY_PATH = os.environ.get("FERNET_KEY_PATH", os.path.join(ARTIFACT_DIR, "fernet.key"))
MODEL_ENC = os.environ.get("MODEL_ENC", os.path.join(ARTIFACT_DIR, "model_v1.pkl.enc"))
SCALER_ENC = os.environ.get("SCALER_ENC", os.path.join(ARTIFACT_DIR, "scaler_v1.pkl.enc"))
FAIRNESS_JSON = os.environ.get("FAIRNESS_JSON", os.path.join(ARTIFACT_DIR, "fairness_v1.json"))
ADMIN_TOKEN = os.environ.get("ADMIN_TOKEN", "replace_admin_token")
MODEL_VERSION = os.environ.get("MODEL_VERSION", "v1")

app = Flask(__name__)
INFERENCE_SERVICE = None
FAIRNESS_CACHE = None

def admin_required(f):
    def wrapped(*args, **kwargs):
        token = request.headers.get("X-Admin-Token") or request.args.get("admin_token")
        if token != ADMIN_TOKEN:
            abort(401, description="Admin token required")
        return f(*args, **kwargs)
    wrapped.__name__ = f.__name__
    return wrapped

def init_services():
    """
    Initialize inference service and fairness cache.
    Registered on app startup using a method compatible with multiple Flask versions.
    """
    global INFERENCE_SERVICE, FAIRNESS_CACHE
    FAIRNESS_CACHE = load_fairness(FAIRNESS_JSON)
    try:
        key = load_key(FERNET_KEY_PATH)
        INFERENCE_SERVICE = InferenceService(MODEL_ENC, SCALER_ENC, key, model_version=MODEL_VERSION)
        app.logger.info("Inference service initialized")
    except Exception as e:
        app.logger.warning(f"Failed to init inference service: {e}")
        INFERENCE_SERVICE = None

# Register init_services in a way that works with Flask 2.x and 3.x
# Flask 3 removed before_first_request; use register_before_serving if available.
try:
    # Flask >= 2.0: register_before_serving exists in many versions
    app.register_before_serving(init_services)
except Exception:
    # Fallback for older versions
    try:
        app.before_first_request(init_services)
    except Exception:
        # As last resort, call it immediately (useful in dev)
        init_services()


@app.route("/health", methods=["GET"])
def health():
    ok = INFERENCE_SERVICE is not None and INFERENCE_SERVICE.model is not None
    return jsonify({"status": "ok" if ok else "degraded", "model_version": MODEL_VERSION})

@app.route("/fairness", methods=["GET"])
def fairness():
    return jsonify(FAIRNESS_CACHE or {"note": "no fairness data"})

@app.route("/predict", methods=["POST"])
def predict():
    global INFERENCE_SERVICE
    if INFERENCE_SERVICE is None:
        return jsonify({"error": "model not loaded"}), 503

    data = None
    if request.is_json:
        payload = request.get_json(force=True, silent=True)
        if not payload:
            return jsonify({"error": "invalid json payload"}), 400
        data = payload
    elif request.form:
        data = {k: request.form[k] for k in request.form.keys()}
    elif request.files and request.files.get("file"):
        f = request.files["file"]
        try:
            df = pd.read_csv(f)
        except Exception as e:
            return jsonify({"error": f"failed to read csv: {e}"}), 400
        if df.shape[0] < 1:
            return jsonify({"error": "csv contains no rows"}), 400
        if "label" in df.columns:
            df = df.drop(columns=["label"])
        input_df = df.iloc[[0]]
        data = input_df.to_dict(orient="records")[0]
    else:
        return jsonify({"error": "no input provided"}), 400

    try:
        normalized = {}
        for k,v in data.items():
            try: normalized[k] = float(v)
            except Exception: normalized[k] = v
        input_df = pd.DataFrame([normalized])
    except Exception as e:
        return jsonify({"error": f"failed to build dataframe: {e}"}), 400

    if INFERENCE_SERVICE.feature_names:
        missing = set(INFERENCE_SERVICE.feature_names) - set(input_df.columns)
        if missing:
            return jsonify({"error": f"missing features: {sorted(list(missing))}"}), 400
        input_df = input_df[INFERENCE_SERVICE.feature_names]

    t0 = time.time()
    try:
        out = INFERENCE_SERVICE.predict_and_explain(input_df)
    except Exception as e:
        return jsonify({"error": f"inference failed: {e}"}), 500
    latency_ms = int((time.time() - t0) * 1000)

    try:
        log_inference(normalized, out.get("model_version", MODEL_VERSION), latency_ms, out.get("probability"), out.get("prediction"))
    except Exception:
        pass

    response = {"model_version": out.get("model_version"), "probability": out.get("probability"), "prediction": out.get("prediction"), "explanation": out.get("explanation"), "shap_b64": out.get("shap_b64"), "latency_ms": latency_ms}
    return jsonify(response)

@app.route("/admin/upload_model", methods=["POST"])
@admin_required
def admin_upload_model():
    global INFERENCE_SERVICE, MODEL_VERSION, MODEL_ENC, SCALER_ENC
    if "model_file" not in request.files or "scaler_file" not in request.files:
        return jsonify({"error": "model_file and scaler_file required"}), 400
    model_file = request.files["model_file"]
    scaler_file = request.files["scaler_file"]
    version = request.form.get("version")
    if not version:
        return jsonify({"error": "version is required"}), 400
    art_dir = os.path.dirname(MODEL_ENC) or "backend/artifacts"
    tmp_model = os.path.join(art_dir, f"model_{version}.pkl.enc.tmp")
    tmp_scaler = os.path.join(art_dir, f"scaler_{version}.pkl.enc.tmp")
    target_model = os.path.join(art_dir, f"model_{version}.pkl.enc")
    target_scaler = os.path.join(art_dir, f"scaler_{version}.pkl.enc")
    try:
        model_file.save(tmp_model); scaler_file.save(tmp_scaler)
        os.replace(tmp_model, target_model); os.replace(tmp_scaler, target_scaler)
        MODEL_ENC = target_model; SCALER_ENC = target_scaler; MODEL_VERSION = version
        key = load_key(FERNET_KEY_PATH)
        INFERENCE_SERVICE = InferenceService(MODEL_ENC, SCALER_ENC, key, model_version=MODEL_VERSION)
        return jsonify({"status": "ok", "model_version": MODEL_VERSION})
    except Exception as e:
        return jsonify({"error": f"failed to upload model: {e}"}), 500

@app.route("/admin/rotate_key", methods=["POST"])
@admin_required
def admin_rotate_key():
    global INFERENCE_SERVICE
    if "new_key" not in request.files:
        return jsonify({"error": "new_key file required"}), 400
    new_key = request.files["new_key"].read()
    from cryptography.fernet import Fernet, InvalidToken
    try: Fernet(new_key)
    except Exception as e:
        return jsonify({"error": f"invalid new key: {e}"}), 400
    try:
        old_key = load_key(FERNET_KEY_PATH)
    except Exception as e:
        return jsonify({"error": f"failed to load current key: {e}"}), 500
    enc_files = [f for f in os.listdir(os.path.dirname(MODEL_ENC) or ".") if f.endswith(".pkl.enc")]
    for fname in enc_files:
        full = os.path.join(os.path.dirname(MODEL_ENC), fname)
        try:
            with open(full, "rb") as ef:
                data = ef.read()
            dec = Fernet(old_key).decrypt(data)
            new_enc = Fernet(new_key).encrypt(dec)
            tmp = full + ".tmp"
            with open(tmp, "wb") as tf:
                tf.write(new_enc)
            os.replace(tmp, full)
        except InvalidToken:
            return jsonify({"error": f"invalid token while decrypting {fname} (old key mismatch)"}), 500
        except Exception as e:
            return jsonify({"error": f"failed to rotate {fname}: {e}"}), 500
    try:
        backup = FERNET_KEY_PATH + ".bak"
        if os.path.exists(FERNET_KEY_PATH):
            os.replace(FERNET_KEY_PATH, backup)
        with open(FERNET_KEY_PATH, "wb") as kf:
            kf.write(new_key)
        os.chmod(FERNET_KEY_PATH, 0o600)
    except Exception as e:
        return jsonify({"error": f"failed to replace key file: {e}"}), 500
    try:
        key = load_key(FERNET_KEY_PATH)
        INFERENCE_SERVICE = InferenceService(MODEL_ENC, SCALER_ENC, key, model_version=MODEL_VERSION)
    except Exception:
        return jsonify({"status": "key_rotated_but_reload_failed"}), 200
    return jsonify({"status": "ok", "message": "key rotated and artifacts re-encrypted"})

if __name__ == "__main__":
    import sys
    print("Starting backend app...")
    print("ARTIFACT_DIR:", ARTIFACT_DIR)
    sys.stdout.flush()

    # --- Force init manually, safe across all Flask versions ---
    try:
        print("Initializing inference service manually...")
        from backend.app import init_services
        init_services()
        print("✅ Inference service initialized successfully.")
    except Exception as e:
        print("⚠️  Inference init failed:", e)

    print("Starting Flask server...")
    sys.stdout.flush()
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
