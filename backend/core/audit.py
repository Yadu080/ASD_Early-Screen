# backend/core/audit.py
import json, time, hashlib, logging, os
LOG_PATH = os.environ.get("LOG_PATH", "backend/logs/inference.log")
os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
logging.basicConfig(filename=LOG_PATH, level=logging.INFO, format="%(message)s")
def log_inference(input_data: dict, model_version: str, latency_ms: int, prob: float, pred: int):
    try:
        items = sorted(((str(k), str(input_data[k])) for k in input_data))
        s = json.dumps(items, separators=(",", ":"))
        input_hash = hashlib.sha256(s.encode("utf-8")).hexdigest()
    except Exception:
        input_hash = None
    entry = {"ts": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "input_hash": input_hash, "model_version": model_version, "latency_ms": latency_ms, "probability": prob, "prediction": pred}
    try:
        logging.info(json.dumps(entry))
    except Exception:
        pass
