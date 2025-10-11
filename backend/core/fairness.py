# backend/core/fairness.py
import json, os
def load_fairness(json_path: str):
    if not os.path.exists(json_path):
        return {"note": "fairness metrics not provided"}
    try:
        with open(json_path, "r") as f:
            return json.load(f)
    except Exception:
        return {"note": "failed to read fairness json"}
