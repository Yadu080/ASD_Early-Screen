import os
import json
from pathlib import Path
from flask import Flask, render_template, request, jsonify, send_from_directory
import requests
import pandas as pd

app = Flask(__name__, static_folder="static", template_folder="templates")
app.config["TEMPLATES_AUTO_RELOAD"] = True

# Backend URL (where your backend app runs)
BACKEND_URL = os.environ.get("BACKEND_URL", "http://127.0.0.1:5000")

# sample path (created by data_prep_uci.py)
SAMPLE_PATH = Path("data/sample/sample_row.csv")

def load_sample_row():
    if SAMPLE_PATH.exists():
        df = pd.read_csv(SAMPLE_PATH)
        row = df.iloc.to_dict()
        # drop label if present
        row.pop("label", None)
        # coerce numeric strings to numbers where possible
        for k,v in list(row.items()):
            try:
                row[k] = float(v)
            except Exception:
                pass
        return row
    return None

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html", backend=BACKEND_URL)

@app.route("/api/sample", methods=["GET"])
def api_sample():
    s = load_sample_row()
    if s is None:
        return jsonify({"error": "sample row not found"}), 404
    return jsonify(s)

@app.route("/api/predict", methods=["POST"])
def api_predict():
    """
    Accept JSON or form data -> proxy to backend /predict -> return JSON.
    Query param include_shap=true to request shap image (default: false to keep response small)
    """
    include_shap = request.args.get("include_shap", "false").lower() == "true"

    # Accept JSON payload
    data = {}
    if request.is_json:
        data = request.get_json(force=True)
    else:
        # Accept form encoded (from the HTML form)
        data = {k: try_parse_number(v) for k,v in request.form.items() if v != ""}

    # Proxy to backend
    try:
        url = f"{BACKEND_URL}/predict"
        if include_shap:
            url += "?include_shap=true"
        resp = requests.post(url, json=data, timeout=30)
        resp.raise_for_status()
        result = resp.json()
        # remove raw shap_b64 from any logs/return unless explicitly requested
        if not include_shap and "shap_b64" in result:
            result.pop("shap_b64", None)
        return jsonify(result)
    except requests.exceptions.RequestException as e:
        return jsonify({"error": f"backend request failed: {e}"}), 502
    except Exception as e:
        return jsonify({"error": str(e)}), 500

def try_parse_number(v):
    try:
        if isinstance(v, (int, float)):
            return v
        if isinstance(v, str) and v.strip() == "":
            return None
        if isinstance(v, str):
            if "." in v:
                return float(v)
            return int(v)
    except Exception:
        pass
    return v

# Serve static files with cache busting
@app.route('/static/<path:filename>')
def static_files(filename):
    return send_from_directory('static', filename)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8890))
    app.run(host="0.0.0.0", port=port, debug=True)
