# backend/core/inference.py
import pandas as pd
import numpy as np
import joblib
import io
import base64
import matplotlib.pyplot as plt

from cryptography.fernet import Fernet, InvalidToken
from .model_store import load_model_and_scaler

# Optional SHAP
try:
    import shap
    import matplotlib.pyplot as plt
except Exception:
    shap = None
    plt = None


class InferenceService:
    def __init__(self, model_enc_path: str, scaler_enc_path: str, fernet_key_bytes: bytes, model_version: str = "v1"):
        self.model_enc_path = model_enc_path
        self.scaler_enc_path = scaler_enc_path
        self.fernet_key = fernet_key_bytes
        self.model_version = model_version

        self.model, self.scaler = load_model_and_scaler(
            self.model_enc_path, self.scaler_enc_path, self.fernet_key
        )

        # Extract feature names
        if hasattr(self.model, "feature_names_in_"):
            self.feature_names = list(self.model.feature_names_in_)
        else:
            self.feature_names = None

        print(f"✅ Model loaded: {type(self.model)}")
        print(f"✅ Scaler loaded: {type(self.scaler)}")

        if hasattr(self.scaler, "mean_"):
            print(f"Scaler expects {len(self.scaler.mean_)} features.")
        else:
            print("Scaler feature count unknown.")
        # Initialize SHAP (robust version with masker)
        global shap
        try:
            import shap
            import matplotlib.pyplot as plt
            print("🔍 SHAP imported successfully.")

            # Build a masker from the scaler's expected input shape
            import numpy as np
            n_features = (
                len(self.scaler.mean_)
                if hasattr(self.scaler, "mean_")
                else (len(self.feature_names) if self.feature_names else 1)
            )
            background = np.zeros((1, n_features))
            masker = shap.maskers.Independent(background)

            # Use LinearExplainer for linear models (like LogisticRegression)
            if "LogisticRegression" in str(type(self.model)):
                self.explainer = shap.LinearExplainer(
                    self.model, masker=masker, feature_perturbation="interventional"
                )
                print("✅ SHAP LinearExplainer initialized with masker for LogisticRegression.")
            else:
                self.explainer = shap.Explainer(self.model, masker=masker)
                print(f"✅ SHAP generic Explainer initialized with masker for {type(self.model)}.")

        except Exception as e:
            print(f"⚠️ SHAP initialization failed: {e}")
            self.explainer = None



    # --------------------------------------------------------------------------
    def _align_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Align incoming dataframe with scaler's expected feature count"""
        if hasattr(self.scaler, "mean_"):
            n_expected = len(self.scaler.mean_)
            if df.shape[1] != n_expected:
                print(f"⚠️ Feature mismatch: expected {n_expected}, got {df.shape[1]}. Auto-aligning...")
                df = df.select_dtypes(include=["number"])
                # pad with zeros if missing
                while df.shape[1] < n_expected:
                    df[f"extra_{df.shape[1]}"] = 0
                # trim extras
                df = df.iloc[:, :n_expected]
        return df

    # --------------------------------------------------------------------------
    def _render_shap_for_row(self, Xs):
        """Generate SHAP waterfall plot for a single input row."""
        if self.explainer is None or plt is None or shap is None:
            return None, ""
        try:
            shap_values = self.explainer(Xs)
            # Use waterfall plot for single prediction
            shap.plots.waterfall(shap_values[0], show=False)
            buf = io.BytesIO()
            plt.tight_layout()
            plt.savefig(buf, format="png", bbox_inches="tight")
            plt.close()
            buf.seek(0)
            img_b64 = base64.b64encode(buf.read()).decode("utf-8")
            return img_b64, "Feature importance visualization available."
        except Exception as e:
            return None, f"SHAP generation failed: {e}"


    # --------------------------------------------------------------------------
    def predict_and_explain(self, df: pd.DataFrame):
        """Run prediction + optional SHAP explanation"""
        if df is None or df.empty:
            raise ValueError("Empty dataframe provided for inference.")

        # Align
        df = self._align_features(df)

        # Scale safely
        try:
            X_scaled = self.scaler.transform(df)
        except Exception as e:
            raise RuntimeError(f"Scaling failed: {e}")

        if X_scaled is None or len(X_scaled) == 0:
            raise ValueError("Scaled input empty after transformation.")

        # Predict safely
        try:
            probas = self.model.predict_proba(X_scaled)
        except Exception as e:
            raise RuntimeError(f"Prediction failed: {e}")

        if probas is None or probas.shape[0] == 0:
            raise ValueError("predict_proba returned empty array.")
        if probas.shape[1] < 2:
            # if model gives only one column
            proba = float(probas[0, 0])
        else:
            proba = float(probas[0, 1])

        pred = int(proba >= 0.5)

        # SHAP / fallback explanation
        shap_b64, explanation = self._render_shap_for_row(df)
        if not explanation:
            if hasattr(self.model, "coef_"):
                coef_abs = np.abs(self.model.coef_[0])
                top_idx = np.argsort(coef_abs)[::-1][:3]
                feats = self.feature_names or [f"f{i}" for i in range(len(coef_abs))]
                top_feats = [feats[i] for i in top_idx]
                explanation = f"Top contributing features: {', '.join(top_feats)}"
            else:
                explanation = "Explainability unavailable for this model."

        return {
            "model_version": self.model_version,
            "probability": proba,
            "prediction": pred,
            "explanation": explanation,
            "shap_b64": shap_b64,
        }
