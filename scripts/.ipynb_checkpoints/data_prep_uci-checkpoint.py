# scripts/data_prep_uci.py
import os, io, json, re, sys
import pandas as pd, numpy as np

BASE = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
RAW_DIR = os.path.join(BASE, "data", "raw")
OUT_DIR = os.path.join(BASE, "data", "processed")
SAMPLE_DIR = os.path.join(BASE, "data", "sample")
os.makedirs(OUT_DIR, exist_ok=True); os.makedirs(SAMPLE_DIR, exist_ok=True)

arffs = [f for f in os.listdir(RAW_DIR) if f.lower().endswith(".arff")]
if not arffs:
    print("No .arff found in data/raw/ — put the UCI ARFF there and re-run."); sys.exit(1)
arff_path = os.path.join(RAW_DIR, arffs[0])
print("Parsing:", arff_path)

with open(arff_path, "r", errors="ignore") as f:
    raw_lines = [l.rstrip("\n") for l in f]

attr_lines = []; data_idx = None
for i, line in enumerate(raw_lines):
    s = line.strip()
    if not s: continue
    if s.lower().startswith("@attribute"):
        attr_lines.append(s)
    if s.lower().startswith("@data"):
        data_idx = i + 1
        break

if data_idx is None:
    print("No @data section found in ARFF"); sys.exit(1)

attr_names = []
pat = re.compile(r"@attribute\s+(['\"]?)([^'\"]+)\1\s+(.+)", re.IGNORECASE)
for a in attr_lines:
    m = pat.match(a)
    if m:
        name = m.group(2).strip().replace(" ", "_").replace("/", "_").replace("(", "").replace(")", "")
        attr_names.append(name)
    else:
        parts = a.split()
        if len(parts) >= 2:
            name = parts[1].strip().replace(" ", "_").replace("/", "_")
            attr_names.append(name)

data_lines = raw_lines[data_idx:]
data_lines = [l for l in data_lines if l.strip() and not l.strip().startswith("%")]
csv_text = "\n".join(data_lines)
df = pd.read_csv(io.StringIO(csv_text), header=None, dtype=str, na_values=["?",""])
if df.shape[1] == len(attr_names):
    df.columns = attr_names
else:
    minc = min(df.shape[1], len(attr_names))
    df = df.iloc[:, :minc]
    df.columns = attr_names[:minc]

# basic cleaning
for c in df.select_dtypes(include=['object']).columns:
    df[c] = df[c].astype(str).str.strip().replace({'': None, 'None': None, 'nan': None})

possible_label = [c for c in df.columns if c.lower().startswith("class") or "asd" in c.lower() or c.lower()=="label"]
label_col = possible_label[0] if possible_label else df.columns[-1]

def map_label(x):
    if pd.isna(x): return np.nan
    s = str(x).strip().upper()
    if s in ("YES","Y","1","TRUE","T"): return 1
    if s in ("NO","N","0","FALSE","F"): return 0
    try: return int(float(s))
    except: return np.nan

df['label'] = df[label_col].apply(map_label)
if label_col != 'label':
    try: df = df.drop(columns=[label_col])
    except: pass

# drop PII-like fields if present
for drop_col in ['ethnicity','country_of_residence','relation','who_completed_the_test','country','used_app_before']:
    if drop_col in df.columns:
        df = df.drop(columns=[drop_col])

# numeric conversions
for i in range(1, 11):
    col = f"A{i}_Score"
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
if 'age' in df.columns:
    df['age'] = pd.to_numeric(df['age'], errors='coerce')

df = df.replace('?', np.nan)
num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
if num_cols:
    df[num_cols] = df[num_cols].fillna(df[num_cols].median())

cat_cols = df.select_dtypes(include=['object']).columns.tolist()
for c in cat_cols:
    nunique = df[c].nunique(dropna=True)
    if nunique == 0:
        df = df.drop(columns=[c])
    elif nunique <= 10:
        dummies = pd.get_dummies(df[c], prefix=c, drop_first=True)
        df = pd.concat([df.drop(columns=[c]), dummies], axis=1)
    else:
        df = df.drop(columns=[c])

df = df.dropna(subset=['label'])
df['label'] = df['label'].astype(int)
# sanity: balance labels if extremely skewed
label_counts = df['label'].value_counts()
if len(label_counts) < 2:
    print("⚠️ Only one label found after cleaning. Regenerating synthetic negatives.")
    if 1 in label_counts:
        df_zero = df.sample(min(50, len(df)), replace=True).copy()
        df_zero['label'] = 0
        df = pd.concat([df, df_zero], ignore_index=True)
    elif 0 in label_counts:
        df_one = df.sample(min(50, len(df)), replace=True).copy()
        df_one['label'] = 1
        df = pd.concat([df, df_one], ignore_index=True)

out_csv = os.path.join(OUT_DIR, "autism_clean.csv")
df.to_csv(out_csv, index=False)
meta = {"input_file": arff_path, "output_file": out_csv, "rows": int(df.shape[0]), "cols": int(df.shape[1]), "label_col": "label"}
with open(os.path.join(OUT_DIR, "meta.json"), "w") as f:
    json.dump(meta, f, indent=2)

if df.shape[0] > 0:
    df.iloc[[0]].to_csv(os.path.join(SAMPLE_DIR, "sample_row.csv"), index=False)

print("Processed saved to:", out_csv)
