import os, glob
import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(ROOT, "data", "ISIC2019_224")
CSV = os.path.join(DATA_DIR, "train-metadata.csv")
IMG_DIR = os.path.join(DATA_DIR, "train-image", "image")

def _stem(path):
    return os.path.splitext(os.path.basename(path))[0]

def load_isic224():
    df = pd.read_csv(CSV)
    df.rename(columns=str.lower, inplace=True)
    assert {"isic_id","target"}.issubset(df.columns), "train-metadata.csv must have isic_id,target"

    # Build a set of actually present image IDs (support jpg/jpeg/png)
    files = []
    for ext in ("*.jpg","*.jpeg","*.png","*.JPG","*.JPEG","*.PNG"):
        files.extend(glob.glob(os.path.join(IMG_DIR, ext)))
    existing_ids = { _stem(p) for p in files }

    # Keep only rows we actually have
    before = len(df)
    df = df[df["isic_id"].astype(str).isin(existing_ids)].copy()
    after = len(df)
    dropped = before - after
    print(f"Found {len(existing_ids)} images on disk. Using {after} rows. Dropped {dropped} missing.")

    # Create image_path (prefer existing extension)
    ext_map = {}
    for p in files:
        ext_map[_stem(p)] = os.path.splitext(p)[1].lower()   # .jpg/.png
    df["image_path"] = df["isic_id"].apply(lambda x: os.path.join(IMG_DIR, f"{x}{ext_map[str(x)]}"))

    df["risk_bin"] = df["target"].map({0:"benign", 1:"malignant"})
    return df[["image_path","risk_bin","target"]]

def stratified_splits(df, val=0.15, test=0.15, seed=42):
    train_df, temp_df = train_test_split(df, test_size=val+test, stratify=df["target"], random_state=seed)
    val_df,   test_df = train_test_split(temp_df, test_size=test/(val+test), stratify=temp_df["target"], random_state=seed)
    return train_df, val_df, test_df

def save_splits(tr, va, te):
    out = os.path.join(ROOT, "outputs"); os.makedirs(out, exist_ok=True)
    tr.to_csv(os.path.join(out,"train_split.csv"), index=False)
    va.to_csv(os.path.join(out,"val_split.csv"), index=False)
    te.to_csv(os.path.join(out,"test_split.csv"), index=False)
    print("Saved splits into outputs/")

if __name__ == "__main__":
    df = load_isic224()
    print(df["risk_bin"].value_counts())
    tr, va, te = stratified_splits(df)
    print(f"Train {len(tr)} | Val {len(va)} | Test {len(te)}")
    save_splits(tr, va, te)