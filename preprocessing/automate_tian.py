"""
automate_tian.py
========================
Otomatisasi preprocessing dataset Breast Cancer (Wisconsin).

File ini merupakan konversi dari langkah-langkah preprocessing manual yang
dikerjakan pada notebook `Eksperimen_Tian.ipynb` menjadi fungsi-fungsi
yang reusable. Logika di sini HARUS identik dengan langkah pada notebook:

    1. Load data mentah (raw)
    2. Pembersihan data (hapus duplikat, tangani missing value)
    3. Encoding target (malignant=1, benign=0)
    4. Split train/test (stratified)
    5. Standardisasi fitur numerik (fit hanya pada data train)
    6. Simpan dataset hasil preprocessing + objek preprocessor

Dipakai oleh GitHub Actions workflow `preprocessing.yml` agar dataset
hasil preprocessing selalu ter-update secara otomatis.

Cara pakai (lokal):
    python "automate_tian.py" \
        --input ../breast_cancer_raw/breast_cancer_raw.csv \
        --outdir breast_cancer_preprocessing
"""

from __future__ import annotations

import argparse
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# Konfigurasi global
# ---------------------------------------------------------------------------
TARGET_COLUMN = "diagnosis"          # kolom target pada dataset
POSITIVE_LABEL = "malignant"          # kelas yang dianggap positif (1)
NEGATIVE_LABEL = "benign"             # kelas yang dianggap negatif (0)
RANDOM_STATE = 42
TEST_SIZE = 0.2


# ---------------------------------------------------------------------------
# 1. Load data
# ---------------------------------------------------------------------------
def load_raw_data(path: str) -> pd.DataFrame:
    """Membaca dataset mentah dari file CSV."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"File raw tidak ditemukan: {path}")
    df = pd.read_csv(path)
    print(f"[load] Data mentah dimuat: {df.shape[0]} baris, {df.shape[1]} kolom")
    return df


# ---------------------------------------------------------------------------
# 2. Pembersihan data
# ---------------------------------------------------------------------------
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Menghapus duplikat dan menangani missing value.

    Untuk dataset Breast Cancer tidak terdapat missing value, namun langkah ini
    tetap dibuat robust agar bisa dipakai ulang pada dataset lain.
    """
    df = df.copy()

    # hapus baris duplikat
    n_dup = int(df.duplicated().sum())
    if n_dup > 0:
        df = df.drop_duplicates().reset_index(drop=True)
    print(f"[clean] Duplikat dihapus: {n_dup}")

    # isi missing value numerik dengan median (jika ada)
    num_cols = df.select_dtypes(include=[np.number]).columns
    n_missing = int(df[num_cols].isna().sum().sum())
    if n_missing > 0:
        df[num_cols] = df[num_cols].fillna(df[num_cols].median())
    print(f"[clean] Missing value numerik diisi median: {n_missing}")

    return df


# ---------------------------------------------------------------------------
# 3. Encoding target
# ---------------------------------------------------------------------------
def encode_target(df: pd.DataFrame) -> pd.DataFrame:
    """Mengubah label target menjadi biner (malignant=1, benign=0)."""
    df = df.copy()
    # Target bertipe non-numerik (object/string) perlu di-encode dulu.
    if not pd.api.types.is_numeric_dtype(df[TARGET_COLUMN]):
        mapping = {NEGATIVE_LABEL: 0, POSITIVE_LABEL: 1}
        df[TARGET_COLUMN] = df[TARGET_COLUMN].map(mapping)
        if df[TARGET_COLUMN].isna().any():
            raise ValueError(
                "Terdapat label target yang tidak dikenal saat encoding. "
                f"Label valid: {list(mapping.keys())}"
            )
    df[TARGET_COLUMN] = df[TARGET_COLUMN].astype(int)
    print(f"[encode] Distribusi target setelah encoding:\n{df[TARGET_COLUMN].value_counts().to_dict()}")
    return df


# ---------------------------------------------------------------------------
# 4 & 5. Split + standardisasi
# ---------------------------------------------------------------------------
def split_and_scale(df: pd.DataFrame):
    """Split stratified lalu standardisasi fitur (fit hanya pada train).

    Mengembalikan: train_df, test_df, scaler, feature_names
    """
    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]
    feature_names = list(X.columns)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    train_df = pd.DataFrame(X_train_scaled, columns=feature_names)
    train_df[TARGET_COLUMN] = y_train.reset_index(drop=True)

    test_df = pd.DataFrame(X_test_scaled, columns=feature_names)
    test_df[TARGET_COLUMN] = y_test.reset_index(drop=True)

    print(f"[split] train: {train_df.shape}, test: {test_df.shape}")
    return train_df, test_df, scaler, feature_names


# ---------------------------------------------------------------------------
# 6. Simpan hasil
# ---------------------------------------------------------------------------
def save_outputs(train_df, test_df, scaler, feature_names, outdir: str):
    """Menyimpan dataset hasil preprocessing dan objek preprocessor."""
    os.makedirs(outdir, exist_ok=True)
    train_path = os.path.join(outdir, "train.csv")
    test_path = os.path.join(outdir, "test.csv")
    scaler_path = os.path.join(outdir, "scaler.joblib")
    features_path = os.path.join(outdir, "feature_names.json")

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    joblib.dump(scaler, scaler_path)
    pd.Series(feature_names).to_json(features_path, orient="values", indent=2)

    print(f"[save] train  -> {train_path}")
    print(f"[save] test   -> {test_path}")
    print(f"[save] scaler -> {scaler_path}")
    print(f"[save] feats  -> {features_path}")


# ---------------------------------------------------------------------------
# Pipeline utama (dipanggil notebook & workflow)
# ---------------------------------------------------------------------------
def preprocess_pipeline(input_path: str, outdir: str):
    """Menjalankan seluruh tahapan preprocessing secara berurutan."""
    df = load_raw_data(input_path)
    df = clean_data(df)
    df = encode_target(df)
    train_df, test_df, scaler, feature_names = split_and_scale(df)
    save_outputs(train_df, test_df, scaler, feature_names, outdir)
    print("[done] Preprocessing selesai.")
    return train_df, test_df


def _parse_args():
    parser = argparse.ArgumentParser(description="Automasi preprocessing dataset breast cancer.")
    parser.add_argument(
        "--input",
        default=os.path.join(
            os.path.dirname(__file__), "..", "breast_cancer_raw", "breast_cancer_raw.csv"
        ),
        help="Path ke file CSV mentah.",
    )
    parser.add_argument(
        "--outdir",
        default=os.path.join(os.path.dirname(__file__), "breast_cancer_preprocessing"),
        help="Folder output dataset hasil preprocessing.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    preprocess_pipeline(args.input, args.outdir)
