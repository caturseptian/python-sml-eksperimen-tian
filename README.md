# Eksperimen SML - Tian

Repository **Kriteria 1** submission MSML (Membangun Sistem Machine Learning).
Berisi eksperimen & otomasi preprocessing dataset **Breast Cancer Wisconsin (Diagnostic)**.

> Repository ini **HARUS public** agar dapat diperiksa reviewer.

## Struktur

```
python-sml-eksperimen-tian/
├── .github/workflows/preprocessing.yml      # GitHub Actions: preprocessing otomatis
├── breast_cancer_raw/
│   └── breast_cancer_raw.csv                 # dataset mentah (raw)
└── preprocessing/
    ├── Eksperimen_Tian.ipynb         # notebook eksperimen (template MSML)
    ├── automate_tian.py              # otomasi preprocessing (reusable functions)
    └── breast_cancer_preprocessing/          # output: train.csv, test.csv, scaler, dst
```

## Dataset

- **Sumber:** Breast Cancer Wisconsin (Diagnostic), tersedia di `sklearn.datasets.load_breast_cancer`.
- **Baris:** 569 | **Fitur:** 30 numerik | **Target:** `diagnosis` (malignant/benign).
- File `breast_cancer_raw.csv` dihasilkan dari sklearn (label target dibuat readable).

## Tahapan Eksperimen (Notebook)

`preprocessing/Eksperimen_Tian.ipynb` mengikuti template MSML:
1. **Import Library**
2. **Data Loading** - membaca CSV mentah
3. **EDA** - statistik, distribusi target, missing value, duplikat, korelasi
4. **Preprocessing manual** - cleaning, encoding target, split, standardisasi
5. **Menyimpan dataset** hasil preprocessing

## Cara Menjalankan

### 1) Notebook
```bash
pip install scikit-learn==1.5.2 pandas==2.2.3 numpy==1.26.4 matplotlib seaborn joblib jupyter
cd preprocessing
jupyter notebook Eksperimen_Tian.ipynb   # jalankan semua cell (Run All)
```

### 2) Script otomasi
> Catatan: nama file memuat tanda kurung, gunakan tanda kutip di terminal.
```bash
cd preprocessing
python "automate_tian.py" \
  --input ../breast_cancer_raw/breast_cancer_raw.csv \
  --outdir breast_cancer_preprocessing
```
Output: `train.csv`, `test.csv`, `scaler.joblib`, `feature_names.json`.

### 3) GitHub Actions (otomatis)
Workflow `.github/workflows/preprocessing.yml` berjalan saat:
- ada **push** yang mengubah data mentah / script automasi, atau
- dijalankan manual via tab **Actions → Automated Preprocessing → Run workflow** (`workflow_dispatch`).

Workflow akan menjalankan `automate_tian.py`, meng-commit dataset hasil
preprocessing terbaru kembali ke repo, dan mengunggahnya sebagai artifact.

## Catatan Konsistensi
Logika preprocessing pada `automate_tian.py` **identik** dengan langkah di notebook,
sehingga hasil pada notebook dan workflow akan sama.
