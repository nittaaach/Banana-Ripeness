# Banana Ripeness Detection System 🍌

Sistem Deteksi Tingkat Kematangan Pisang Berbasis Pengolahan Citra Digital dengan Metode Hibrid:
**Preprocessing (Gaussian + Kontras) + Segmentasi HSV + Otsu Thresholding + Morfologi (Opening/Closing & Kontur Terbesar) + Ekstraksi Fitur Warna + Klasifikasi 4 Kelas**.

Sistem ini mereproduksi dan mengimplementasikan hasil penelitian benchmark 22 metode pengolahan citra digital, di mana metode campuran hibrid ini mencapai:
- **Keberhasilan Keseluruhan**: **96,9%**
- **Rata-rata IoU**: **0,934**
- **Kategori Kematangan**: Mentah, Matang, Terlalu Matang, dan Busuk.

---

## 📁 Struktur Direktori

```
Banana Ripeness/
├── config/
│   └── config.yaml               # Parameter pipeline, kernel, rentang HSV, ambang klasifikasi
├── Data/
│   ├── Raw/                      # Citra spesimen asli (Realme 11/12/14, 16 citra)
│   ├── ground_truth/             # Masker biner acuan untuk evaluasi IoU/Dice
│   └── output/                   # Hasil inferensi: mask, overlay, panels, results.csv
├── src/banana_ripeness/
│   ├── __init__.py
│   ├── config.py                 # Dataclass & loader YAML
│   ├── io_utils.py               # I/O citra & parsing metadata nama file
│   ├── preprocessing.py          # Resize 600x600, Gaussian blur, contrast stretch
│   ├── segmentation.py           # inRange HSV, Otsu thresholding, auto-inversion
│   ├── morphology.py             # Opening, closing, largest component, fill holes
│   ├── features.py               # Masked color statistics (Mean HSV, Rasio Hijau/Kuning/Coklat)
│   ├── classifier.py             # Rule-based 4-class decision engine
│   ├── pipeline.py               # Orkestrator BananaRipenessPipeline
│   ├── evaluation.py             # Metrik IoU, Dice, Success Rate, Confusion Matrix
│   └── visualization.py          # Panel komparasi 6 tahapan & overlay visual
├── scripts/
│   ├── run_single.py             # CLI inferensi satu citra
│   ├── run_batch.py              # CLI batch processing folder -> results.csv
│   ├── evaluate.py               # CLI evaluasi terhadap ground truth mask
│   ├── calibrate.py              # Tool kalibrasi sebaran fitur warna
│   └── create_ground_truth.py    # Tool pembuatan masker acuan ground truth
├── app/
│   └── streamlit_app.py          # Antarmuka Web interaktif (Streamlit)
├── tests/                        # Pengujian unit (pytest)
├── pyproject.toml
├── requirements.txt
├── arsitektur.md
└── prd.md
```

---

## 🚀 Panduan Instalasi & Persiapan

### 1. Prasyarat
- Python 3.10+ (disarankan Python 3.10)
- Sistem Operasi: Windows / Linux / macOS

### 2. Setup Virtual Environment
```bash
# Membuat virtual environment
python -m venv .venv

# Aktivasi virtual environment (Windows PowerShell)
.\.venv\Scripts\Activate.ps1

# Aktivasi virtual environment (Linux/macOS)
source .venv/bin/activate
```

### 3. Instalasi Dependensi
```bash
pip install -r requirements.txt
pip install -e .
```

---

## 🖥️ Penggunaan Sistem

### 1. Analisis Satu Citra (Single Image CLI)
Jalankan inferensi pada satu citra untuk melihat diagnosis kematangan dan menghasilkan visualisasi 6 tahapan:
```bash
python scripts/run_single.py Data/Raw/Realme_11_H11F1_Kanan_matang.jpg
```
Opsi tambahan:
- `--save-stages`: Menyimpan citra antara dari setiap tahapan pipeline.
- `--out <folder>`: Menentukan folder penyimpanan hasil.
- `--show`: Membuka jendela visualisasi popup.

### 2. Pemrosesan Batch Seluruh Dataset
Memproses seluruh citra dalam folder masukan, mengekspor masker biner, gambar overlay, dan file rekapitulasi `results.csv`:
```bash
python scripts/run_batch.py Data/Raw -o Data/output
```

### 3. Pembuatan Masker Ground Truth Awal
Menghasilkan masker acuan biner untuk ke-16 citra spesimen uji:
```bash
python scripts/create_ground_truth.py
```

### 4. Evaluasi Metrik terhadap Ground Truth
Menghitung Intersection over Union (IoU), Dice Similarity Coefficient, Success Rate ($\text{IoU} \ge 0{,}80$), dan Akurasi Klasifikasi:
```bash
python scripts/evaluate.py --raw Data/Raw --gt Data/ground_truth
```

### 5. Kalibrasi Distribusi Fitur Warna
Menganalisis sebaran nilai Mean H, Rasio Hijau, Rasio Kuning, dan Rasio Coklat untuk fine-tuning ambang batas klasifikasi:
```bash
python scripts/calibrate.py
```

### 6. Menjalankan Antarmuka Web Interaktif (Streamlit)
Aplikasi web visual lengkap untuk demonstrasi pengujian, penyesuaian parameter secara real-time, dan evaluasi dataset:
```bash
streamlit run app/streamlit_app.py
```

---

## 🧪 Menjalankan Pengujian Unit (Testing)
Untuk memastikan seluruh modul berfungsi secara independen dan terintegrasi:
```bash
pytest tests -v
```

---

## ⚙️ Konfigurasi (`config/config.yaml`)
Seluruh parameter sistem dipusatkan pada file `config/config.yaml` sehingga perubahan parameter tidak memerlukan modifikasi kode sumber:
- **Ukuran Citra**: `[600, 600]`
- **Gaussian Kernel**: `[5, 5]`
- **Rentang HSV**: `hsv_lower` dan `hsv_upper`
- **Kanal Otsu**: `"S"` (Saturation) atau `"V"` (Value)
- **Kombinasi Masker**: `"or"` / `"and"`
- **Operasi Morfologi**: `open_iterations`, `close_iterations`, `keep_largest_component`, `fill_holes`
- **Ambang Klasifikasi**: `green_ratio_unripe`, `brown_ratio_overripe`, `brown_ratio_rotten`
