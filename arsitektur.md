# Arsitektur — Pipeline Hibrid Deteksi Kematangan Pisang

Dokumen ini menjelaskan struktur kode, alur data, dan desain tiap modul untuk metode **Preprocessing + HSV + Otsu + Morfologi**.

---

## 1. Gambaran Umum

Sistem berupa pipeline linear dengan dua fase besar:

1. **Segmentasi** — memisahkan pisang dari latar (metode hibrid yang di-benchmark).
2. **Klasifikasi** — mengubah area pisang yang tersegmentasi menjadi label kematangan.

```
 Citra input
     │
     ▼
┌─────────────┐   ┌───────────────┐   ┌────────────────────┐   ┌─────────────┐
│ 1. Load &   │──▶│ 2. Preprocess │──▶│ 3. Segmentasi      │──▶│ 4. Morfologi│
│    Resize   │   │ Gaussian,     │   │ HSV mask + Otsu    │   │ Open→Close, │
│  (600×600)  │   │ (kontras)     │   │ (gabung mask)      │   │ kontur terbesar│
└─────────────┘   └───────────────┘   └────────────────────┘   └──────┬──────┘
                                                                      │ mask biner
                                   ┌──────────────────────────────────┤
                                   ▼                                  ▼
                          ┌─────────────────┐               ┌──────────────────┐
                          │ 5. Ekstraksi    │──────────────▶│ 6. Klasifikasi   │
                          │ fitur warna     │               │ kematangan       │
                          └─────────────────┘               └────────┬─────────┘
                                                                     ▼
                                                  Label + mask + overlay + CSV
                                                                     │
                                                                     ▼
                                                    7. Evaluasi (IoU, akurasi) *
```
\* hanya saat ground truth tersedia.

## 2. Struktur Proyek

```
banana-ripeness/
├── config/
│   └── config.yaml              # semua parameter (kernel, rentang HSV, ambang klasifikasi)
├── data/
│   ├── raw/                     # citra asli
│   ├── ground_truth/            # masker referensi untuk IoU
│   └── output/                  # mask, overlay, results.csv
├── src/banana_ripeness/
│   ├── __init__.py
│   ├── config.py                # loader & dataclass konfigurasi
│   ├── io_utils.py              # baca/tulis citra, daftar file, parse label dari nama file
│   ├── preprocessing.py         # resize, Gaussian, contrast stretching
│   ├── segmentation.py          # mask HSV, Otsu, penggabungan mask
│   ├── morphology.py            # opening, closing, kontur terbesar, isi lubang
│   ├── features.py              # fitur warna dari area mask
│   ├── classifier.py            # klasifikasi 4 kelas
│   ├── pipeline.py              # orkestrasi seluruh tahap
│   ├── evaluation.py            # IoU, Dice, akurasi, confusion matrix
│   └── visualization.py         # overlay & panel visual tiap tahap
├── scripts/
│   ├── run_single.py            # CLI satu citra
│   ├── run_batch.py             # CLI satu folder → CSV
│   └── evaluate.py              # CLI evaluasi terhadap ground truth
├── app/
│   └── streamlit_app.py         # (opsional) UI web
├── tests/
│   ├── test_preprocessing.py
│   ├── test_segmentation.py
│   ├── test_morphology.py
│   ├── test_features.py
│   ├── test_classifier.py
│   └── test_evaluation.py
├── docs/
│   ├── prd.md
│   └── arsitektur.md
├── requirements.txt
└── README.md
```

## 3. Rancangan Modul

### 3.1 `config.py` + `config.yaml`

Memusatkan parameter agar tuning tidak mengubah kode.

```yaml
image:
  size: [600, 600]

preprocessing:
  gaussian_kernel: [5, 5]
  gaussian_sigma: 0
  contrast_stretch: true
  stretch_percentiles: [2, 98]

segmentation:
  hsv_lower: [10, 40, 40]      # kalibrasi dengan data sendiri
  hsv_upper: [90, 255, 255]    # mencakup hijau–kuning–jingga
  otsu_channel: "S"            # S atau V
  combine_mode: "or"           # "and" atau "or" (uji keduanya)

morphology:
  kernel_shape: "ellipse"
  kernel_size: [5, 5]
  open_iterations: 1
  close_iterations: 2
  keep_largest_component: true
  fill_holes: true

classification:
  # ambang diisi hasil kalibrasi dari data (lihat bagian 5)
  hue_green_max: 35
  hue_yellow_max: 28
  brown_ratio_overripe: 0.15
  brown_ratio_rotten: 0.45
```

> Nilai di atas adalah titik awal, **bukan hasil final**. Rentang HSV OpenCV: H 0–179, S 0–255, V 0–255. Kalibrasi wajib dilakukan pada dataset Anda.

### 3.2 `preprocessing.py`

```python
def resize(img, size=(600, 600)): ...
def gaussian_blur(img, ksize=(5, 5), sigma=0): ...
def contrast_stretch(img, p_low=2, p_high=98): ...   # opsional, per kanal / pada V
def preprocess(img, cfg) -> np.ndarray: ...
```

### 3.3 `segmentation.py`

```python
def to_hsv(img_bgr) -> np.ndarray: ...
def hsv_mask(hsv, lower, upper) -> np.ndarray: ...          # cv2.inRange
def otsu_mask(channel) -> np.ndarray:                        # cv2.threshold(... THRESH_BINARY + THRESH_OTSU)
    ...
def combine_masks(m_hsv, m_otsu, mode="or") -> np.ndarray: ...
def segment(img_pre, cfg) -> dict:                           # kembalikan mask antara untuk visualisasi
    ...
```

Catatan desain: Otsu mengasumsikan histogram bimodal. Jika latar lebih terang/lebih saturasi daripada objek, mask bisa terbalik. Tambahkan pemeriksaan (mis. mask yang menutupi tepi citra dianggap latar) lalu invert bila perlu.

### 3.4 `morphology.py`

```python
def make_kernel(shape, size) -> np.ndarray: ...
def opening(mask, kernel, iters=1): ...      # buang bintik latar
def closing(mask, kernel, iters=2): ...      # tutup lubang akibat bercak busuk
def largest_component(mask): ...             # cv2.findContours / connectedComponentsWithStats
def fill_holes(mask): ...                    # drawContours dengan FILLED
def refine(mask, cfg): ...                   # rangkai semuanya
```

### 3.5 `features.py`

Fitur dihitung **hanya pada piksel di dalam mask**:

| Fitur | Keterangan |
|---|---|
| `mean_h, mean_s, mean_v` | rata-rata HSV (gunakan rata-rata sirkular untuk H bila perlu) |
| `ratio_green` | proporsi piksel dengan H pada rentang hijau |
| `ratio_yellow` | proporsi piksel kuning |
| `ratio_brown` | proporsi piksel coklat/gelap (H rendah + V rendah) |
| `area_ratio` | luas mask / luas citra (untuk sanity check) |

```python
def extract_features(img_bgr, mask) -> dict: ...
```

### 3.6 `classifier.py`

Tahap awal: **klasifikasi berbasis aturan** (transparan, sesuai materi kuliah).

```python
def classify(features: dict, cfg) -> str:
    if features["ratio_brown"] >= cfg.brown_ratio_rotten:
        return "Busuk"
    if features["ratio_brown"] >= cfg.brown_ratio_overripe:
        return "Terlalu Matang"
    if features["ratio_green"] >= 0.5:
        return "Mentah"
    return "Matang"
```

Opsi peningkatan (jika aturan kurang akurat): latih `DecisionTree`/`kNN`/`SVM` dari scikit-learn pada vektor fitur di atas; antarmuka `classify()` tidak berubah.

### 3.7 `pipeline.py`

```python
@dataclass
class PipelineResult:
    label: str
    mask: np.ndarray
    features: dict
    stages: dict          # citra antara untuk visualisasi

class BananaRipenessPipeline:
    def __init__(self, cfg): ...
    def run(self, img_bgr) -> PipelineResult:
        img = preprocess(img_bgr, self.cfg)
        seg = segment(img, self.cfg)
        mask = refine(seg["combined"], self.cfg)
        feats = extract_features(img, mask)
        label = classify(feats, self.cfg)
        return PipelineResult(label, mask, feats, {**seg, "final": mask})
```

### 3.8 `evaluation.py`

```python
def iou(pred, gt) -> float:
    inter = np.logical_and(pred > 0, gt > 0).sum()
    union = np.logical_or(pred > 0, gt > 0).sum()
    return inter / union if union else 1.0

def dice(pred, gt) -> float: ...
def success_rate(iou_list, threshold=0.8) -> float: ...
def confusion(y_true, y_pred, labels): ...    # sklearn.metrics
def evaluate_folder(raw_dir, gt_dir, pipeline) -> pandas.DataFrame: ...
```

Output evaluasi: tabel per citra (file, kondisi, IoU, label prediksi, benar/salah), rekap per kondisi, dan rekap keseluruhan. Format ini setara dengan Tabel 2 & 4 di dashboard sehingga hasil dapat dibandingkan langsung.

### 3.9 `visualization.py`

```python
def overlay(img, mask, alpha=0.4): ...
def stage_panel(stages: dict, path): ...      # grid: asli | blur | HSV | Otsu | morfologi | overlay
```

## 4. Antarmuka (CLI)

| Perintah | Fungsi |
|---|---|
| `python scripts/run_single.py data/raw/foto.jpg --show` | satu citra → label + overlay |
| `python scripts/run_batch.py data/raw --out data/output` | batch → `results.csv` + mask |
| `python scripts/evaluate.py --raw data/raw --gt data/ground_truth` | IoU, akurasi, confusion matrix |

## 5. Alur Kalibrasi (urutan kerja yang disarankan)

1. Kumpulkan 16 citra + buat ground-truth mask.
2. Jalankan segmentasi saja; tuning `hsv_lower/upper` dan `combine_mode` sampai IoU memuaskan, dilihat per kondisi (terutama Busuk).
3. Ukur distribusi fitur (`ratio_*`, `mean_h`) per kelas — buat boxplot/tabel.
4. Tetapkan ambang klasifikasi dari distribusi tersebut.
5. Evaluasi ulang menyeluruh; bila akurasi kurang, ganti aturan dengan classifier scikit-learn.
6. Hindari menyetel ambang pada seluruh 16 citra lalu melaporkan hasil pada citra yang sama tanpa catatan; pisahkan data kalibrasi dan uji bila memungkinkan (mis. kalibrasi di sisi Kanan, uji di sisi Kiri).

## 6. Dependensi

```
opencv-python>=4.9
numpy>=1.26
scikit-image>=0.22
scikit-learn>=1.4
pandas>=2.2
matplotlib>=3.8
pyyaml>=6.0
pytest>=8.0
streamlit>=1.35      # opsional
```

## 7. Strategi Pengujian

| Modul | Uji |
|---|---|
| preprocessing | ukuran output = 600×600; blur menurunkan variansi |
| segmentation | citra sintetis (kotak kuning di latar putih) menghasilkan mask yang benar |
| morphology | lubang kecil tertutup; komponen kecil terbuang; hanya 1 komponen tersisa |
| features | rasio berjumlah ≤ 1; nilai pada citra warna tunggal sesuai harapan |
| classifier | kasus batas tiap kelas |
| evaluation | IoU = 1 untuk mask identik, 0 untuk mask terpisah |
| pipeline | tes integrasi pada 1 citra per kondisi |

## 8. Keputusan Desain & Alasan

| Keputusan | Alasan |
|---|---|
| Pipeline fungsional kecil-kecil | Mudah diuji dan dipakai ulang di notebook |
| Konfigurasi di YAML | Tuning tanpa mengubah kode |
| Klasifikasi berbasis aturan dulu | Sesuai ruang lingkup mata kuliah, mudah dijelaskan saat presentasi |
| Menyimpan citra antara (`stages`) | Memudahkan debugging dan figur laporan |
| Evaluasi terpisah dari pipeline | Pipeline tetap bisa dipakai tanpa ground truth |

## 9. Pengembangan Lanjutan

- GrabCut / watershed untuk latar tidak polos.
- Normalisasi warna per perangkat kamera.
- Klasifier ML dengan validasi silang.
- Pengukuran waktu proses dan profiling.
- Deployment ke web (Streamlit) atau aplikasi mobile.
