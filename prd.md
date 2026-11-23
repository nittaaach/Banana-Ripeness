# PRD — Sistem Deteksi Tingkat Kematangan Pisang Berbasis Pengolahan Citra

| Item | Keterangan |
|---|---|
| Versi | 1.0 (draft) |
| Tanggal | 20 November 2026 |
| Metode inti | Hibrid: Preprocessing (Gaussian) + HSV + Otsu + Morfologi (Opening/Closing) |
| Bahasa / Library | Python 3.10+, OpenCV, NumPy, scikit-image, (opsional) Streamlit |

---

## 1. Latar Belakang

Penentuan kematangan pisang selama ini dilakukan secara visual dan subjektif. Pada tugas Pengolahan Citra Digital, dilakukan benchmark 22 metode (12 tunggal dan 10 campuran) pada 16 citra uji (600×600 px) yang mewakili 4 fase: Mentah (H1), Matang (H11), Terlalu Matang (H15), Busuk (H17).

Hasil benchmark menunjukkan metode campuran **Preprocessing + HSV + Otsu + Morfologi** paling unggul:

| Metrik | Otsu tunggal (terbaik tunggal) | Hibrid terbaik |
|---|---|---|
| Keberhasilan keseluruhan | 86,0% | **96,9%** |
| Rata-rata IoU | 0,818 | **0,934** |
| Mentah / Matang / T. Matang / Busuk | 87,5 / 93,8 / 84,4 / 78,3 | 97,5 / 100,0 / 96,5 / 93,6 |

PRD ini mendefinisikan produk perangkat lunak yang mengimplementasikan metode tersebut menjadi sistem yang dapat dijalankan.

> **Catatan penting:** angka di benchmark mengukur keberhasilan **segmentasi** objek pisang (IoU terhadap masker referensi). Untuk menghasilkan label kematangan, sistem membutuhkan tahap tambahan: **ekstraksi fitur warna + klasifikasi** setelah segmentasi. Tahap ini dijabarkan di bawah.

## 2. Tujuan

1. Mengimplementasikan pipeline hibrid agar dapat memisahkan pisang dari latar belakang secara otomatis.
2. Mengklasifikasikan kematangan ke 4 kelas berdasarkan warna kulit: **Mentah, Matang, Terlalu Matang, Busuk**.
3. Mereproduksi (atau mendekati) hasil benchmark: keberhasilan ≥ 90% dan IoU ≥ 0,85 pada dataset uji.
4. Menyediakan antarmuka CLI (wajib) dan antarmuka web sederhana (opsional) untuk demo.

## 3. Non-Tujuan (Out of Scope)

- Deteksi banyak pisang sekaligus dengan instance segmentation (deep learning).
- Pelatihan model CNN/deep learning.
- Prediksi umur simpan (shelf-life) dalam hari.
- Deployment mobile/embedded.

## 4. Pengguna & Skenario

| Pengguna | Kebutuhan |
|---|---|
| Mahasiswa / peneliti | Menjalankan pipeline pada dataset, mengevaluasi IoU & akurasi, mengekspor tabel hasil |
| Dosen / penguji | Melihat demo satu citra beserta visualisasi tiap tahap |
| (Opsional) Pelaku usaha kecil | Mengunggah foto pisang dan mendapat label kematangan |

Skenario utama: pengguna memberi satu citra pisang, sistem mengembalikan label kematangan, masker segmentasi, dan gambar overlay.

## 5. Kebutuhan Fungsional

| ID | Kebutuhan | Prioritas |
|---|---|---|
| FR-01 | Membaca citra (JPG/PNG), resize ke 600×600 px | Must |
| FR-02 | Preprocessing: Gaussian blur (kernel 5×5), opsional contrast stretching | Must |
| FR-03 | Konversi BGR → HSV dan pembuatan masker warna (rentang hijau–kuning–coklat) | Must |
| FR-04 | Otsu thresholding otomatis pada kanal Saturation (atau V) | Must |
| FR-05 | Penggabungan masker HSV dan Otsu (operasi AND/OR terkonfigurasi) | Must |
| FR-06 | Morfologi Opening lalu Closing (kernel 5×5, elips) | Must |
| FR-07 | Pengambilan kontur terbesar sebagai objek pisang, pengisian lubang | Must |
| FR-08 | Ekstraksi fitur warna dari area masker: rata-rata H/S/V, rasio piksel hijau/kuning/coklat | Must |
| FR-09 | Klasifikasi 4 kelas kematangan (berbasis aturan, dapat dikalibrasi dari data) | Must |
| FR-10 | Mode batch: memproses folder citra dan menyimpan CSV hasil | Must |
| FR-11 | Evaluasi: hitung IoU (dan Dice) terhadap ground-truth mask, akurasi klasifikasi, confusion matrix | Must |
| FR-12 | Visualisasi tiap tahap (asli, blur, HSV, Otsu, morfologi, overlay) | Should |
| FR-13 | Konfigurasi parameter melalui file YAML tanpa mengubah kode | Should |
| FR-14 | Antarmuka web (Streamlit) untuk unggah citra | Could |

## 6. Kebutuhan Non-Fungsional

| ID | Kebutuhan | Target |
|---|---|---|
| NFR-01 | Waktu proses per citra 600×600 | < 1 detik pada laptop biasa (CPU) |
| NFR-02 | Reprodusibilitas | Hasil deterministik; versi library dikunci di `requirements.txt` |
| NFR-03 | Keterbacaan kode | Modular, type hints, docstring, PEP8 |
| NFR-04 | Pengujian | Unit test untuk tiap modul inti, cakupan ≥ 70% |
| NFR-05 | Portabilitas | Berjalan di Windows/Linux/macOS tanpa GPU |

## 7. Data

- **Input**: citra pisang tampak samping, 600×600 px, latar relatif polos, pencahayaan seragam.
- **Dataset acuan**: 16 citra (4 kondisi × 4 citra; perangkat Realme 11/12/14; sisi Kanan & Kiri).
- **Ground truth**: masker biner objek pisang per citra (dibuat manual, misalnya dengan LabelMe/GIMP) untuk perhitungan IoU.
- **Label kelas**: Mentah, Matang, Terlalu Matang, Busuk (dari nama file / metadata hari).

Struktur folder data:

```
data/
├── raw/            # citra asli
├── ground_truth/   # masker biner (nama file sama dengan citra)
└── output/         # hasil: mask, overlay, results.csv
```

## 8. Metrik Keberhasilan

| Metrik | Definisi | Target |
|---|---|---|
| IoU | \|A ∩ B\| / \|A ∪ B\| antara masker prediksi dan ground truth | rata-rata ≥ 0,85 (acuan benchmark 0,934) |
| Keberhasilan segmentasi | Proporsi citra dengan IoU ≥ ambang (ditetapkan, mis. 0,80) | ≥ 90% (acuan 96,9%) |
| Akurasi klasifikasi | Prediksi kelas = label sebenarnya | ≥ 85% |
| Waktu proses | Rata-rata per citra | < 1 detik |

Ambang "berhasil" pada benchmark harus didefinisikan eksplisit di laporan agar angka 96,9% dapat direproduksi.

## 9. Asumsi & Risiko

| Risiko | Dampak | Mitigasi |
|---|---|---|
| Pisang Busuk gelap, saturasi rendah → segmentasi bolong | IoU turun | Closing morfologi, fallback ke kanal V, kalibrasi rentang HSV |
| Latar tidak polos / bayangan | Piksel latar ikut terdeteksi | Syarat pengambilan foto; kontur terbesar; opsional GrabCut |
| Perbedaan kamera (Realme 11/12/14) | Pergeseran warna | Normalisasi kontras, kalibrasi ambang per perangkat |
| Dataset kecil (16 citra) | Aturan klasifikasi overfit | Gunakan validasi silang, tambah data, laporkan keterbatasan |
| Angka benchmark belum tentu identik saat diimplementasi | Selisih hasil | Verifikasi ulang dengan menjalankan kode pada dataset yang sama |

## 10. Milestone

| Tahap | Deliverable | Estimasi |
|---|---|---|
| M1 | Setup proyek, struktur folder, loader citra | 0,5 hari |
| M2 | Preprocessing + HSV + Otsu + Morfologi (segmentasi jalan) | 1–2 hari |
| M3 | Ekstraksi fitur + klasifikasi aturan | 1 hari |
| M4 | Modul evaluasi (IoU, akurasi, confusion matrix), batch CSV | 1 hari |
| M5 | Unit test, visualisasi, tuning parameter | 1–2 hari |
| M6 | (Opsional) UI Streamlit, dokumentasi, laporan | 1 hari |

## 11. Kriteria Penerimaan

- [ ] `python scripts/run_single.py <citra>` menghasilkan label dan overlay.
- [ ] `python scripts/run_batch.py` memproses 16 citra dan menulis `results.csv`.
- [ ] `python scripts/evaluate.py` menampilkan IoU rata-rata, akurasi, dan confusion matrix.
- [ ] Semua unit test lulus.
- [ ] Parameter dapat diubah lewat `config/config.yaml`.
