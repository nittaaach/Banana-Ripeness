"""Streamlit Web Application: Interactive Banana Ripeness Inspection and Pipeline Demo."""

from pathlib import Path
import sys
import cv2
import numpy as np
import pandas as pd
import streamlit as st

# Ensure src is in python path
src_dir = Path(__file__).resolve().parent.parent / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from banana_ripeness.config import Config, load_config
from banana_ripeness.evaluation import evaluate_folder
from banana_ripeness.io_utils import CANONICAL_CLASSES, list_images, parse_filename_metadata, read_image
from banana_ripeness.pipeline import BananaRipenessPipeline
from banana_ripeness.visualization import stage_panel

# Set page configuration
st.set_page_config(
    page_title="Sistem Deteksi Kematangan Pisang",
    page_icon="🍌",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling
st.markdown(
    """
    <style>
    .main-header { font-size: 2.2rem; font-weight: 700; color: #1E293B; margin-bottom: 0.2rem; }
    .sub-header { font-size: 1.1rem; color: #64748B; margin-bottom: 1.5rem; }
    .badge-mentah { background-color: #22c55e; color: white; padding: 6px 16px; border-radius: 9999px; font-weight: 600; }
    .badge-matang { background-color: #eab308; color: black; padding: 6px 16px; border-radius: 9999px; font-weight: 600; }
    .badge-terlalu-matang { background-color: #f97316; color: white; padding: 6px 16px; border-radius: 9999px; font-weight: 600; }
    .badge-busuk { background-color: #ef4444; color: white; padding: 6px 16px; border-radius: 9999px; font-weight: 600; }
    .metric-card { background: #f8fafc; border-radius: 8px; padding: 15px; border-left: 4px solid #3b82f6; }
    </style>
    """,
    unsafe_allow_html=True,
)


def get_badge_html(label: str) -> str:
    slug = label.lower().replace(" ", "-")
    cls_map = {
        "mentah": "badge-mentah",
        "matang": "badge-matang",
        "terlalu-matang": "badge-terlalu-matang",
        "busuk": "badge-busuk",
    }
    css_class = cls_map.get(slug, "badge-matang")
    return f'<span class="{css_class}">{label.upper()}</span>'


# Load default config
base_cfg = load_config()

# Sidebar: Controls & Parameter Tuning
st.sidebar.title("⚙️ Parameter Pipeline")

with st.sidebar.expander("1. Preprocessing", expanded=False):
    contrast_stretch = st.checkbox("Contrast Stretching (Kanal V)", value=base_cfg.preprocessing.contrast_stretch)
    blur_k = st.slider("Ukuran Gaussian Kernel", min_value=3, max_value=11, value=5, step=2)

with st.sidebar.expander("2. Segmentasi Hibrid", expanded=True):
    h_low = st.slider("HUE Min (Lower)", 0, 180, base_cfg.segmentation.hsv_lower[0])
    h_high = st.slider("HUE Max (Upper)", 0, 180, base_cfg.segmentation.hsv_upper[0])
    s_low = st.slider("SAT Min (Lower)", 0, 255, base_cfg.segmentation.hsv_lower[1])
    v_low = st.slider("VAL Min (Lower)", 0, 255, base_cfg.segmentation.hsv_lower[2])
    otsu_ch = st.selectbox("Kanal untuk Otsu", ["S", "V"], index=0 if base_cfg.segmentation.otsu_channel == "S" else 1)
    combine_mode = st.radio("Metode Gabung Masker", ["or", "and"], index=0 if base_cfg.segmentation.combine_mode == "or" else 1)

with st.sidebar.expander("3. Morfologi", expanded=False):
    close_iters = st.slider("Iterasi Closing", 1, 5, base_cfg.morphology.close_iterations)
    open_iters = st.slider("Iterasi Opening", 0, 3, base_cfg.morphology.open_iterations)
    keep_largest = st.checkbox("Ambil Kontur Terbesar", value=base_cfg.morphology.keep_largest_component)
    fill_holes_opt = st.checkbox("Isi Lubang (Fill Holes)", value=base_cfg.morphology.fill_holes)

with st.sidebar.expander("4. Ambang Klasifikasi", expanded=False):
    th_green = st.slider("Min Rasio Hijau (Mentah)", 0.1, 0.9, base_cfg.classification.green_ratio_unripe, 0.05)
    th_overripe = st.slider("Min Rasio Coklat (Terlalu Matang)", 0.05, 0.4, base_cfg.classification.brown_ratio_overripe, 0.02)
    th_rotten = st.slider("Min Rasio Coklat (Busuk)", 0.2, 0.8, base_cfg.classification.brown_ratio_rotten, 0.02)

# Apply dynamic configuration
active_cfg = Config()
active_cfg.preprocessing.contrast_stretch = contrast_stretch
active_cfg.preprocessing.gaussian_kernel = (blur_k, blur_k)
active_cfg.segmentation.hsv_lower = [h_low, s_low, v_low]
active_cfg.segmentation.hsv_upper = [h_high, 255, 255]
active_cfg.segmentation.otsu_channel = otsu_ch
active_cfg.segmentation.combine_mode = combine_mode
active_cfg.morphology.close_iterations = close_iters
active_cfg.morphology.open_iterations = open_iters
active_cfg.morphology.keep_largest_component = keep_largest
active_cfg.morphology.fill_holes = fill_holes_opt
active_cfg.classification.green_ratio_unripe = th_green
active_cfg.classification.brown_ratio_overripe = th_overripe
active_cfg.classification.brown_ratio_rotten = th_rotten

pipeline = BananaRipenessPipeline(cfg=active_cfg)

# Main Page Tabs
st.markdown('<div class="main-header">🍌 Sistem Deteksi Tingkat Kematangan Pisang</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Pipeline Hibrid: Preprocessing + HSV + Otsu + Morfologi</div>', unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["🔍 Uji Citra Tunggal", "📊 Evaluasi Batch & Metrik", "📖 Tentang Arsitektur"])

with tab1:
    col_input, col_result = st.columns([1, 1])

    with col_input:
        st.subheader("Pilih atau Unggah Citra")
        input_choice = st.radio("Sumber Citra:", ["Pilih dari Dataset Uji (16 Citra)", "Unggah Citra Sendiri"], horizontal=True)

        selected_img_bgr = None
        current_filename = "custom_image.jpg"
        meta = {}

        if input_choice == "Pilih dari Dataset Uji (16 Citra)":
            raw_dir = Path("Data/Raw").resolve()
            img_files = list_images(raw_dir)
            if img_files:
                file_names = [f.name for f in img_files]
                chosen_name = st.selectbox("Daftar Citra Spesimen Uji:", file_names)
                chosen_path = raw_dir / chosen_name
                selected_img_bgr = read_image(chosen_path)
                current_filename = chosen_name
                meta = parse_filename_metadata(chosen_path)
            else:
                st.warning("Folder Data/Raw kosong.")
        else:
            uploaded_file = st.file_uploader("Unggah foto pisang (JPG/PNG)", type=["jpg", "jpeg", "png"])
            if uploaded_file is not None:
                file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
                selected_img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                current_filename = uploaded_file.name
                meta = parse_filename_metadata(uploaded_file.name)

    if selected_img_bgr is not None:
        # Run inference
        res = pipeline.run(selected_img_bgr)

        with col_result:
            st.subheader("Hasil Diagnosis Kematangan")
            badge = get_badge_html(res.label)
            st.markdown(f"### Status: {badge}", unsafe_allow_html=True)

            if meta.get("label"):
                match_icon = "✅ Sesuai" if meta["label"] == res.label else "❌ Berbeda"
                st.caption(f"Ground Truth dari nama berkas: **{meta['label']}** ({match_icon})")

            st.write(f"**Aturan Pemicu:** `{res.decision_info.get('rule')}`")

            # Metrics row
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Rasio Hijau", f"{res.features['ratio_green']*100:.1f}%")
            m2.metric("Rasio Kuning", f"{res.features['ratio_yellow']*100:.1f}%")
            m3.metric("Rasio Coklat", f"{res.features['ratio_brown']*100:.1f}%")
            m4.metric("Luas Pisang", f"{res.features['area_ratio']*100:.1f}%")

            # Distribution chart
            dist_data = pd.DataFrame({
                "Spektrum Warna": ["Hijau (Mentah)", "Kuning (Matang)", "Coklat/Bercak (Busuk)"],
                "Persentase": [
                    res.features["ratio_green"] * 100,
                    res.features["ratio_yellow"] * 100,
                    res.features["ratio_brown"] * 100,
                ],
            })
            st.bar_chart(dist_data.set_index("Spektrum Warna"))

        st.divider()
        st.subheader("Visualisasi 6 Tahapan Pipeline")
        grid_cols = st.columns(3)
        grid_cols[0].image(cv2.cvtColor(res.stages["raw"], cv2.COLOR_BGR2RGB), caption="1. Citra Asli (600x600)", use_container_width=True)
        grid_cols[1].image(cv2.cvtColor(res.stages["preprocessed"], cv2.COLOR_BGR2RGB), caption="2. Preprocessed (Blur + Kontras)", use_container_width=True)
        grid_cols[2].image(res.stages["mask_hsv"], caption="3. Masker HSV (Hijau-Kuning-Coklat)", use_container_width=True)

        grid_cols2 = st.columns(3)
        grid_cols2[0].image(res.stages["mask_otsu"], caption="4. Masker Otsu (Auto-polaritas)", use_container_width=True)
        grid_cols2[1].image(res.stages["final"], caption="5. Morfologi (Opening/Closing & Kontur)", use_container_width=True)
        grid_cols2[2].image(cv2.cvtColor(res.stages["overlay"], cv2.COLOR_BGR2RGB), caption="6. Overlay & Segmentasi Akhir", use_container_width=True)

with tab2:
    st.subheader("Evaluasi Seluruh Dataset Uji terhadap Ground Truth")
    if st.button("🚀 Jalankan Evaluasi Batch Sekarang", type="primary"):
        with st.spinner("Memproses seluruh dataset dan menghitung metrik IoU..."):
            raw_dir = Path("Data/Raw").resolve()
            gt_dir = Path("Data/ground_truth").resolve()

            if not gt_dir.exists() or len(list_images(gt_dir)) == 0:
                st.info("Masker Ground Truth belum ada. Menghasilkan baseline ground truth...")
                from scripts.create_ground_truth import generate_precise_mask
                from banana_ripeness.io_utils import write_image
                gt_dir.mkdir(parents=True, exist_ok=True)
                for r_img in list_images(raw_dir):
                    g_img = read_image(r_img)
                    g_mask = generate_precise_mask(g_img)
                    write_image(gt_dir / f"{r_img.stem}.png", g_mask)

            df, summary = evaluate_folder(raw_dir, gt_dir, pipeline=pipeline)

            st.success("Evaluasi Selesai!")

            k1, k2, k3, k4 = st.columns(4)
            k1.metric("Rata-rata IoU", f"{summary['mean_iou']:.4f}", help="Target PRD >= 0.85 (Acuan: 0.934)")
            k2.metric("Keberhasilan Segmentasi", f"{summary['success_rate']:.1f}%", help="IoU >= 0.80, Target PRD >= 90%")
            k3.metric("Akurasi Klasifikasi", f"{summary['accuracy']:.1f}%", help="Target PRD >= 85%")
            k4.metric("Rata-rata Dice", f"{summary['mean_dice']:.4f}")

            st.write("### Rincian Per Citra")
            st.dataframe(df[["filename", "true_label", "pred_label", "iou", "dice", "label_match", "ratio_green", "ratio_yellow", "ratio_brown"]], use_container_width=True)

            st.write("### Rekapitulasi Per Kondisi")
            cond_records = []
            for c_name, c_s in summary.get("condition_stats", {}).items():
                cond_records.append({
                    "Kondisi": c_name,
                    "Jumlah": c_s["count"],
                    "Akurasi (%)": round(c_s["accuracy"] * 100, 1),
                    "Mean IoU": round(c_s["mean_iou"], 4) if c_s["mean_iou"] is not None else None,
                    "Success Rate (%)": round(c_s["success_rate"] * 100, 1) if c_s["success_rate"] is not None else None,
                })
            st.table(pd.DataFrame(cond_records))

with tab3:
    st.subheader("Arsitektur & Spesifikasi Pipeline Hibrid")
    st.markdown(
        """
        Sistem ini mengimplementasikan kombinasi terbaik dari benchmark 22 metode pengolahan citra:
        1. **Preprocessing**: Standarisasi ke 600×600 px, perataan kontras persentil pada kanal V, dan Gaussian Blur 5×5 untuk meredam tekstur noise.
        2. **HSV inRange**: Mengisolasi spektrum warna pisang (hijau, kuning, coklat).
        3. **Otsu Thresholding**: Segmentasi otomatis berbasis variansi intensitas intra-kelas dengan deteksi pembalikan polaritas latar belakang.
        4. **Operasi Morfologi**:
           - **Opening** (5×5 elips): Menghilangkan bintik noise di latar belakang.
           - **Closing** (5×5 elips): Menutup lubang akibat bercak gelap/busuk pada pisang.
           - **Kontur Terbesar**: Mengeliminasi objek luar selain pisang utama.
           - **Fill Holes**: Menutup lubang internal.
        5. **Ekstraksi Fitur & Klasifikasi**: Ekstraksi rasio warna berbobot untuk menentukan fase kematangan (*Mentah, Matang, Terlalu Matang, Busuk*).
        """
    )
