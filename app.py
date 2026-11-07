"""
Aplikasi Streamlit: Klasifikasi Anomali dan Serangan Jaringan (UNSW-NB15) dengan Random Forest
Proyek 01, Machine Learning for Network Data, D4 TRKJ Universitas Bung Hatta

Jalankan lokal:  streamlit run app.py

PERUBAHAN DARI TEMPLATE:
1. Identitas mahasiswa di judul dan footer (GANTI NAMA & NIM di bawah).
2. Metrik tambahan di sidebar (precision dan F1 serangan, jika ada di metadata).
3. Checkbox filter "hanya tampilkan flow SERANGAN" pada tabel hasil.
4. Grafik baru proporsi Normal vs Serangan.
5. Tabel 10 flow dengan probabilitas serangan tertinggi.
6. Tab baru "Simulasi sensitivitas": precision/recall/F1 per threshold, threshold terbaik, confusion matrix.
"""
import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

# ====== GANTI DENGAN DATA KAMU ======
NAMA = "Berlani Reshalia"
NIM = "2410017514009"
# ====================================

BASE = Path(__file__).parent
MODEL_DIR = BASE / "model"
SAMPLE_CSV = BASE / "data" / "sample_flows.csv"

st.set_page_config(page_title="Deteksi Serangan UNSW-NB15", page_icon="🛡️", layout="wide")


# ---------- Muat model (sekali saja, lalu di-cache) ----------
@st.cache_resource
def load_models():
    meta = json.loads((MODEL_DIR / "metadata.json").read_text())
    rf_bin = joblib.load(MODEL_DIR / "rf_binary.joblib")
    rf_cat = joblib.load(MODEL_DIR / "rf_multiclass.joblib")
    return rf_bin, rf_cat, meta


rf_bin, rf_cat, meta = load_models()
FEATURES = meta["features"]


def predict(df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """Lengkapi kolom yang hilang dengan nilai default, lalu prediksi dengan kedua model."""
    X = df.copy()
    for col in FEATURES:
        if col not in X.columns:
            X[col] = meta["defaults"][col]
    X = X[FEATURES]
    for col in meta["cat_cols"]:
        X[col] = X[col].astype(str)
    prob = rf_bin.predict_proba(X)[:, 1]
    out = pd.DataFrame({
        "prob_serangan": prob.round(3),
        "prediksi": ["SERANGAN" if p >= threshold else "normal" for p in prob],
        "kategori_prediksi": rf_cat.predict(X),
    })
    return out


# ---------- Sidebar ----------
with st.sidebar:
    st.header("⚙️ Pengaturan")
    threshold = st.slider("Threshold probabilitas serangan", 0.05, 0.95, 0.50, 0.05,
                          help="Flow dianggap serangan bila probabilitasnya ≥ threshold. "
                               "Turunkan untuk menangkap lebih banyak serangan (recall naik), "
                               "dengan risiko alarm palsu bertambah.")
    st.divider()
    st.subheader("Performa model (data test)")
    mb, mc = meta["metrics_binary"], meta["metrics_multiclass"]
    c1, c2 = st.columns(2)
    c1.metric("Akurasi biner", f"{mb['accuracy']:.1%}")
    c2.metric("ROC-AUC", f"{mb['roc_auc']:.3f}")
    c1.metric("Recall serangan", f"{mb['recall']:.1%}")
    c2.metric("Akurasi kategori", f"{mc['accuracy']:.1%}")
    # [PERUBAHAN] metrik tambahan: precision dan F1 (hanya tampil jika ada di metadata)
    if "precision" in mb:
        c1.metric("Precision serangan", f"{mb['precision']:.1%}")
    if "f1" in mb:
        c2.metric("F1 serangan", f"{mb['f1']:.1%}")
    st.caption(f"Dataset: {meta['dataset']}")
    st.divider()
    st.caption(f"👤 {NAMA} ({NIM})")

# ---------- Halaman utama ----------
st.title("🛡️ Deteksi Anomali dan Serangan Jaringan")
# [PERUBAHAN] identitas mahasiswa
st.caption(f"Dibuat oleh: {NAMA} - {NIM} | D4 TRKJ Universitas Bung Hatta")
st.write("Model **Random Forest** yang dilatih pada dataset **UNSW-NB15** memprediksi apakah sebuah "
         "flow jaringan normal atau serangan, sekaligus menebak kategori serangannya.")

# [PERUBAHAN] tab baru "Simulasi sensitivitas"
tab_csv, tab_manual, tab_sim, tab_info = st.tabs(
    ["📄 Prediksi dari CSV", "✍️ Input manual", "🎚️ Simulasi sensitivitas", "ℹ️ Tentang model"])

# --- Tab 1: CSV ---
with tab_csv:
    st.write("Unggah file CSV berisi flow dengan kolom seperti dataset UNSW-NB15, "
             "atau pakai 300 flow contoh dari data test.")
    uploaded = st.file_uploader("File CSV", type="csv")
    use_sample = st.toggle("Gunakan data contoh", value=uploaded is None)

    df = None
    if uploaded is not None and not use_sample:
        df = pd.read_csv(uploaded)
    elif use_sample:
        df = pd.read_csv(SAMPLE_CSV)

    if df is not None:
        missing = [c for c in FEATURES if c not in df.columns]
        if missing:
            st.warning(f"{len(missing)} kolom tidak ada dan diisi nilai default (median data latih): "
                       f"{', '.join(missing[:10])}{' ...' if len(missing) > 10 else ''}")
        res = predict(df, threshold)
        shown = pd.concat([res, df.drop(columns=[c for c in res.columns if c in df.columns])], axis=1)

        n_att = int((res["prediksi"] == "SERANGAN").sum())
        k1, k2, k3 = st.columns(3)
        k1.metric("Jumlah flow", len(res))
        k2.metric("Diprediksi serangan", n_att)
        k3.metric("Persentase serangan", f"{n_att / len(res):.1%}")

        if "label" in df.columns:
            acc = ((res["prediksi"] == "SERANGAN").astype(int) == df["label"]).mean()
            st.info(f"File memuat kolom `label`, jadi prediksi bisa dicek: akurasi biner = **{acc:.1%}**")

        left, right = st.columns([1, 1])
        with left:
            st.subheader("Kategori yang diprediksi")
            kat = res["kategori_prediksi"].value_counts().rename_axis("kategori").reset_index(name="jumlah")
            st.bar_chart(kat, x="kategori", y="jumlah", horizontal=True, sort="-jumlah")
        with right:
            st.subheader("Sebaran probabilitas serangan")
            st.bar_chart(pd.cut(res["prob_serangan"], bins=[0, .1, .3, .5, .7, .9, 1.0], include_lowest=True)
                         .astype(str).value_counts().sort_index())

        # [PERUBAHAN] grafik proporsi normal vs serangan
        st.subheader("Proporsi Normal vs Serangan")
        prop = res["prediksi"].value_counts().rename_axis("prediksi").reset_index(name="jumlah")
        st.bar_chart(prop, x="prediksi", y="jumlah")

        # [PERUBAHAN] 10 flow dengan probabilitas serangan tertinggi
        st.subheader("10 flow paling mencurigakan")
        top10 = shown.sort_values("prob_serangan", ascending=False).head(10)
        st.dataframe(top10, width="stretch")

        st.subheader("Hasil per flow")
        # [PERUBAHAN] filter hanya flow yang diprediksi SERANGAN
        hanya_serangan = st.checkbox("Tampilkan hanya flow yang diprediksi SERANGAN")
        if hanya_serangan:
            shown = shown[shown["prediksi"] == "SERANGAN"]
        st.caption(f"Menampilkan {len(shown)} flow")
        st.dataframe(shown, width="stretch", height=360)
        st.download_button("⬇️ Unduh hasil (CSV)", shown.to_csv(index=False).encode(),
                           "hasil_prediksi.csv", "text/csv")

# --- Tab 2: Input manual ---
with tab_manual:
    st.write("Pilih flow contoh sebagai titik awal, lalu ubah fitur-fitur terpenting dan tekan **Prediksi**. "
             "Fitur yang tidak ditampilkan memakai nilai dari flow contoh tersebut.")
    samples = pd.read_csv(SAMPLE_CSV)
    pilihan = st.selectbox("Mulai dari contoh flow kategori", sorted(samples["attack_cat"].unique()),
                           index=sorted(samples["attack_cat"].unique()).index("Normal"))
    base = samples[samples["attack_cat"] == pilihan].iloc[0]
    with st.form("manual"):
        cols = st.columns(3)
        row = {c: base[c] for c in FEATURES}
        for i, c in enumerate(meta["cat_cols"]):
            opts = sorted(set(meta["cat_options"][c]) | {str(base[c])})
            row[c] = cols[i].selectbox(c, opts, index=opts.index(str(base[c])), key=f"{pilihan}-{c}")
        cols = st.columns(4)
        for i, c in enumerate(meta["top_numeric"]):
            lo, hi = meta["num_ranges"][c]
            row[c] = cols[i % 4].number_input(c, min_value=float(lo), value=float(base[c]), key=f"{pilihan}-{c}",
                                              help=f"Rentang umum data latih: {lo:g} sampai {hi:g}")
        submitted = st.form_submit_button("🔍 Prediksi", type="primary")
    if submitted:
        r = predict(pd.DataFrame([row]), threshold).iloc[0]
        if r["prediksi"] == "SERANGAN":
            st.error(f"**SERANGAN** terdeteksi (probabilitas {r['prob_serangan']:.1%}). "
                     f"Kategori paling mungkin: **{r['kategori_prediksi']}**")
        else:
            st.success(f"**Normal** (probabilitas serangan {r['prob_serangan']:.1%})")

# --- Tab baru: Simulasi sensitivitas (pengaruh threshold terhadap performa) ---
with tab_sim:
    st.write("Tab ini menunjukkan bagaimana **threshold** mengubah kualitas deteksi. "
             "Model diuji pada 300 flow contoh yang punya label asli, lalu dihitung precision, recall, "
             "dan F1 untuk setiap threshold.")
    data_sim = pd.read_csv(SAMPLE_CSV)
    if "label" not in data_sim.columns:
        st.info("Data contoh tidak memiliki kolom `label`, jadi simulasi tidak dapat dihitung.")
    else:
        prob_sim = predict(data_sim, 0.5)["prob_serangan"]
        y_true = data_sim["label"].astype(int)

        def hitung_metrik(t: float) -> dict:
            pred = (prob_sim >= t).astype(int)
            tp = int(((pred == 1) & (y_true == 1)).sum())
            fp = int(((pred == 1) & (y_true == 0)).sum())
            fn = int(((pred == 0) & (y_true == 1)).sum())
            tn = int(((pred == 0) & (y_true == 0)).sum())
            prec = tp / (tp + fp) if (tp + fp) else 0.0
            rec = tp / (tp + fn) if (tp + fn) else 0.0
            f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
            return {"threshold": round(t, 2), "akurasi": (tp + tn) / len(y_true),
                    "precision": prec, "recall": rec, "f1": f1,
                    "TP": tp, "FP": fp, "FN": fn, "TN": tn}

        tabel = pd.DataFrame([hitung_metrik(i * 0.05) for i in range(1, 20)])
        terbaik = tabel.loc[tabel["f1"].idxmax()]
        st.success(f"Threshold dengan F1 tertinggi pada data contoh: **{terbaik['threshold']:.2f}** "
                   f"(F1 = {terbaik['f1']:.1%}, recall = {terbaik['recall']:.1%}, "
                   f"precision = {terbaik['precision']:.1%})")

        st.subheader("Precision, recall, dan F1 menurut threshold")
        st.line_chart(tabel.set_index("threshold")[["precision", "recall", "f1"]])

        st.subheader("Detail pada threshold tertentu")
        t_pilih = st.slider("Pilih threshold", 0.05, 0.95, float(threshold), 0.05, key="slider_sim")
        m = hitung_metrik(t_pilih)
        a, b, c, d = st.columns(4)
        a.metric("Akurasi", f"{m['akurasi']:.1%}")
        b.metric("Precision", f"{m['precision']:.1%}")
        c.metric("Recall", f"{m['recall']:.1%}")
        d.metric("F1", f"{m['f1']:.1%}")
        st.write("**Confusion matrix**")
        st.table(pd.DataFrame(
            {"Prediksi normal": [m["TN"], m["FN"]], "Prediksi serangan": [m["FP"], m["TP"]]},
            index=["Aktual normal", "Aktual serangan"]))
        st.caption("FN (serangan lolos) menurun saat threshold diturunkan, tetapi FP (alarm palsu) bertambah.")

        with st.expander("Lihat tabel lengkap semua threshold"):
            st.dataframe(tabel.round(3), width="stretch")

# --- Tab 4: Info model ---
with tab_info:
    st.subheader("Fitur terpenting (model biner)")
    imp = pd.Series(meta["importance_top15"]).rename_axis("fitur").reset_index(name="importance")
    st.bar_chart(imp, x="fitur", y="importance", horizontal=True, sort="-importance")
    st.subheader("Metrik pada data test")
    st.table(pd.DataFrame({
        "Model biner": {k: round(v, 4) for k, v in meta["metrics_binary"].items()},
    }))
    st.table(pd.DataFrame({
        "Model multikelas": {k: round(v, 4) for k, v in meta["metrics_multiclass"].items()},
    }))
    st.caption(f"Dilatih dengan Python {meta['versions']['python']}, "
               f"scikit-learn {meta['versions']['scikit-learn']}, numpy {meta['versions']['numpy']}.")

# [PERUBAHAN] footer
st.divider()
st.caption(f"© 2026 {NAMA} ({NIM}) - Proyek 01 Machine Learning for Network Data, D4 TRKJ Universitas Bung Hatta")
