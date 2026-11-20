import os
import tempfile

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from sklearn.metrics import r2_score
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.models import Sequential

# ---------------------------------------------------------
# STYLING & CONFIG
# ---------------------------------------------------------
st.set_page_config(
    page_title="Dashboard Neural Networks - Henri Ilham",
    page_icon="🧠",
    layout="wide",
)

# Custom CSS untuk mempercantik kartu identitas & UI
st.markdown(
    """
    <style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0px;
    }
    .sub-title {
        font-size: 1.1rem;
        color: #4B5563;
        margin-bottom: 20px;
    }
    .profile-card {
        background-color: #F0F9FF;
        padding: 15px;
        border-radius: 10px;
        border-left: 5px solid #0284C7;
        margin-bottom: 20px;
    }
    </style>
""",
    unsafe_allow_html=True,
)

# ---------------------------------------------------------
# SIDEBAR (Identitas & Navigasi)
# ---------------------------------------------------------
st.sidebar.image(
    "https://cdn-icons-png.flaticon.com/512/4712/4712109.png", width=80
)
st.sidebar.title("Dashboard Tugas NN")

# Identitas Diri
st.sidebar.markdown(
    """
<div class="profile-card">
    <small><b>Penyusun Tugas:</b></small><br>
    <b style="color: #0369A1; font-size: 1.05rem;">Henri Ilham</b><br>
    <small>Mata Kuliah: Neural Networks</small><br>
    <small>Status: 🟢 Active Deployment</small>
</div>
""",
    unsafe_allow_html=True,
)

st.sidebar.header("📌 Navigasi Aplikasi")
pilihan_app = st.sidebar.radio(
    "Pilih Modul Pembelajaran:",
    [
        "Aplikasi 1: Regresi (Prediksi Tarif)",
        "Aplikasi 2: Klasifikasi Biner",
        "Aplikasi 3: Klasifikasi Multikelas & Callbacks",
    ],
)

st.sidebar.divider()
st.sidebar.caption("© 2026 Henri Ilham • Streamlit Deployment")

# ---------------------------------------------------------
# HEADER UTAMA
# ---------------------------------------------------------
st.markdown(
    '<div class="main-title">🧠 Dashboard Neural Networks (Keras & TensorFlow)</div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="sub-title">Simulasi & Visualisasi Model Deep Learning | Disusun oleh <b>Henri Ilham Caniago</b></div>',
    unsafe_allow_html=True,
)
st.divider()

# ---------------------------------------------------------
# APLIKASI 1: REGRESI
# ---------------------------------------------------------
if pilihan_app == "Aplikasi 1: Regresi (Prediksi Tarif)":
    st.subheader("1. Aplikasi Regresi - Prediksi Nilai Kontinu")
    st.write(
        "Model regresi memprediksi variabel kontinu (seperti tarif taksi) tanpa menggunakan fungsi aktivasi pada layer output."
    )

    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("### ⚙️ Parameter Input Simulasi")
        f1 = st.slider("Jarak Tempuh (km)", 1.0, 50.0, 10.0)
        f2 = st.slider("Waktu Tempuh (menit)", 5.0, 120.0, 25.0)
        f3 = st.slider("Tingkat Kemacetan (1-5)", 1.0, 5.0, 2.0)

        run_btn = st.button("🚀 Train Model & Prediksi", type="primary")

    with col2:
        st.markdown("### 💻 Arsitektur Model")
        kode_regresi = """
model = Sequential([
    Dense(512, activation='relu', input_dim=3),
    Dense(512, activation='relu'),
    Dense(1) # Output tanpa aktivasi (Regresi)
])
model.compile(optimizer='adam', loss='mae')
        """
        st.code(kode_regresi, language="python")

    if run_btn:
        with st.spinner("Melatih Neural Network di latar belakang..."):
            X_train = np.random.rand(1000, 3) * [50, 120, 5]
            y_train = (
                X_train[:, 0] * 3.5 + X_train[:, 1] * 1.2 + X_train[:, 2] * 4.0
            )

            model = Sequential(
                [
                    Dense(512, activation="relu", input_dim=3),
                    Dense(512, activation="relu"),
                    Dense(1),
                ]
            )
            model.compile(optimizer="adam", loss="mae", metrics=["mae"])
            history = model.fit(
                X_train, y_train, epochs=20, batch_size=32, verbose=0
            )

            sample_input = np.array([[f1, f2, f3]])
            pred_val = model.predict(sample_input)[0][0]
            y_pred = model.predict(X_train)
            r2 = r2_score(y_train, y_pred)

        st.success("✅ Pelatihan Model Selesai!")
        res_col1, res_col2 = st.columns(2)
        res_col1.metric("Hasil Prediksi Estimasi Tarif", f"Rp {pred_val*1000:,.0f}")
        res_col2.metric("Akurasi Model (R² Score)", f"{r2:.4f}")

        # Visualisasi Training Loss
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.plot(history.history["loss"], color="#0284C7", linewidth=2)
        ax.set_title("Grafik Konvergensi Loss (MAE) per Epoch")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        st.pyplot(fig)

# ---------------------------------------------------------
# APLIKASI 2: KLASIFIKASI BINER
# ---------------------------------------------------------
elif pilihan_app == "Aplikasi 2: Klasifikasi Biner":
    st.subheader("2. Aplikasi Klasifikasi Biner")
    st.write(
        "Memisahkan data ke dalam 2 kategori pilihan menggunakan fungsi aktivasi Sigmoid pada layer output."
    )
    st.caption(
        "Fitur 1 dan Fitur 2 adalah koordinat sebuah titik (x, y). Data latih disimulasikan: "
        "1000 titik acak, dan titik dinyatakan Kelas 1 bila berada di dalam lingkaran "
        "(x² + y² < 0,5), selain itu Kelas 0. Model tidak diberi rumus ini, ia mempelajarinya dari contoh."
    )

    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("### ⚙️ Parameter Input")
        val1 = st.number_input("Fitur 1 (Range -1.0 s/d 1.0)", -1.0, 1.0, -0.3)
        val2 = st.number_input("Fitur 2 (Range -1.0 s/d 1.0)", -1.0, 1.0, 0.4)
        run_btn2 = st.button("🚀 Jalankan Klasifikasi Biner", type="primary")

    with col2:
        st.markdown("### 💻 Arsitektur Model")
        kode_biner = """
model = Sequential([
    Dense(128, activation='relu', input_dim=2),
    Dense(1, activation='sigmoid') # Sigmoid untuk 0/1
])
model.compile(optimizer='adam', loss='binary_crossentropy')
        """
        st.code(kode_biner, language="python")

    if run_btn2:
        with st.spinner("Proses klasifikasi berlangsung..."):
            X_train = np.random.uniform(-1, 1, (1000, 2))
            y_train = ((X_train[:, 0] ** 2 + X_train[:, 1] ** 2) < 0.5).astype(
                int
            )

            model = Sequential(
                [
                    Dense(128, activation="relu", input_dim=2),
                    Dense(1, activation="sigmoid"),
                ]
            )
            model.compile(
                optimizer="adam",
                loss="binary_crossentropy",
                metrics=["accuracy"],
            )
            model.fit(X_train, y_train, epochs=15, batch_size=16, verbose=0)

            sample = np.array([[val1, val2]])
            prob = model.predict(sample)[0][0]
            pred_class = 1 if prob > 0.5 else 0

        st.success("✅ Klasifikasi Selesai!")
        st.info(
            f"Probabilitas Positif: **{prob*100:.2f}%** ➔ Hasil Kategori: **Kelas {pred_class}**"
        )
        # Pembanding: jawaban benar menurut rumus yang dipakai membuat data latih
        jawaban_benar = int((val1**2 + val2**2) < 0.5)
        if jawaban_benar == pred_class:
            st.success(
                f"Jawaban benar menurut rumus: **Kelas {jawaban_benar}** "
                f"(x² + y² = {val1**2 + val2**2:.3f}). Tebakan model sesuai."
            )
        else:
            st.warning(
                f"Jawaban benar menurut rumus: **Kelas {jawaban_benar}** "
                f"(x² + y² = {val1**2 + val2**2:.3f}). Tebakan model meleset; "
                "titik ini dekat batas lingkaran, atau model belum cukup belajar."
            )

# ---------------------------------------------------------
# APLIKASI 3: KLASIFIKASI MULTIKELAS
# ---------------------------------------------------------
elif pilihan_app == "Aplikasi 3: Klasifikasi Multikelas & Callbacks":
    st.subheader("3. Aplikasi Klasifikasi Multikelas & Callbacks")
    st.write(
        "Memilah data ke dalam 4 kategori ($N > 2$) menggunakan aktivasi Softmax, "
        "regularisasi Dropout, dan **Callbacks** (EarlyStopping & ModelCheckpoint) "
        "untuk mengontrol proses training."
    )
    st.caption(
        "Data simulasi: titik (fitur 1, fitur 2) di rentang 0-1 dibagi menjadi 4 kuadran "
        "(Kelas 0-3), sehingga model benar-benar bisa mempelajari polanya. "
        "Fitur 1 dan Fitur 2 adalah koordinat titik (x1, x2); kelas = (x1 > 0,5) x 2 + (x2 > 0,5). "
        "Dibuat 2000 titik latih acak."
    )

    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("### ⚙️ Parameter Training & Input")
        max_epochs = st.slider("Maksimum Epochs", 10, 200, 100, step=10)
        patience = st.slider("Patience EarlyStopping (epoch)", 1, 15, 5)
        in1 = st.slider("Fitur 1 (0-1)", 0.0, 1.0, 0.2)
        in2 = st.slider("Fitur 2 (0-1)", 0.0, 1.0, 0.8)
        run_btn3 = st.button("🚀 Jalankan Simulasi Multikelas", type="primary")

    with col2:
        st.markdown("### 💻 Arsitektur & Callbacks")
        kode_multi = """
model = Sequential([
    Dense(64, activation='relu', input_dim=2),
    Dropout(0.2),
    Dense(4, activation='softmax')
])
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy')

callbacks = [
    EarlyStopping(monitor='val_loss', patience=5,
                  min_delta=0.01, restore_best_weights=True),
    ModelCheckpoint('model_terbaik.keras',
                    monitor='val_loss', save_best_only=True),
]
model.fit(X, y, epochs=100, validation_split=0.2,
          callbacks=callbacks)
        """
        st.code(kode_multi, language="python")

    if run_btn3:
        with st.spinner("Melatih Model Multikelas dengan Callbacks..."):
            X_train = np.random.rand(2000, 2)
            # 4 kuadran -> kelas 0..3
            y_train = (X_train[:, 0] > 0.5).astype(int) * 2 + (
                X_train[:, 1] > 0.5
            ).astype(int)

            model = Sequential(
                [
                    Dense(64, activation="relu", input_dim=2),
                    Dropout(0.2),
                    Dense(4, activation="softmax"),
                ]
            )
            model.compile(
                optimizer="adam",
                loss="sparse_categorical_crossentropy",
                metrics=["accuracy"],
            )

            ckpt_path = os.path.join(tempfile.gettempdir(), "model_terbaik.keras")
            early_stop = EarlyStopping(
                monitor="val_loss",
                patience=patience,
                min_delta=0.01,
                restore_best_weights=True,
            )
            callbacks = [
                early_stop,
                ModelCheckpoint(ckpt_path, monitor="val_loss", save_best_only=True),
            ]
            history = model.fit(
                X_train,
                y_train,
                epochs=max_epochs,
                batch_size=32,
                validation_split=0.2,
                callbacks=callbacks,
                verbose=0,
            )

            sample = np.array([[in1, in2]])
            probs = model.predict(sample, verbose=0)[0]
            pred_class = int(np.argmax(probs))

        epochs_run = len(history.history["loss"])
        # epoch terbaik menurut EarlyStopping (perbaikan minimal min_delta)
        best_epoch = int(getattr(early_stop, "best_epoch", 0)) + 1
        stopped_early = epochs_run < max_epochs

        st.success("✅ Simulasi Selesai!")
        if stopped_early:
            st.info(
                f"⏹️ **EarlyStopping aktif:** training berhenti di epoch **{epochs_run}** "
                f"dari maksimum {max_epochs}, karena val_loss tidak membaik "
                f"(lebih dari 0.01) selama {patience} epoch berturut-turut. "
                f"Bobot terbaik dari epoch {best_epoch} dikembalikan."
            )
        else:
            st.warning(
                f"EarlyStopping tidak sempat berhenti: seluruh {max_epochs} epoch dijalankan "
                "(val_loss masih terus membaik). Coba perkecil Patience atau perbesar Maksimum Epochs."
            )

        m1, m2, m3 = st.columns(3)
        m1.metric("Epoch Dijalankan", f"{epochs_run} / {max_epochs}")
        m2.metric("Epoch Terbaik (val_loss)", best_epoch)
        m3.metric("Akurasi Validasi", f"{history.history['val_accuracy'][-1]:.3f}")

        st.write(f"Kategori Prediksi Tertinggi: **Kelas {pred_class}**")
        # Pembanding: jawaban benar menurut rumus kuadran
        jawaban_benar = int(in1 > 0.5) * 2 + int(in2 > 0.5)
        if jawaban_benar == pred_class:
            st.success(f"Jawaban benar menurut rumus kuadran: **Kelas {jawaban_benar}**. Tebakan model sesuai.")
        else:
            st.warning(
                f"Jawaban benar menurut rumus kuadran: **Kelas {jawaban_benar}**. "
                "Tebakan model meleset; titik ini dekat batas kuadran (0,5), atau model belum cukup belajar."
            )
        st.caption(
            "ModelCheckpoint menyimpan model dengan val_loss terbaik; "
            "restore_best_weights mengembalikan bobot terbaik setelah training berhenti."
        )

        g1, g2 = st.columns(2)
        with g1:
            fig, ax = plt.subplots(figsize=(6, 3))
            ax.plot(history.history["loss"], label="loss (training)", linewidth=2)
            ax.plot(history.history["val_loss"], label="val_loss (validasi)", linewidth=2)
            ax.axvline(best_epoch - 1, color="gray", linestyle="--", label="epoch terbaik")
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Loss")
            ax.set_title("Training vs Validation Loss")
            ax.legend()
            st.pyplot(fig)
        with g2:
            fig2, ax2 = plt.subplots(figsize=(6, 3))
            ax2.bar(
                ["Kelas 0", "Kelas 1", "Kelas 2", "Kelas 3"],
                probs,
                color="#0284C7",
            )
            ax2.set_ylabel("Probabilitas")
            ax2.set_title("Distribusi Output Softmax per Kelas")
            st.pyplot(fig2)
