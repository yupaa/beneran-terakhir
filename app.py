import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px


# ============================================================
# CONFIG
# ============================================================

st.set_page_config(
    page_title="Nowcasting PDB Sektoral",
    page_icon="📊",
    layout="wide"
)


# ============================================================
# LOAD DATA
# ============================================================

@st.cache_data
def load_prediction_data():

    return pd.read_excel(
        "hasil_aktual_prediksi.xlsx",
        sheet_name="Aktual_Prediksi"
    )


@st.cache_resource
def load_model():

    return joblib.load(
        "model_nowcasting.pkl"
    )


df_prediksi = load_prediction_data()
model_data = load_model()


# ============================================================
# DAFTAR SEKTOR
# ============================================================

sektor_nama = {

    "A": "Pertanian, Kehutanan, dan Perikanan",

    "B": "Pertambangan dan Penggalian",

    "C": "Industri Pengolahan",

    "D": "Pengadaan Listrik dan Gas",

    "E": "Pengelolaan Air, Sampah, Limbah dan Daur Ulang",

    "F": "Konstruksi",

    "G": "Perdagangan Besar dan Eceran",

    "H": "Transportasi dan Pergudangan",

    "I": "Penyediaan Akomodasi dan Makan Minum",

    "J": "Informasi dan Komunikasi",

    "K": "Jasa Keuangan dan Asuransi",

    "L": "Real Estat",

    "MN": "Jasa Perusahaan",

    "O": "Administrasi Pemerintahan",

    "P": "Jasa Pendidikan",

    "Q": "Jasa Kesehatan dan Kegiatan Sosial",

    "RSTU": "Jasa Lainnya"
}


sektor_list = list(sektor_nama.keys())


# ============================================================
# GOOGLE TRENDS
# ============================================================

gt_columns = [

    "Perkebunan",
    "Peternakan",
    "Hortikultura",
    "Perburuan",
    "Perikanan",

    "Gas alam",
    "Minyak bumi",
    "Energi panas bumi",

    "Industri",
    "Industri makanan",
    "Industri tekstil",
    "Industri kimia",
    "Farmasi industri",
    "Industri pulp dan kertas",
    "Industri plastik",
    "Perabotan",

    "Listrik",
    "Es batu",

    "Pengelolaan sampah",
    "Pengolahan limbah",
    "Penyediaan air",
    "Daur ulang",

    "Konstruksi",

    "Perkulakan",
    "Perdagangan",

    "Transportasi",

    "Akomodasi",
    "Makanan dan minuman",

    "Komunikasi",
    "Informasi",
    "Telekomunikasi",
    "Penerbitan",

    "Asuransi",
    "Jasa keuangan",

    "Lahan yasan",

    "Konsultan",
    "Alih daya",

    "Pengadaan",
    "Pemerintahan",

    "Pendidikan",

    "Kesehatan",
    "Pekerja sosial",

    "Hiburan",
    "Kesenian"
]


official_columns = [
    "inflasi",
    "jisdor",
    "import",
    "export"
]


all_predictor_columns = (
    official_columns +
    gt_columns
)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("📊 Nowcasting PDB")

menu = st.sidebar.radio(
    "Menu",
    [
        "Visualisasi",
        "Prediksi"
    ]
)

st.sidebar.markdown("---")

st.sidebar.caption(
    "Nowcasting Pertumbuhan PDB Sektoral YoY"
)


# ============================================================
# MENU 1 — VISUALISASI
# ============================================================

if menu == "Visualisasi":

    st.title(
        "📈 Visualisasi Pertumbuhan PDB Sektoral"
    )

    st.write(
        "Visualisasi pertumbuhan PDB YoY aktual dan "
        "prediksi berdasarkan best model SVR setiap sektor."
    )


    # --------------------------------------------------------
    # PILIH SEKTOR
    # --------------------------------------------------------

    selected_sector = st.selectbox(

        "Pilih sektor",

        sektor_list,

        format_func=lambda x:
            f"{x} — {sektor_nama[x]}"
    )


    # --------------------------------------------------------
    # FILTER DATA
    # --------------------------------------------------------

    data_sector = df_prediksi[
        df_prediksi["Sektor"] == selected_sector
    ].copy()


    if data_sector.empty:

        st.warning(
            "Data visualisasi untuk sektor ini tidak ditemukan."
        )

        st.stop()


    # --------------------------------------------------------
    # INFORMASI BEST MODEL
    # --------------------------------------------------------

    best_scenario = data_sector[
        "Skenario"
    ].iloc[0]

    rmse = data_sector[
        "RMSE"
    ].iloc[0]

    mae = data_sector[
        "MAE"
    ].iloc[0]

    r2 = data_sector[
        "R2"
    ].iloc[0]


    st.subheader(
        f"Sektor {selected_sector}"
    )

    st.caption(
        sektor_nama[selected_sector]
    )


    # --------------------------------------------------------
    # METRICS
    # --------------------------------------------------------

    col1, col2, col3, col4 = st.columns(4)


    col1.metric(
        "Best Scenario",
        best_scenario
    )


    col2.metric(
        "RMSE",
        f"{rmse:.4f}"
    )


    col3.metric(
        "MAE",
        f"{mae:.4f}"
    )


    col4.metric(
        "R²",
        f"{r2:.4f}"
    )


    # --------------------------------------------------------
    # GRAFIK
    # --------------------------------------------------------

    fig = px.line(

        data_sector,

        x="Periode",

        y=[
            "Aktual (%)",
            "Prediksi (%)"
        ],

        markers=True,

        labels={

            "value":
                "Pertumbuhan PDB YoY (%)",

            "variable":
                "Keterangan",

            "Periode":
                "Periode"
        },

        title=(
            f"Aktual vs Prediksi Pertumbuhan "
            f"PDB YoY — Sektor {selected_sector}"
        )
    )


    fig.update_layout(

        hovermode="x unified",

        legend_title_text=""

    )


    st.plotly_chart(
        fig,
        use_container_width=True
    )


    # --------------------------------------------------------
    # TABEL
    # --------------------------------------------------------

    st.subheader(
        "Data Aktual dan Prediksi"
    )


    kolom_tabel = [

        "Periode",
        "Aktual (%)",
        "Prediksi (%)",
        "Skenario",
        "RMSE",
        "MAE",
        "MAPE (%)",
        "NRMSE (%)",
        "R2",
        "Pearson r"

    ]


    st.dataframe(

        data_sector[kolom_tabel],

        use_container_width=True,

        hide_index=True
    )


# ============================================================
# MENU 2 — PREDIKSI
# ============================================================

elif menu == "Prediksi":

    st.title(
        "🔮 Nowcasting Pertumbuhan PDB Sektoral"
    )

    st.write(
        "Masukkan nilai predictor dalam bentuk yang sama "
        "dengan data yang digunakan saat training model. "
        "Variabel jisdor, import, dan export dimasukkan "
        "dalam bentuk pertumbuhan YoY."
    )


    st.info(
        "⚠️ Input tidak dihitung YoY oleh dashboard. "
        "Pengguna harus menghitung YoY terlebih dahulu "
        "dengan membandingkan bulan yang sama dengan "
        "tahun sebelumnya."
    )


    # ========================================================
    # PILIH SEKTOR
    # ========================================================

    selected_sector = st.selectbox(

        "Pilih sektor yang akan di-nowcast",

        sektor_list,

        format_func=lambda x:
            f"{x} — {sektor_nama[x]}"
    )


    # ========================================================
    # AMBIL BEST MODEL
    # ========================================================

    if selected_sector not in model_data["models"]:

        st.error(
            f"Model untuk sektor {selected_sector} "
            "tidak ditemukan."
        )

        st.stop()


    model_info = model_data[
        "models"
    ][selected_sector]


    scenario = model_info[
        "skenario"
    ]


    # ========================================================
    # INFORMASI MODEL
    # ========================================================

    st.subheader(
        "Model Terbaik"
    )


    col1, col2, col3, col4 = st.columns(4)


    col1.metric(
        "Skenario",
        scenario
    )


    col2.metric(
        "Kernel",
        model_info["best_kernel"]
    )


    col3.metric(
        "RMSE",
        f"{model_info['RMSE']:.4f}"
    )


    col4.metric(
        "MAE",
        f"{model_info['MAE']:.4f}"
    )


    st.caption(
        f"Model otomatis menggunakan best model "
        f"sektor {selected_sector} berdasarkan RMSE test."
    )


    # ========================================================
    # DOWNLOAD TEMPLATE
    # ========================================================

    st.subheader(
        "📥 Template Input"
    )


    st.write(
        "Gunakan template berikut untuk memasukkan "
        "48 variabel predictor."
    )


    template = pd.DataFrame({

        "Variabel":
            all_predictor_columns,

        "Nilai":
            [None] * len(all_predictor_columns)

    })


    # Template CSV untuk preview sederhana
    csv_template = template.to_csv(
        index=False
    ).encode("utf-8")


    st.download_button(

        label="⬇️ Download Template CSV",

        data=csv_template,

        file_name=
            "template_input_nowcasting.csv",

        mime="text/csv"
    )


    st.markdown("---")


    # ========================================================
    # UPLOAD INPUT
    # ========================================================

    st.subheader(
        "📤 Upload Data Predictor"
    )


    uploaded_file = st.file_uploader(

        "Upload file CSV atau Excel",

        type=[
            "csv",
            "xlsx"
        ]
    )


    if uploaded_file is not None:

        try:

            # ------------------------------------------------
            # BACA FILE
            # ------------------------------------------------

            if uploaded_file.name.endswith(".csv"):

                input_df = pd.read_csv(
                    uploaded_file
                )

            else:

                input_df = pd.read_excel(
                    uploaded_file
                )


            st.subheader(
                "Preview Data"
            )


            st.dataframe(

                input_df,

                use_container_width=True,

                hide_index=True
            )


            # ------------------------------------------------
            # VALIDASI KOLOM
            # ------------------------------------------------

            missing_columns = [

                col

                for col in all_predictor_columns

                if col not in input_df.columns

            ]


            if missing_columns:

                st.error(
                    "Kolom berikut belum tersedia "
                    "dalam file input:"
                )


                st.write(
                    missing_columns
                )


                st.stop()


            # ------------------------------------------------
            # VALIDASI JUMLAH BARIS
            # ------------------------------------------------

            if len(input_df) != 1:

                st.warning(
                    "Untuk prediksi, masukkan tepat "
                    "1 baris data predictor."
                )

                st.stop()


            # ------------------------------------------------
            # AMBIL FITUR SESUAI BEST MODEL
            # ------------------------------------------------

            fitur = model_info[
                "fitur"
            ]


            st.subheader(
                "Fitur yang Digunakan Model"
            )


            st.write(
                f"Best model {selected_sector} "
                f"menggunakan {len(fitur)} fitur."
            )


            with st.expander(
                "Lihat fitur model"
            ):

                st.write(
                    fitur
                )


            # ------------------------------------------------
            # SIAPKAN INPUT
            # ------------------------------------------------

            if scenario in [
                "S1",
                "S2",
                "S4"
            ]:

                X_input = input_df[
                    fitur
                ].copy()


            elif scenario in [
                "S3",
                "S5"
            ]:

                # --------------------------------------------
                # AMBIL SELURUH GT
                # --------------------------------------------

                X_GT_input = input_df[
                    gt_columns
                ].copy()


                # --------------------------------------------
                # PCA PREPROCESSING
                # --------------------------------------------

                scaler_pca = model_data[
                    "pca"
                ][
                    "scaler_pca"
                ]


                pca = model_data[
                    "pca"
                ][
                    "pca"
                ]


                pc_names = model_data[
                    "pca"
                ][
                    "pc_names"
                ]


                X_GT_scaled = scaler_pca.transform(
                    X_GT_input
                )


                X_GT_pca = pca.transform(
                    X_GT_scaled
                )


                PC_input = pd.DataFrame(

                    X_GT_pca,

                    columns=pc_names,

                    index=input_df.index
                )


                # --------------------------------------------
                # S3
                # --------------------------------------------

                if scenario == "S3":

                    X_input = PC_input[
                        fitur
                    ]


                # --------------------------------------------
                # S5
                # --------------------------------------------

                elif scenario == "S5":

                    X_input = pd.concat(

                        [

                            PC_input,

                            input_df[
                                official_columns
                            ]

                        ],

                        axis=1
                    )


                    X_input = X_input[
                        fitur
                    ]


            else:

                st.error(
                    f"Skenario {scenario} tidak dikenali."
                )

                st.stop()


            # =================================================
            # CEK NUMERIC
            # =================================================

            X_input = X_input.apply(
                pd.to_numeric,
                errors="coerce"
            )


            if X_input.isna().any().any():

                st.error(
                    "Terdapat nilai kosong atau "
                    "non-numerik pada fitur model."
                )

                st.dataframe(
                    X_input[
                        X_input.isna().any(axis=1)
                    ]
                )

                st.stop()


            # =================================================
            # STANDARDISASI SESUAI MODEL
            # =================================================

            scaler_model = model_info[
                "scaler"
            ]


            model = model_info[
                "model"
            ]


            X_scaled = scaler_model.transform(
                X_input
            )


            # =================================================
            # PREDIKSI
            # =================================================

            prediction = model.predict(
                X_scaled
            )[0]


            # =================================================
            # HASIL
            # =================================================

            st.success(
                "Nowcasting berhasil dilakukan!"
            )


            st.markdown(
                "## 🎯 Hasil Nowcasting"
            )


            st.metric(

                "Prediksi Pertumbuhan PDB YoY",

                f"{prediction:.4f}%"
            )


            # =================================================
            # DETAIL INPUT MODEL
            # =================================================

            with st.expander(
                "Lihat data yang masuk ke model"
            ):

                st.dataframe(

                    X_input,

                    use_container_width=True,

                    hide_index=True
                )


        except Exception as e:

            st.error(
                f"Terjadi kesalahan saat memproses file: {e}"
            )


# ============================================================
# FOOTER
# ============================================================

st.sidebar.markdown("---")

st.sidebar.caption(
    "Dashboard Nowcasting PDB Sektoral • SVR"
)