import streamlit as st
import pandas as pd
import joblib
import numpy as np
import plotly.graph_objects as go

# ==========================================
# 1. KONFIGURASI HALAMAN WEB
# ==========================================
st.set_page_config(
    page_title="Football Talent Scout",
    page_icon="⚽",
    layout="wide"
)

# Judul Utama
st.title("⚽ Football Talent Scout: The Next Superstar Finder")
st.markdown("---")

# ==========================================
# 2. FUNGSI LOAD DATA & MODEL (Supaya Cepat)
# ==========================================
@st.cache_data
def load_data():
    # Load dataset yang sudah bersih
    # Path '..' berarti mundur satu folder dari 'src' ke folder utama
    df = pd.read_csv('data/processed/players_clean.csv')
    return df

@st.cache_resource
def load_models():
    # Load model KNN dan Scaler
    knn = joblib.load('models/knn_model.pkl')
    scaler = joblib.load('models/scaler.pkl')
    return knn, scaler

try:
    df = load_data()
    knn_model, scaler = load_models()
except FileNotFoundError:
    st.error("❌ File tidak ditemukan! Pastikan kamu sudah menjalankan notebook dan path-nya benar.")
    st.stop()

# Daftar Fitur Skill yang dipakai AI (Harus sama persis dengan urutan di Notebook)
feature_cols = [
    'pace', 'shooting', 'passing', 'dribbling', 'defending', 'physic',
    'attacking_crossing', 'attacking_finishing', 'attacking_heading_accuracy',
    'attacking_short_passing', 'attacking_volleys',
    'skill_dribbling', 'skill_curve', 'skill_fk_accuracy', 'skill_long_passing',
    'skill_ball_control',
    'movement_acceleration', 'movement_sprint_speed', 'movement_agility',
    'movement_reactions', 'movement_balance',
    'power_shot_power', 'power_jumping', 'power_stamina', 'power_strength',
    'power_long_shots',
    'mentality_aggression', 'mentality_interceptions', 'mentality_positioning',
    'mentality_vision', 'mentality_penalties', 'mentality_composure',
    'defending_marking_awareness', 'defending_standing_tackle', 'defending_sliding_tackle'
]

# ==========================================
# 3. FITUR VISUALISASI (RADAR CHART)
# ==========================================
def plot_radar_chart(player1, player2):
    # Kita bandingkan 6 Atribut Utama saja biar grafik enak dilihat
    categories = ['pace', 'shooting', 'passing', 'dribbling', 'defending', 'physic']
    
    # Ambil data player 1 dan 2
    val1 = player1[categories].values.flatten().tolist()
    val2 = player2[categories].values.flatten().tolist()
    
    # Supaya grafiknya nyambung, ulangi titik pertama di akhir
    val1 += val1[:1]
    val2 += val2[:1]
    categories += categories[:1]

    fig = go.Figure()

    # Grafik Player Target (Merah)
    fig.add_trace(go.Scatterpolar(
        r=val1, theta=categories, fill='toself', name=player1['short_name'].values[0],
        line_color='red'
    ))

    # Grafik Player Rekomendasi (Biru)
    fig.add_trace(go.Scatterpolar(
        r=val2, theta=categories, fill='toself', name=player2['short_name'].values[0],
        line_color='blue'
    ))

    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
        showlegend=True,
        height=400
    )
    
    return fig

# ==========================================
# 4. SIDEBAR (PANEL KIRI)
# ==========================================
st.sidebar.header("🔍 Cari Pemain")

# Dropdown Pilih Pemain
# Kita urutkan nama pemain biar gampang dicari
player_list = df['short_name'].sort_values().unique()
selected_player_name = st.sidebar.selectbox("Pilih Pemain Target:", player_list)

# Filter Tambahan (Opsional)
st.sidebar.markdown("---")
st.sidebar.subheader("⚙️ Filter Rekomendasi")
max_age = st.sidebar.slider("Maksimal Umur (Rekomendasi)", 15, 40, 25)
max_price = st.sidebar.number_input("Maksimal Harga (€)", value=100000000, step=1000000)

# Tombol Eksekusi
btn_search = st.sidebar.button("🚀 Cari 'The Next' Superstar")

# ==========================================
# 5. LOGIKA UTAMA (MAIN APP)
# ==========================================
if btn_search:
    # 1. Ambil Data Pemain Target
    target_player = df[df['short_name'] == selected_player_name]
    
    if len(target_player) == 0:
        st.error("Pemain tidak ditemukan!")
    else:
        # Tampilkan Info Pemain Target
        c1, c2, c3 = st.columns([1, 2, 2])
        with c1:
            # Tampilkan Foto (Jika kolom URL ada)
            if 'player_face_url' in target_player.columns:
                st.image(target_player['player_face_url'].values[0], width=100)
            else:
                st.write("📷 No Image")
        with c2:
            st.subheader(target_player['short_name'].values[0])
            st.write(f"**Club:** {target_player['club_name'].values[0]}")
            st.write(f"**Age:** {target_player['age'].values[0]}")
            st.write(f"**Market Value:** €{target_player['value_eur'].values[0]:,.0f}")
        with c3:
            st.metric("Overall Rating", target_player['overall'].values[0])

        st.markdown("### 🔥 Top 5 Rekomendasi Pemain Mirip")
        
        # 2. PROSES MACHINE LEARNING
        # Ambil fitur skill dari target player
        target_features = target_player[feature_cols].values
        
        # Normalisasi data (PENTING: Pakai scaler yang sama dengan saat training)
        target_features_scaled = scaler.transform(target_features)
        
        # Cari tetangga terdekat (KNN)
        # n_neighbors=20 (ambil agak banyak dulu, nanti difilter umur/harga)
        distances, indices = knn_model.kneighbors(target_features_scaled, n_neighbors=20)
        
        # 3. TAMPILKAN HASIL
        recommendations = df.iloc[indices[0]] # Ambil data pemain berdasarkan index
        
        # Filter: Buang pemain itu sendiri (Diri sendiri pasti paling mirip)
        recommendations = recommendations[recommendations['short_name'] != selected_player_name]
        
        # Filter: Sesuai Slider (Umur & Harga)
        filtered_recs = recommendations[
            (recommendations['age'] <= max_age) & 
            (recommendations['value_eur'] <= max_price)
        ].head(5) # Ambil 5 teratas setelah filter

        if len(filtered_recs) == 0:
            st.warning("⚠️ Tidak ada pemain yang cocok dengan filter umur/harga kamu. Coba longgarkan filter.")
        else:
            # Loop untuk menampilkan kartu pemain
            for idx, row in filtered_recs.iterrows():
                with st.container():
                    st.markdown("---")
                    col_img, col_info, col_chart = st.columns([1, 2, 3])
                    
                    with col_img:
                        if 'player_face_url' in row:
                            st.image(row['player_face_url'], width=80)
                        st.caption(f"Rank #{idx}") # Ini bug kecil indexnya ngaco gpp
                    
                    with col_info:
                        st.subheader(row['short_name'])
                        st.write(f"🌍 {row['nationality_name']} | 👕 {row['club_name']}")
                        st.write(f"🎂 {row['age']} Tahun")
                        st.success(f"💰 €{row['value_eur']:,.0f}")
                    
                    with col_chart:
                        # Tampilkan Radar Chart perbandingan
                        # Mengubah row (Series) menjadi DataFrame 1 baris agar fungsi plot bisa baca
                        row_df = pd.DataFrame([row]) 
                        chart = plot_radar_chart(target_player, row_df)
                        st.plotly_chart(chart, use_container_width=True)

else:
    # Tampilan Awal sebelum search
    st.info("👈 Silakan pilih pemain di menu sebelah kiri untuk memulai pencarian.")
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/d/d3/Soccerball.svg/1200px-Soccerball.svg.png", width=200)