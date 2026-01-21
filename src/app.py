import streamlit as st
import pandas as pd
import joblib
import numpy as np
import plotly.graph_objects as go
import os # <--- Tambahan wajib untuk Path

# ==========================================
# 1. SETUP & KONFIGURASI HALAMAN
# ==========================================
st.set_page_config(
    page_title="Football Talent Scout AI",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .metric-card {background-color: #f0f2f6; border-radius: 10px; padding: 15px; text-align: center;}
    h1 {color: #1f77b4;}
</style>
""", unsafe_allow_html=True)

st.markdown("<h1 style='text-align: center;'>⚽ The Next Superstar Finder</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Powered by Hybrid AI (Random Forest Classifier + KNN Euclidean)</p>", unsafe_allow_html=True)
st.markdown("---")

# ==========================================
# 2. LOAD DATA & MODEL (BULLETPROOF PATH)
# ==========================================
@st.cache_data
def load_data():
    # Cari lokasi file app.py saat ini
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # Mundur satu folder ke belakang (folder project root)
    project_root = os.path.dirname(current_dir)
    # Gabungkan path secara absolut
    file_path = os.path.join(project_root, 'data', 'processed', 'players_labeled.csv')
    
    df = pd.read_csv(file_path) 
    return df

@st.cache_resource
def load_models():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    
    # Definisi path model
    path_knn = os.path.join(project_root, 'models', 'knn_model.pkl')
    path_scaler = os.path.join(project_root, 'models', 'scaler.pkl')
    path_rf = os.path.join(project_root, 'models', 'rf_classifier.pkl')

    knn = joblib.load(path_knn)
    scaler = joblib.load(path_scaler)
    rf_classifier = joblib.load(path_rf)
    return knn, scaler, rf_classifier

try:
    df = load_data()
    knn_model, scaler, rf_model = load_models()
except FileNotFoundError as e:
    st.error(f"❌ File Data/Model tidak ditemukan: {e}")
    st.stop()
except Exception as e:
    st.error(f"❌ Terjadi kesalahan sistem: {e}")
    st.stop()

feature_cols = [
    'height_cm', 'weight_kg',
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
# 3. FUNGSI VISUALISASI
# ==========================================
def plot_radar_chart(player_name, player_stats, comparison_name, comparison_stats):
    categories = ['Pace', 'Shooting', 'Passing', 'Dribbling', 'Defending', 'Physic']
    
    def get_val(stats, col):
        if isinstance(stats, dict): return stats.get(col, 0)
        try:
            return stats[col].values[0]
        except:
            return 0

    val1 = [get_val(player_stats, c.lower()) for c in categories]
    val2 = [get_val(comparison_stats, c.lower()) for c in categories]

    val1 += val1[:1]
    val2 += val2[:1]
    plot_cat = categories + categories[:1]

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(r=val1, theta=plot_cat, fill='toself', name=player_name, line_color='red'))
    fig.add_trace(go.Scatterpolar(r=val2, theta=plot_cat, fill='toself', name=comparison_name, line_color='blue'))

    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 100])), 
        height=300, 
        margin=dict(t=30, b=30),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    return fig

def show_player_image(row):
    img_url = None
    if 'player_face_url' in row and pd.notna(row['player_face_url']):
        img_url = row['player_face_url']
    elif 'player_url' in row and pd.notna(row['player_url']):
        val_url = str(row['player_url'])
        if val_url.startswith('http'):
            img_url = val_url
            
    if img_url:
        try:
            st.image(img_url, width=100)
        except:
            st.write("👤")
    else:
        st.write("👤")

# ==========================================
# 4. TAB NAVIGASI UTAMA
# ==========================================
tab1, tab2 = st.tabs(["🕵️ Cari Pemain Serupa (Scouting)", "⭐ Pro Player Matcher (Input Sendiri)"])

# ==========================================
# TAB 1: SCOUTING MODE
# ==========================================
with tab1:
    st.header("🔍 Player Similarity Search")
    st.caption("Cari alternatif pemain transfer berdasarkan kemiripan statistik.")
    
    col_search, col_filter = st.columns([1, 2])
    
    with col_search:
        player_list = df['short_name'].sort_values().unique()
        selected_player = st.selectbox("Pilih Pemain Target:", player_list, index=0)
        btn_scout = st.button("🚀 Cari Rekomendasi", type="primary")
        
    with col_filter:
        c1, c2 = st.columns(2)
        with c1:
            max_age = st.slider("Maksimal Umur", 16, 40, 28)
        with c2:
            max_price = st.number_input("Maksimal Budget (€)", value=100_000_000, step=1_000_000)

    if btn_scout:
        target_row = df[df['short_name'] == selected_player]
        
        raw_stats = target_row[feature_cols].values
        scaled_stats = scaler.transform(raw_stats)
        distances, indices = knn_model.kneighbors(scaled_stats, n_neighbors=50)
        
        st.markdown("#### 🎯 Target Profile")
        col_t1, col_t2, col_t3, col_t4 = st.columns(4)
        col_t1.metric("Position", target_row['simple_position'].values[0])
        col_t2.metric("Age", f"{target_row['age'].values[0]} th")
        col_t3.metric("Height", f"{target_row['height_cm'].values[0]} cm")
        col_t4.metric("Value", f"€{target_row['value_eur'].values[0]:,.0f}")
        
        st.markdown("---")
        st.markdown("#### ✅ Top Recommendations")

        candidates = df.iloc[indices[0]]
        filtered = candidates[
            (candidates['short_name'] != selected_player) & 
            (candidates['age'] <= max_age) & 
            (candidates['value_eur'] <= max_price)
        ].head(5)
        
        if len(filtered) == 0:
            st.warning("⚠️ Tidak ditemukan pemain yang cocok dengan filter umur/budget ini.")
        else:
            for idx, row in filtered.iterrows():
                with st.expander(f"#{idx+1} {row['short_name']} ({row['club_name']})", expanded=True):
                    c_img, c_info, c_radar = st.columns([1, 2, 2])
                    
                    with c_img:
                        show_player_image(row)
                    with c_info:
                        st.subheader(row['short_name'])
                        st.caption(f"{row['nationality_name']} | {row['player_positions']}")
                        st.write(f"**Umur:** {row['age']} | **Tinggi:** {row['height_cm']} cm")
                        st.write(f"**Kaki:** {row['preferred_foot']}")
                        st.success(f"💰 Market Value: €{row['value_eur']:,.0f}")
                    with c_radar:
                        chart = plot_radar_chart(selected_player, target_row, row['short_name'], pd.DataFrame([row]))
                        st.plotly_chart(chart, use_container_width=True)


# ==========================================
# TAB 2: MATCHER MODE
# ==========================================
with tab2:
    st.header("⭐ Pro Player Matcher")
    st.caption("Masukkan data fisik & statistikmu, AI akan menebak posisimu dan mencarikan 'kembaran' pro-mu.")

    with st.form("input_form"):
        col_phys, col_def, col_att = st.columns(3)
        
        with col_phys:
            st.markdown("### 🧬 Physical")
            ui_height = st.slider("Tinggi Badan (cm)", 150, 210, 175)
            ui_weight = st.slider("Berat Badan (kg)", 50, 100, 70)
            ui_foot = st.radio("Kaki Dominan", ["Right", "Left", "Both"], horizontal=True)
            st.markdown("---")
            st.markdown("### ⚡ Movement")
            ui_pace = st.slider("Pace / Speed", 0, 99, 70)
            ui_accel = st.slider("Acceleration", 0, 99, 70)
            ui_stamina = st.slider("Stamina", 0, 99, 65)
            ui_strength = st.slider("Strength", 0, 99, 65)

        with col_def:
            st.markdown("### 🛡️ Defending")
            ui_defending = st.slider("Defending Overall", 0, 99, 50)
            ui_stand_tackle = st.slider("Stand Tackle", 0, 99, 50)
            ui_slide_tackle = st.slider("Slide Tackle", 0, 99, 50)
            ui_interception = st.slider("Interceptions", 0, 99, 50)
            ui_aggression = st.slider("Aggression", 0, 99, 60)
            ui_composure = st.slider("Composure", 0, 99, 60)
            ui_vision = st.slider("Vision", 0, 99, 60)

        with col_att:
            st.markdown("### ⚽ Attacking")
            ui_shooting = st.slider("Shooting Overall", 0, 99, 55)
            ui_finishing = st.slider("Finishing", 0, 99, 55)
            ui_passing = st.slider("Passing Overall", 0, 99, 65)
            ui_short_pass = st.slider("Short Pass", 0, 99, 65)
            ui_dribbling = st.slider("Dribbling Overall", 0, 99, 65)
            ui_ball_control = st.slider("Ball Control", 0, 99, 65)
            ui_crossing = st.slider("Crossing", 0, 99, 60)

        submitted = st.form_submit_button("🔍 Analisis & Cari Kembaran")

    if submitted:
        # Mapping input ke dictionary
        input_data = {
            'height_cm': ui_height, 'weight_kg': ui_weight, 'pace': ui_pace,
            'shooting': ui_shooting, 'passing': ui_passing, 'dribbling': ui_dribbling,
            'defending': ui_defending, 'physic': (ui_strength + ui_stamina)/2, 
            'attacking_crossing': ui_crossing, 'attacking_finishing': ui_finishing,
            'attacking_heading_accuracy': ui_height/2.5, 'attacking_short_passing': ui_short_pass,
            'attacking_volleys': ui_shooting - 10, 'skill_dribbling': ui_dribbling,
            'skill_curve': ui_passing - 5, 'skill_fk_accuracy': ui_shooting - 15,
            'skill_long_passing': ui_passing - 5, 'skill_ball_control': ui_ball_control,
            'movement_acceleration': ui_accel, 'movement_sprint_speed': ui_pace,
            'movement_agility': (110 - ui_weight), 'movement_reactions': ui_composure,
            'movement_balance': (110 - ui_height/2.2), 'power_shot_power': ui_shooting + 5,
            'power_jumping': ui_stamina, 'power_stamina': ui_stamina, 'power_strength': ui_strength,
            'power_long_shots': ui_shooting, 'mentality_aggression': ui_aggression,
            'mentality_interceptions': ui_interception, 'mentality_positioning': ui_dribbling - 5,
            'mentality_vision': ui_vision, 'mentality_penalties': ui_shooting,
            'mentality_composure': ui_composure, 'defending_marking_awareness': ui_defending,
            'defending_standing_tackle': ui_stand_tackle, 'defending_sliding_tackle': ui_slide_tackle
        }

        try:
            input_vector = np.array([input_data[col] for col in feature_cols]).reshape(1, -1)
            pred_position = rf_model.predict(input_vector)[0]
            
            st.markdown("---")
            st.info(f"🤖 **AI Analysis:** Posisi idealmu adalah **{pred_position.upper()}**")

            input_scaled = scaler.transform(input_vector)
            distances, indices = knn_model.kneighbors(input_scaled, n_neighbors=500)
            
            candidates = df.iloc[indices[0]].copy()
            candidates = candidates[candidates['simple_position'] == pred_position]
            
            if ui_foot != "Both":
                candidates = candidates[candidates['preferred_foot'] == ui_foot]
                st.caption(f"🎯 Filter: Pemain kaki **{ui_foot}**")
            
            final_results = candidates.head(5)

            if len(final_results) == 0:
                st.warning(f"Tidak ada pemain {pred_position} kaki {ui_foot} yang mirip.")
            else:
                st.markdown(f"### 🔥 Top 5 Pemain Mirip Kamu")
                
                for idx, row in final_results.iterrows():
                    with st.container():
                        # Layout Hasil (Tanpa Foto, Pakai Chart)
                        col_info, col_chart = st.columns([1, 1])
                        
                        with col_info:
                            # Tampilkan nama TANPA Pagar/Nomor
                            st.subheader(f"{row['short_name']}") 
                            st.caption(f"{row['club_name']} | {row['nationality_name']}")
                            st.write(f"📏 {row['height_cm']} cm | ⚖️ {row['weight_kg']} kg")
                            st.write(f"🦶 Foot: **{row['preferred_foot']}**")
                            
                            # Tampilkan Market Value (Kotak Hijau)
                            st.success(f"💰 Market Value: €{row['value_eur']:,.0f}")
                            
                            # Similarity Score
                            dist = distances[0][list(candidates.index).index(idx)]
                            match_score = max(0, 100 - (dist * 5))
                            st.progress(int(match_score)/100, text=f"Similarity: {int(match_score)}%")

                        with col_chart:
                            chart = plot_radar_chart("Saya (User)", input_data, row['short_name'], pd.DataFrame([row]))
                            st.plotly_chart(chart, use_container_width=True)
                        
                        st.divider()

        except Exception as e:
            st.error(f"Terjadi kesalahan: {e}")