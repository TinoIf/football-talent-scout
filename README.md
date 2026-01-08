# ⚽ The Next Superstar Football Finder

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-red)
![Scikit-Learn](https://img.shields.io/badge/ML-Scikit--Learn-orange)

**The Next Superstar Finder** is a Machine Learning-based recommendation system designed to help football scouts and managers identify potential talents by finding players with similar statistical profiles to established stars.

Using **Unsupervised Learning (K-Nearest Neighbors)** and data from FC 24, this tool enables "Moneyball-style" scouting to find undervalued players with world-class attributes.

## 🚀 Features

- **Player Similarity Search:** Find top 5 players statistically similar to any target player.
- **Interactive Radar Charts:** Visual comparison of player attributes (Pace, Shooting, Passing, etc.).
- **Smart Filtering:** Filter recommendations by Age, Market Value, and Position.
- **Comprehensive Database:** Based on 18,000+ players from major leagues.

## 🛠️ Tech Stack

- **Core:** Python
- **Data Processing:** Pandas, NumPy
- **Machine Learning:** Scikit-Learn (Nearest Neighbors, StandardScaler)
- **Visualization:** Matplotlib / Plotly
- **Web Framework:** Streamlit

## 📂 Project Structure

```bash
├── data/          # Raw and processed datasets
├── models/        # Trained .pkl models
├── notebooks/     # Jupyter notebooks for EDA and prototyping
├── src/           # Source code for the Streamlit application
└── requirements.txt
```

## Installation & Setup
Ikuti langkah berikut untuk menjalankan project secara lokal.

### 1️⃣ Clone Repository
```bash
git clone https://github.com/TinoIf/football-talent-scout.git ##jika menggunakan https
git clone git@github.com:TinoIf/football-talent-scout.git ## jika menggunakan ssh

cd football-talent-scout
```

### 2️⃣ Create Virtual Environment
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Mac / Linux
python3 -m venv venv
source venv/bin/activate
```

### 3️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

### 4️⃣ Setup the Data
Important: Dataset tidak disertakan dalam repository.

- Download EA Sports FC 24 Complete Player Database dari Kaggle [Kaggle_Dataset](https://www.kaggle.com/datasets/rehandl23/fifa-24-player-stats-dataset)
- Extract file ZIP
- Rename file utama menjadi male_players.csv
- Pindahkan ke folder: ```bash data/raw/male_players.csv ```

### 5️⃣ Run the Application
```bash
streamlit run src/app.py
```

## 🧠 How It Works (Methodology)

Sistem rekomendasi menggunakan pipeline Unsupervised Learning:

🔹 1. Data Preprocessing

Remove Goalkeepers

Handle missing values

Hapus metadata tidak relevan

🔹 2. Feature Engineering

Memilih 30+ atribut teknis (Finishing, Crossing, Ball Control, dll)

Normalisasi menggunakan StandardScaler

🔹 3. Modeling

Algoritma: K-Nearest Neighbors (KNN)

Menggunakan Euclidean Distance untuk menghitung kemiripan antar pemain

🔹 4. Inference

Sistem mengembalikan k pemain terdekat berdasarkan jarak vektor statistik


## 🔮 Future Improvements

Tambahkan filter berdasarkan liga

Fitur "Squad Builder" berdasarkan budget

Deployment ke Streamlit Cloud

## 🤝 Contributing

Kontribusi sangat terbuka!
Silakan buat issues atau pull request.

## 📝 License

Project ini menggunakan MIT License.

**Developed By MaglonMartino**