# ⚽ The Next Superstar Football Finder

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-red)
![Scikit-Learn](https://img.shields.io/badge/ML-Hybrid%20AI-orange)
![License](https://img.shields.io/badge/License-MIT-green)

**The Next Superstar Finder** is a sophisticated Machine Learning-based scouting tool designed to identify football talents. Unlike traditional recommendation systems, this project utilizes a **Hybrid AI Approach** combining Classification and Clustering to provide highly accurate "Moneyball-style" scouting.

It serves two main purposes: looking for replacements for established stars, and matching a user's own statistics to professional players.

## 🚀 Key Features

### 1. 🕵️ Player Similarity Search (Scouting Mode)
- **Data-Driven Scouting:** Find the top 5 players statistically identical to any target player (e.g., "Find me the next Erling Haaland").
- **Smart Filtering:** Narrow down results by **Age**, **Market Value**, and specific attributes.
- **Visual Comparison:** Interactive Radar Charts to compare the target vs. recommendations side-by-side.

### 2. ⭐ Pro Player Matcher (Input Your Stats)
- **"Which Star Are You?":** Input your own physical (Height, Weight) and technical stats.
- **AI Position Prediction:** The system uses a **Random Forest Classifier** to analyze your stats and predict your ideal position on the field.
- **Physical Profiling:** Takes **Height** and **Weight** into account—ensuring a 170cm winger isn't compared to a 195cm center-back.
- **Dominant Foot Filter:** Specifically search for Left-footed or Right-footed players.

## 🛠️ Tech Stack & Methodology

The core intelligence of this application relies on a **Hybrid Machine Learning Pipeline**:

### 1.  **Feature Engineering:**
    * Processes **37 Attributes** including technical skills (Finishing, Tackle), physical traits (Height, Weight), and mental stats.
    * Data Normalization using `StandardScaler`.

### 2.  **Stage 1: Position Classification (Random Forest):**
    * A supervised model predicts the broad position (Attacker, Midfielder, Defender) based on input statistics.
    * *Accuracy:* ~88% on test data.

### 3.  **Stage 2: Similarity Search (K-Nearest Neighbors):**
    * Unsupervised learning calculates the Euclidean distance between players within the predicted position "cluster".
    * Ensures recommendations are positionally and physically relevant.

## 📂 Project Structure

```bash
football-talent-scout/
├── data/
│   ├── raw/             # Place your 'players_24_complete.csv' here
│   └── processed/       # Generated files (players_clean.csv, players_labeled.csv)
├── models/              # Saved AI models (.pkl files)
├── notebooks/           # Jupyter notebooks for training
│   ├── 01_model_building.ipynb    # Cleaning & KNN Training
│   └── 02_build_classifier.ipynb  # Random Forest Training
├── src/
│   └── app.py           # Main Streamlit Application
└── requirements.txt     # Python dependencies
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

- Download EA Sports FC 24 Complete Player Database dari Kaggle [Kaggle_Dataset](https://www.kaggle.com/datasets/stefanoleone992/ea-sports-fc-24-complete-player-dataset?select=male_players.csv)
- Extract file ZIP
- Rename file utama menjadi male_players.csv
- Pindahkan ke folder: ```bash data/raw/male_players.csv ```

### 5️⃣ Train the Models

Sebelum menjalankan aplikasi, Anda perlu melatih model agar file .pkl terbentuk. Jalankan notebook di folder notebooks/ secara berurutan:

    Run bash 01_model_building.ipynb

    Run 02_build_classifier.ipynb

### 6️⃣ Run the Application
```bash
streamlit run src/app.py
```

## 🔮 Future Improvements

    [ ] League Filtering: Filter recommendations based on specific leagues (Premier League, La Liga, etc.).

    [ ] Squad Builder: AI-driven team creation within a budget cap.

    [ ] Historical Data: Comparing players across different seasons.

## 🤝 Contributing

Kontribusi sangat terbuka!
Silakan buat issues atau pull request.

## 📝 License

Project ini menggunakan MIT License.

**Developed By MaglonMartino**