# Premier League CGAN Analysis - Skripsi Project
## IMPLEMENTASI ALGORITMA GANS MODEL CGANS PADA PEMODELAN DATA PASSING NETWORKS: PERTANDINGAN LIGA INGGRIS 2024/2025

Aplikasi web interaktif menggunakan Conditional Generative Adversarial Networks (CGAN) untuk menganalisis dan menghasilkan passing networks Premier League 2024/2025 dengan data autentik Fantasy Premier League (579 pemain).

## 🎯 Fitur Utama CGAN

- **Enhanced Passing Networks**: Generate passing networks dengan CGAN real-time
- **6 Fitur AI Analysis**: Passing combinations, shot prediction, ball direction, goal probability zones
- **Authentic FPL Data**: 579 pemain Premier League 2024/2025 dengan statistik autentik
- **5 Formation Support**: 4-3-3, 4-4-2, 4-2-3-1, 3-5-2, 5-3-2
- **Interactive Web Interface**: Streamlit application dengan kontrol real-time
- **CGAN Architecture**: Generator (164→512→1024→512→22) + Discriminator (86→256→128→64→1)

## 🛠️ Requirements

### Python Version
- Python 3.8 atau lebih tinggi

### Dependencies
```
streamlit>=1.28.0
pandas>=1.5.0
numpy>=1.21.0
matplotlib>=3.5.0
torch>=1.12.0
torchvision>=0.13.0
requests>=2.28.0
beautifulsoup4>=4.11.0
scikit-learn>=1.1.0
seaborn>=0.11.0
selenium>=4.0.0
trafilatura>=1.6.0
```

## 🚀 TUTORIAL LENGKAP MENJALANKAN DI VSCODE

### LANGKAH 1: Download & Setup Project
1. **Download semua file dari project ini:**
   - `app.py` (file utama aplikasi)
   - `Fantasy Premier League export 2025-06-16 20-22-18_1750105499834.csv` (data FPL autentik)
   - `premier_league_full_380_matches.csv` (data pertandingan)
   - `api_football_client.py` (client API)
   - `requirements.txt` (dependencies)

2. **Buat folder project baru di komputer:**
   ```bash
   mkdir premier-league-cgan
   cd premier-league-cgan
   ```

3. **Copy semua file yang didownload ke folder ini**

### LANGKAH 2: Setup Python Environment
1. **Buka VSCode** dan open folder `premier-league-cgan`

2. **Buka Terminal di VSCode** (Ctrl+` atau Terminal → New Terminal)

3. **Buat Virtual Environment:**
   ```bash
   # Windows
   python -m venv venv
   venv\Scripts\activate
   
   # macOS/Linux  
   python3 -m venv venv
   source venv/bin/activate
   ```

4. **Install semua dependencies:**
   ```bash
   pip install streamlit pandas numpy matplotlib torch torchvision requests beautifulsoup4 scikit-learn seaborn selenium trafilatura
   ```

### LANGKAH 3: Setup API Key (WAJIB)
1. **Daftar di Football-Data.org:**
   - Buka https://www.football-data.org/
   - Klik "Get your free API key"
   - Daftar dengan email Anda
   - Konfirmasi email dan login

2. **Buat file `.env` di root folder:**
   ```bash
   # Di terminal VSCode
   touch .env  # Linux/Mac
   type nul > .env  # Windows
   ```

3. **Edit file `.env` dan isi dengan API key:**
   ```
   FOOTBALL_DATA_ORG_KEY=your_actual_api_key_here
   ```

### LANGKAH 4: Setup Streamlit Configuration
1. **Buat folder `.streamlit`:**
   ```bash
   mkdir .streamlit
   ```

2. **Buat file `config.toml` di dalam folder `.streamlit`:**
   ```toml
   [server]
   headless = true
   address = "0.0.0.0"
   port = 5000
   
   [theme]
   backgroundColor = "#0E1117"
   secondaryBackgroundColor = "#262730"
   textColor = "#FAFAFA"
   ```

### LANGKAH 5: Menjalankan Aplikasi
1. **Pastikan virtual environment aktif** (terlihat `(venv)` di terminal)

2. **Jalankan aplikasi:**
   ```bash
   streamlit run app.py --server.port 5000
   ```

3. **Buka browser dan akses:**
   ```
   http://localhost:5000
   ```

### LANGKAH 6: Jalankan Setup Script (OTOMATIS)
**Cara mudah menggunakan script setup otomatis:**

```bash
# Jalankan setup script
python setup_vscode.py
```

Script ini akan otomatis:
- ✅ Cek versi Python
- ✅ Cek virtual environment
- ✅ Install semua dependencies
- ✅ Cek file yang diperlukan
- ✅ Setup konfigurasi Streamlit
- ✅ Cek API key

### LANGKAH 7: Troubleshooting Manual
Jika ada error, jalankan perintah ini:

```bash
# Cek versi Python
python --version

# Upgrade pip
pip install --upgrade pip

# Install dari file requirements
pip install -r requirements_vscode.txt

# Install dependencies satu per satu jika gagal
pip install streamlit pandas numpy matplotlib torch

# Cek installed packages
pip list

# Restart aplikasi
streamlit run app.py --server.port 5000
```

## 🎮 Running the Application

### Via Command Line
```bash
streamlit run app.py
```

### Via VS Code
1. Open terminal di VS Code
2. Aktifkan virtual environment
3. Run: `streamlit run app.py`
4. Browser akan terbuka otomatis di `http://localhost:8501`

## 📊 Data Structure

### Matches Schedule (`matches_schedule.csv`)
```csv
match_id,home_team,away_team,match_date,match_time,status
1,Arsenal FC,Manchester City,2024-08-17,15:00,FINISHED
```

### Match Lineups (`match_lineups.csv`)
```csv
match_id,team_name,player_name,position,jersey_number,starting_eleven
1,Arsenal FC,David Raya,GK,22,True
```

### Match Events (`match_events.csv`)
```csv
match_id,team_name,player_name,event_type,event_time,minute
1,Arsenal FC,Martin Ødegaard,Pass,00:45,1
```

### Passing Events (`passing_events.csv`)
```csv
match_id,passer_name,receiver_name,pass_time,x_start,y_start,x_end,y_end
1,Martin Ødegaard,Bukayo Saka,00:45,0.6,0.3,0.8,0.4
```

## 🎨 Features Overview

### 1. Dual Field Visualization
- Format seperti referensi gambar yang diberikan
- Lapangan kiri: Home team passing network
- Lapangan kanan: Away team passing network
- Background abu-abu gelap dengan garis putih

### 2. Player Representation
- Lingkaran kuning dengan initial pemain
- Posisi mencakup seluruh lapangan
- Striker dan attacking midfielder di area lawan

### 3. Passing Network Analysis
- Garis putih menghubungkan pemain
- Ketebalan garis menunjukkan frekuensi passing
- Data autentik Premier League 2024/2025

### 4. Interactive Controls
- Filter pertandingan
- Rentang waktu analisis
- Informasi statistik detail

## 🔧 Configuration

### Streamlit Configuration
Buat folder `.streamlit` dan file `config.toml`:
```toml
[server]
headless = true
address = "0.0.0.0"
port = 8501

[theme]
backgroundColor = "#0E1117"
secondaryBackgroundColor = "#262730"
textColor = "#FAFAFA"
```

## 📁 File Structure
```
premier-league-cgan-analysis/
├── app.py                          # Main Streamlit application
├── requirements.txt                # Python dependencies
├── README.md                       # Documentation
├── .env                           # Environment variables (API keys)
├── .streamlit/
│   └── config.toml                # Streamlit configuration
├── matches_schedule.csv           # Match fixtures data
├── match_lineups.csv             # Player lineups data
├── match_events.csv              # Match events data
├── passing_events.csv            # Detailed passing data
├── api_football_client.py        # API client for Football-Data.org
├── authentic_premier_league_data.py # Authentic data handler
└── data_processor.py             # Data processing utilities
```

## 🌐 API Integration

### Football-Data.org API
- **Free Tier**: 10 requests/minute, 100 requests/day
- **Paid Tier**: Unlimited requests untuk data real-time
- **Coverage**: Premier League 2024/2025 season

### Example API Usage
```python
from api_football_client import APIFootballClient

client = APIFootballClient(api_key="your_key")
teams = client.get_premier_league_teams()
matches = client.get_premier_league_fixtures()
```

## 🎯 Research Components

### CGAN Architecture
- **Generator**: Random noise + conditional vector → Player positions
- **Discriminator**: Position validation dengan conditional input
- **Training**: Binary cross-entropy loss, Adam optimizer

### Thesis Integration
- Metodologi penelitian CGAN untuk analisis sepakbola
- Integrasi data Premier League autentik
- Framework scalable untuk liga berbeda

## 🐛 Troubleshooting

### Common Issues

1. **ModuleNotFoundError**
   ```bash
   pip install -r requirements.txt
   ```

2. **API Key Error**
   - Periksa file `.env`
   - Verifikasi API key di Football-Data.org

3. **Port Already in Use**
   ```bash
   streamlit run app.py --server.port 8502
   ```

4. **CSV File Missing**
   - Pastikan semua file CSV ada di root directory
   - Jalankan script data generation jika diperlukan

### Performance Tips
- Gunakan virtual environment
- Batasi rentang waktu analisis untuk performa optimal
- Cache data untuk mengurangi API calls

## 📝 License

Project ini menggunakan data autentik Premier League melalui Football-Data.org API. Pastikan mematuhi terms of service API.

## 👨‍💻 Development

### Local Development
1. Fork repository
2. Buat branch fitur baru
3. Implement changes
4. Test dengan data sample
5. Submit pull request

### Contributing
- Follow Python PEP 8 style guide
- Gunakan data autentik Premier League
- Document semua functions
- Test dengan multiple scenarios

## 📞 Support

Untuk pertanyaan atau masalah:
1. Check troubleshooting section
2. Verify API key dan dependencies
3. Pastikan semua file CSV tersedia
4. Test dengan data sample terlebih dahulu

---

**Note**: Aplikasi ini menggunakan data autentik Premier League 2024/2025. Pastikan memiliki API key yang valid dari Football-Data.org untuk fungsi penuh.#   P a s s i n g - N e t w o r k s 
 
 #   P a s s i n g N e t w o r k s - E P L 2 4 - 2 5 
 
 #   P a s s i n g N e t w o r k s - E P L 2 4 - 2 5 
 
 #   P a s s i n g N e t w o r k s - E P L 2 4 - 2 5 
 
 
#   P a s s i n g N e t w o r k s _ E P L 2 4 - 2 5  
 #   P a s s i n g N e t w o r k s _ E P L 2 4 - 2 5  
 