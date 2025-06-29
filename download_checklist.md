# 📥 DOWNLOAD CHECKLIST - File yang Harus Didownload

## ✅ FILE WAJIB UNTUK VSCODE

### 1. File Utama Aplikasi
- [ ] **app.py** - Aplikasi Streamlit utama dengan CGAN
- [ ] **setup_vscode.py** - Script setup otomatis untuk VSCode
- [ ] **requirements_vscode.txt** - Dependencies Python

### 2. Data Autentik
- [ ] **Fantasy Premier League export 2025-06-16 20-22-18_1750105499834.csv** - Data 579 pemain FPL
- [ ] **premier_league_full_380_matches.csv** - Data 380 pertandingan lengkap

### 3. File Pendukung
- [ ] **api_football_client.py** - Client untuk Football-Data.org API
- [ ] **README.md** - Tutorial lengkap
- [ ] **.env.example** - Template file environment

## 📋 STRUKTUR FOLDER YANG BENAR

```
premier-league-cgan/
├── app.py
├── setup_vscode.py
├── requirements_vscode.txt
├── README.md
├── .env (buat sendiri)
├── .streamlit/
│   └── config.toml (dibuat otomatis)
├── Fantasy Premier League export 2025-06-16 20-22-18_1750105499834.csv
├── premier_league_full_380_matches.csv
├── api_football_client.py
└── .env.example
```

## 🚀 URUTAN SETUP SETELAH DOWNLOAD

1. **Buat folder baru:**
   ```bash
   mkdir premier-league-cgan
   cd premier-league-cgan
   ```

2. **Copy semua file yang didownload ke folder ini**

3. **Buka VSCode dan open folder `premier-league-cgan`**

4. **Setup virtual environment:**
   ```bash
   python -m venv venv
   venv\Scripts\activate  # Windows
   source venv/bin/activate  # Mac/Linux
   ```

5. **Jalankan setup otomatis:**
   ```bash
   python setup_vscode.py
   ```

6. **Buat file .env dan isi API key:**
   ```
   FOOTBALL_DATA_ORG_KEY=your_actual_api_key_here
   ```

7. **Jalankan aplikasi:**
   ```bash
   streamlit run app.py --server.port 5000
   ```

## 🔑 PENTING: API KEY

1. **Daftar di https://www.football-data.org/**
2. **Klik "Get your free API key"**
3. **Konfirmasi email dan login**
4. **Copy API key ke file .env**

## ⚠️ TROUBLESHOOTING

Jika ada error:
- Jalankan `python setup_vscode.py` untuk diagnosis
- Pastikan Python 3.8+ terinstall
- Aktifkan virtual environment sebelum install
- Cek file .env berisi API key yang benar

## 📞 SUPPORT

Jika masih ada masalah:
1. Cek semua file sudah didownload
2. Struktur folder sesuai dengan checklist
3. Virtual environment aktif
4. API key valid dan terdaftar