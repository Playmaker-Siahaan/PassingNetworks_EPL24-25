
# 🤖 AI Features Setup Guide

## Overview
Aplikasi ini sekarang dilengkapi dengan fitur AI canggih yang mengintegrasikan multiple API untuk analisis mendalam:

### AI Features yang Tersedia:
1. **🔮 AI Match Prediction** - Prediksi hasil pertandingan dengan ML
2. **✍️ AI Content Generator** - Generate laporan, script, dan analisis
3. **🧠 AI Enhanced Insights** - Analisis mendalam dengan multiple AI models

## API Keys yang Diperlukan

### 1. OpenAI API (GPT-3.5)
- **Fungsi**: Analisis taktik mendalam, content generation
- **Cara Mendapat**: https://platform.openai.com/api-keys
- **Biaya**: Pay-per-use, mulai $0.002 per 1K tokens

### 2. Stability AI API
- **Fungsi**: Generate visualisasi formasi taktik
- **Cara Mendapat**: https://platform.stability.ai/account/keys
- **Biaya**: Pay-per-use, mulai $0.002 per image

### 3. Hugging Face API
- **Fungsi**: Player insights, sentiment analysis
- **Cara Mendapat**: https://huggingface.co/settings/tokens
- **Biaya**: Free tier tersedia

## Setup Instructions

### Method 1: Replit Secrets (Recommended)
1. Buka Replit project Anda
2. Klik "Secrets" di sidebar kiri
3. Tambahkan secrets berikut:
   - `OPENAI_API_KEY`: Your OpenAI API key
   - `STABILITY_API_KEY`: Your Stability AI API key  
   - `HUGGINGFACE_TOKEN`: Your Hugging Face token

### Method 2: Environment Variables
1. Copy `.env.example` to `.env`
2. Replace placeholder values dengan API keys Anda
3. Restart aplikasi

## Testing AI Features

### 1. Test AI Match Prediction
```python
# Di sidebar, aktifkan "🔮 AI Match Prediction"
# Pilih tim dan lihat prediksi AI
```

### 2. Test AI Content Generator
```python
# Di sidebar, aktifkan "✍️ AI Content Generator"
# Generate match reports, highlight scripts, dll
```

### 3. Test AI Enhanced Insights
```python
# Di sidebar, aktifkan "🧠 AI Enhanced Insights"
# Lihat analisis mendalam dan trend prediction
```

## Fallback Mode
Jika API keys tidak tersedia, aplikasi akan tetap berjalan dengan:
- Analisis pre-generated yang berkualitas
- Insights berdasarkan data FPL
- Prediksi menggunakan model internal

## Troubleshooting

### Error "API Key not found"
- Pastikan API keys sudah ditambahkan ke Replit Secrets
- Check spelling dan formatting API keys
- Restart aplikasi setelah menambah secrets

### Error "Rate limit exceeded"
- Tunggu beberapa menit sebelum mencoba lagi
- Upgrade API plan jika diperlukan
- Gunakan fallback mode sementara

### Error "Invalid API response"
- Check internet connection
- Verify API keys masih valid
- Check API service status

## Best Practices

### 1. API Usage Optimization
- Gunakan AI features secara selektif
- Cache hasil untuk mengurangi API calls
- Monitor usage melalui dashboard API

### 2. Content Quality
- Verify AI-generated content sebelum sharing
- Combine AI insights dengan domain knowledge
- Use AI sebagai enhancement, bukan replacement

### 3. Security
- Jangan share API keys
- Use environment variables/secrets
- Regularly rotate API keys

## Advanced Features

### Custom AI Models
Anda bisa menambahkan model AI custom dengan mengmodifikasi `AIServices` class:

```python
def custom_analysis(self, data):
    # Your custom AI logic here
    return analysis_result
```

### Integration dengan APIs Lain
Mudah untuk menambahkan API lain seperti:
- Google Cloud AI
- Azure Cognitive Services
- AWS Comprehend

## Support
Jika mengalami masalah:
1. Check console logs untuk error details
2. Verify API keys dan network connection
3. Test dengan fallback mode
4. Contact support dengan error logs

---

**Note**: Semua AI features bersifat optional. Aplikasi tetap fully functional tanpa API keys.
