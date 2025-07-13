
# 🚀 Quick AI Setup Guide

## 1. Get API Keys (5 minutes)

### OpenAI (Recommended)
1. Go to https://platform.openai.com/api-keys
2. Create new secret key
3. Copy and save securely

### Stability AI (Optional)
1. Go to https://platform.stability.ai/account/keys
2. Create new API key
3. Copy and save securely

### Hugging Face (Free)
1. Go to https://huggingface.co/settings/tokens
2. Create new token (read access)
3. Copy and save securely

## 2. Add to Replit Secrets (2 minutes)

1. In your Replit project, click **Secrets** in sidebar
2. Add these secrets:
   - `OPENAI_API_KEY` = your_openai_key
   - `STABILITY_API_KEY` = your_stability_key
   - `HUGGINGFACE_TOKEN` = your_huggingface_token

## 3. Test Setup (1 minute)

1. Run the validation script:
   ```bash
   python validate_ai_setup.py
   ```

2. Or check in app sidebar:
   - Click "🔍 Check API Status"
   - Should show ✅ for configured APIs

## 4. Enable AI Features

In the app sidebar, check these boxes:
- ✅ 🔮 AI Match Prediction
- ✅ ✍️ AI Content Generator  
- ✅ 🧠 AI Enhanced Insights

## 5. Start Using AI Features!

### AI Match Prediction
- Select teams → Get ML-powered predictions
- See win probabilities and predicted scores

### AI Content Generator
- Generate match reports automatically
- Create highlight scripts
- Statistical analysis with AI insights

### AI Enhanced Insights
- Deep tactical analysis
- Player performance insights
- Trend predictions

## Troubleshooting

### "API Key not found"
- Check Replit Secrets spelling
- Restart the app after adding secrets

### "Rate limit exceeded"
- Wait a few minutes
- Consider upgrading API plans

### Still not working?
- App works fine without AI keys (fallback mode)
- All core CGAN features remain available

---

**Total Setup Time: ~8 minutes**
**Cost: OpenAI ~$0.50/month for typical usage**
