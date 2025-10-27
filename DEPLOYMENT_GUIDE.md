# 🚀 Streamlit App Deployment Guide

## Quick Start - Deploy to Streamlit Cloud (FREE)

### Prerequisites
- GitHub account
- Your code pushed to a GitHub repository

---

## 📦 Files Needed for Deployment

Your app will automatically deploy with these files:

```
landai_qa_coaching_bot_optimized.py  ← Main app (already configured!)
requirements_streamlit.txt           ← Dependencies
.streamlit/config.toml              ← UI settings
```

**Note:** Do NOT commit `.streamlit/secrets.toml` to Git - it contains API keys!

---

## 🔧 Step-by-Step Deployment

### 1. Push to GitHub

```bash
# Add to .gitignore (to protect secrets)
echo ".streamlit/secrets.toml" >> .gitignore
echo "venv/" >> .gitignore
echo "__pycache__/" >> .gitignore
echo "*.pyc" >> .gitignore

# Commit and push
git add landai_qa_coaching_bot_optimized.py
git add requirements_streamlit.txt
git add .streamlit/config.toml
git add .gitignore
git commit -m "Add Streamlit coaching app for deployment"
git push origin main
```

### 2. Deploy on Streamlit Cloud

1. **Go to:** https://share.streamlit.io/

2. **Sign in** with your GitHub account

3. **Click "New app"**

4. **Configure:**
   - **Repository:** Select your repo (e.g., `yourusername/qa-bot`)
   - **Branch:** `main`
   - **Main file path:** `landai_qa_coaching_bot_optimized.py`
   - **App URL:** Choose a custom name (e.g., `landai-qa-coaching`)

5. **Add Secrets** (Click "Advanced settings" → "Secrets"):
   ```toml
   [supabase]
   url = "https://yeflauigtjsexadhiqiq.supabase.co"
   key = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InllZmxhdWlndGpzZXhhZGhpcWlxIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NDgzNjA0MjMsImV4cCI6MjA2MzkzNjQyM30.W-6c2_rsa1TRM1S7pvEy4vT1kGx7bevpqRKgbHio1gE"

   [openai]
   api_key = "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"
   ```

6. **Click "Deploy"** 🚀

7. **Wait 2-3 minutes** for initial deployment

8. **Your app is live!** 🎉
   - URL: `https://your-app-name.streamlit.app`
   - Share this URL with your team

---

## 🔄 Auto-Updates

After initial deployment, any changes you push to GitHub will automatically redeploy:

```bash
# Make changes to your app
vim landai_qa_coaching_bot_optimized.py

# Push to GitHub
git add landai_qa_coaching_bot_optimized.py
git commit -m "Update coaching metrics"
git push origin main

# Streamlit Cloud will auto-deploy within 1-2 minutes!
```

---

## 🎨 Customization

### Change App Theme
Edit `.streamlit/config.toml`:
```toml
[theme]
primaryColor = "#4CAF50"      # Green accent color
backgroundColor = "#0E1117"    # Dark background
secondaryBackgroundColor = "#1E1E1E"
textColor = "#FAFAFA"
```

### Change App Title
Edit the app file (line ~600):
```python
st.set_page_config(
    page_title="Your Company - QA Coaching",
    page_icon="🎯",
    layout="wide"
)
```

---

## 🔒 Security Best Practices

✅ **DO:**
- Use Streamlit Secrets for API keys
- Add `.streamlit/secrets.toml` to `.gitignore`
- Use environment variables for sensitive data
- Keep your GitHub repo private if needed

❌ **DON'T:**
- Commit API keys to Git
- Share your secrets.toml file
- Use production API keys in development

---

## 📊 Monitoring & Logs

**View Logs:**
1. Go to your app on Streamlit Cloud
2. Click "Manage app" (bottom right)
3. Click "Logs" to see real-time logs
4. Check for errors or performance issues

**Restart App:**
- Click "Reboot app" if needed
- Or push a commit to trigger redeployment

---

## 💰 Cost Breakdown

### Streamlit Cloud (FREE Tier):
- ✅ **1 private app** (or unlimited public)
- ✅ **1 GB RAM**
- ✅ **1 CPU core**
- ✅ **Unlimited viewers**
- ✅ **Auto-SSL/HTTPS**
- ✅ **Custom subdomain**

**Need more?** Upgrade to Streamlit Cloud Pro ($20/month):
- 3 private apps
- More resources
- Priority support

### Your Current Costs:
- **Streamlit Cloud:** $0/month (FREE tier)
- **Supabase:** $0/month (your current tier)
- **OpenAI API:** Pay per use (~$0.002 per chat message)

**Total:** ~$0-5/month depending on OpenAI usage

---

## 🐛 Troubleshooting

### "ModuleNotFoundError"
- Check `requirements_streamlit.txt` has all dependencies
- Redeploy with updated requirements

### "Connection timeout"
- Increase timeout in app settings
- Check Supabase is accessible from Streamlit Cloud

### "Secrets not found"
- Verify secrets are added in Streamlit Cloud dashboard
- Check TOML formatting (no quotes around section headers)

### App is slow
- Check data query optimization
- Review cache settings (`@st.cache_data`)
- Consider pagination for large datasets

---

## 📞 Support

- **Streamlit Docs:** https://docs.streamlit.io/
- **Community Forum:** https://discuss.streamlit.io/
- **GitHub Issues:** https://github.com/streamlit/streamlit/issues

---

## 🎯 What's Next?

After deployment, you can:

1. **Share the URL** with your team
2. **Set up custom domain** (in Streamlit settings)
3. **Add authentication** (Streamlit supports OAuth)
4. **Monitor usage** (view analytics in dashboard)
5. **Scale up** if needed (upgrade to Pro)

---

**Your app is ready to deploy! 🚀**

Just follow steps 1-2 above and you'll be live in minutes!

