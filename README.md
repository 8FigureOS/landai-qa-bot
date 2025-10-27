# LandAI QA Coaching Bot 🎯

An AI-powered coaching dashboard built with Streamlit for analyzing call quality and providing personalized coaching insights.

## Features

✨ **Real-time Analytics**
- Agent performance metrics
- QA score tracking and trends
- Call volume statistics
- Filter by date, campaign, and score range

🤖 **AI-Powered Coaching**
- Interactive chatbot for coaching insights
- Personalized improvement suggestions
- Analysis of key strengths and areas for improvement
- Suggested coaching questions

📊 **Comprehensive Dashboard**
- Beautiful dark-themed UI
- Agent selection and comparison
- Score distribution visualization
- Campaign performance tracking

## Quick Start

### Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run landai_qa_coaching_bot_optimized.py
```

### Deployment

See [DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md) for detailed deployment instructions.

**Recommended:** Deploy for FREE on [Streamlit Cloud](https://share.streamlit.io/)

## Configuration

The app uses Streamlit Secrets for secure API key management:

1. Create `.streamlit/secrets.toml` (NOT committed to Git)
2. Add your credentials:

```toml
[supabase]
url = "your-supabase-url"
key = "your-supabase-key"

[openai]
api_key = "your-openai-key"
```

## Technology Stack

- **Frontend:** Streamlit
- **Database:** Supabase (PostgreSQL)
- **AI:** OpenAI GPT-4
- **Language:** Python 3.9+

## Security

- ✅ API keys stored in Streamlit Secrets
- ✅ Not committed to version control
- ✅ Secure communication with HTTPS
- ✅ Environment-based configuration

## License

Proprietary - 8 Figure Agency

---

**Built with ❤️ for coaching excellence**
