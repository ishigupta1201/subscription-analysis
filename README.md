# Subscription Analytics Platform

A comprehensive subscription analytics platform with API server, universal client, and MCP integration for AI assistants. **Now live and accessible remotely!**

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   API Server    │    │ Universal Client│    │   MCP Client    │
│   (Railway)     │    │                 │    │                 │
│ • FastAPI       │◄───┤ • HTTP Client   │◄───┤ • MCP Wrapper   │
│ • MySQL         │    │ • Gemini NLP    │    │ • Claude Server │
│ • Analytics     │    │ • Formatting    │    │ • Tool Bridge   │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## 🌐 Live Server

**🚀 API Server:** `https://subscription-analysis-production.up.railway.app`

**✅ Status:** Live and operational  
**🔐 Authentication:** API key required  
**📊 Database:** Railway MySQL with full subscription data

## 🚀 Components

### 1. API Server (Deployed on Railway)
- **FastAPI** backend with Railway MySQL database
- **Authentication** with API keys
- **Analytics tools** for subscription and payment data
- **RESTful API** endpoints
- **Auto-deployment** from GitHub

### 2. Universal Client (Remote Access)
- **HTTP client** for remote API server
- **Gemini AI integration** for natural language queries
- **SSL certificate handling** for secure connections
- **Beautiful formatting** of results
- **Interactive and CLI modes**

### 3. MCP Client 
- **MCP (Model Context Protocol)** wrapper
- **Claude Desktop integration**
- **Tool server** functionality
- **Bridge** between AI assistants and analytics

## 📦 Quick Start for Users

### Option 1: Use the Live Server (Recommended)

**No server setup required!** Just use the remote API:

1. **Download the client:**
```bash
# Create project folder
mkdir subscription-analytics-client
cd subscription-analytics-client

# Download client files
curl -O https://raw.githubusercontent.com/Abhyuday-Gupta912/subscription-analytics/main/client/universal_client.py
curl -O https://raw.githubusercontent.com/Abhyuday-Gupta912/subscription-analytics/main/client/requirements.txt
```

2. **Set up virtual environment:**
```bash
# Create virtual environment
python3 -m venv analytics-env

# Activate it
source analytics-env/bin/activate  # Mac/Linux
# analytics-env\Scripts\activate   # Windows
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

4. **Configure access:**
```bash
# Create .env file
cat > .env << 'EOF'
SUBSCRIPTION_API_URL=https://subscription-analysis-production.up.railway.app
SUBSCRIPTION_API_KEY=sub_analytics_mhHT-jo1FcowxIKbqf3hAAMyUrRHKODxXhcd_PCHT5k
GEMINI_API_KEY=your_gemini_api_key_here
EOF
```

5. **Get your Gemini API key:**
   - Visit: https://ai.google.dev/
   - Create account and get API key
   - Replace `your_gemini_api_key_here` in `.env`

6. **Start using:**
```bash
python universal_client.py "database status"
python universal_client.py "show me subscription summary for last 30 days"
python universal_client.py "compare 7 days vs 14 days performance"
```

### Option 2: Deploy Your Own Server

1. **Fork the repository:**
   - Go to: https://github.com/Abhyuday-Gupta912/subscription-analytics
   - Click "Fork"

2. **Deploy to Railway:**
   - Connect your GitHub repo to Railway
   - Set environment variables (see Configuration section)
   - Railway will auto-deploy

3. **Set up your database:**
   - Add Railway MySQL service
   - Migrate your data using provided scripts
   - Update client configuration

## 🔧 Configuration

### Live Server Access
```env
SUBSCRIPTION_API_URL=https://subscription-analysis-production.up.railway.app
SUBSCRIPTION_API_KEY=sub_analytics_mhHT-jo1FcowxIKbqf3hAAMyUrRHKODxXhcd_PCHT5k
GEMINI_API_KEY=your_gemini_api_key_here
```

### Your Own Server Deployment
```env
# Railway Environment Variables
DB_HOST=yamanote.proxy.rlwy.net
DB_PORT=50495
DB_USER=root
DB_PASSWORD=your_railway_mysql_password
DB_NAME=railway
API_KEY_1=your_secure_api_key_1
API_KEY_2=your_secure_api_key_2
```

## 📊 Features

- **🤖 Natural Language Queries**: Ask questions in plain English using Gemini AI
- **📈 Subscription Analytics**: Track new subscriptions, churn, retention rates
- **💳 Payment Analytics**: Monitor success rates, revenue, failed transactions
- **⏱️ Time Period Comparisons**: Compare metrics across different timeframes
- **🗄️ Database Monitoring**: Check connection status and basic statistics
- **🔗 AI Integration**: Works with Claude Desktop via MCP protocol
- **🎨 Beautiful Formatting**: Human-readable output with emojis and tables
- **🔒 Secure Access**: API key authentication
- **🌐 Remote Access**: No local server setup required

## 🤖 Example Queries

```bash
# Subscription analysis
python universal_client.py "Compare subscription performance for 7 days vs 30 days"
python universal_client.py "How many new subscriptions did we get this month?"
python universal_client.py "What's our retention rate for the last 2 weeks?"

# Payment analysis  
python universal_client.py "What's our payment success rate for the last 2 weeks?"
python universal_client.py "Show me revenue trends and failed transactions"
python universal_client.py "Compare payment performance across different periods"

# General monitoring
python universal_client.py "Show me database status and recent subscription summary"
python universal_client.py "Give me a comprehensive analytics overview"
```

## 🔌 Claude Desktop Integration

**Add to your Claude Desktop config** (`~/Library/Application Support/Claude/claude_desktop_config.json`):

```json
{
  "servers": {
    "subscription-analytics": {
      "command": "python",
      "args": ["/path/to/client/mcp_client.py", "--mcp"],
      "env": {
        "SUBSCRIPTION_API_URL": "https://subscription-analysis-production.up.railway.app",
        "SUBSCRIPTION_API_KEY": "sub_analytics_mhHT-jo1FcowxIKbqf3hAAMyUrRHKODxXhcd_PCHT5k",
        "GEMINI_API_KEY": "your_gemini_api_key"
      }
    }
  }
}
```

Then ask Claude directly: *"Show me our subscription analytics for the last month"*

## 🚀 Deployment Options

### Option 1: Use Live Server (Easiest)
- ✅ No setup required
- ✅ Always updated
- ✅ Reliable hosting
- ✅ Just need client setup

### Option 2: Railway Deployment
1. Fork repository
2. Connect to Railway
3. Set environment variables
4. Auto-deployment from GitHub

### Option 3: Docker
```bash
docker-compose up -d
```

## 🛠️ Development Setup

```bash
# Clone repository
git clone https://github.com/Abhyuday-Gupta912/subscription-analytics.git
cd subscription-analytics

# Server setup
cd server
pip install -r requirements.txt
python api_server.py

# Client setup
cd ../client
pip install -r requirements.txt
python universal_client.py
```

## 📁 Project Structure

```
subscription-analytics/
├── api_server.py           # Main FastAPI application (moved to root for Railway)
├── requirements.txt        # Server dependencies
├── client/                 # Client Applications
│   ├── universal_client.py # Main client with Gemini AI
│   ├── mcp_client.py      # MCP wrapper for Claude Desktop
│   ├── requirements.txt   # Client dependencies
│   └── .env.example       # Environment template
├── docs/                  # Documentation
├── README.md
├── .gitignore
└── LICENSE
```

## 🔒 Security & Access

- **🔑 API Key Authentication**: All requests require valid API key
- **🔐 SSL/TLS Encryption**: All communications encrypted
- **🚫 No Data Storage**: Client doesn't store sensitive data locally
- **⚡ Rate Limiting**: API includes rate limiting for stability
- **🛡️ Input Validation**: All inputs validated and sanitized

## 🌍 Access from Anywhere

**Your analytics are now accessible from:**
- ✅ Any computer with Python
- ✅ Claude Desktop (via MCP)
- ✅ Command line interface
- ✅ Interactive mode
- ✅ Custom applications (via API)

## 📞 API Endpoints

Base URL: `https://subscription-analysis-production.up.railway.app`

- `GET /health` - Health check
- `GET /` - API status  
- `POST /execute` - Execute analytics tools
- `GET /tools` - List available tools

## 🆘 Troubleshooting

### SSL Certificate Issues (Mac)
```bash
# Run Python certificate installer
/Applications/Python\ 3.x/Install\ Certificates.command

# Or disable SSL verification temporarily
export SSL_CERT_FILE=""
```

### Network Issues
```bash
# Test connectivity
curl https://subscription-analysis-production.up.railway.app/health

# Try different network (mobile hotspot)
# Check corporate firewall settings
```

### Environment Variables
```bash
# Verify environment variables are loaded
python3 -c "
from dotenv import load_dotenv
import os
load_dotenv()
print('URL:', os.getenv('SUBSCRIPTION_API_URL'))
print('API Key:', os.getenv('SUBSCRIPTION_API_KEY'))
"
```
