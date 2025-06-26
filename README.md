# Subscription Analytics Platform

A comprehensive subscription analytics platform with API server, universal client, and MCP integration for AI assistants. **Now live and accessible remotely!**

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   API Server    │    │ Universal Client│    │   MCP Client    │
│   (Render)      │    │                 │    │                 │
│ • FastAPI       │◄───┤ • HTTP Client   │◄───┤ • MCP Wrapper   │
│ • PostgreSQL    │    │ • Gemini NLP    │    │ • Claude Server │
│ • Analytics     │    │ • Graph Gen     │    │ • Tool Bridge   │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## 🌐 Live Server

**🚀 API Server:** `https://subscription-analytics.onrender.com`

**✅ Status:** Live and operational  
**🔐 Authentication:** API key required  
**📊 Database:** Railway PostgreSQL with full subscription data

## 🚀 Components

### 1. API Server (Deployed on Render)
- **FastAPI** backend with PostgreSQL database
- **Authentication** with API keys
- **Analytics tools** for subscription and payment data
- **RESTful API** endpoints
- **Auto-deployment** from GitHub
- **Docker containerization** for consistent deployment

### 2. Universal Client (Remote Access)
- **HTTP client** for remote API server
- **Gemini AI integration** for natural language queries
- **Graph generation** with automatic visualization
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
curl -O https://raw.githubusercontent.com/Abhyuday-Gupta912/subscription-analytics/main/client/config_manager.py
curl -O https://raw.githubusercontent.com/Abhyuday-Gupta912/subscription-analytics/main/client/requirements.txt
curl -O https://raw.githubusercontent.com/Abhyuday-Gupta912/subscription-analytics/main/client/config.json
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
When you run the universal client for the first time, it will ask you for the url, key and gemini api key which are as follows:
```
SUBSCRIPTION_API_URL=https://subscription-analytics.onrender.com
SUBSCRIPTION_API_KEY=sub_analytics_mhHT-jo1FcowxIKbqf3hAAMyUrRHKODxXhcd_PCHT5k
GEMINI_API_KEY=your_gemini_api_key_here
```

5. **Get your Gemini API key:**
   - **Gemini API key:** Visit https://ai.google.dev/ to get your API key
   - Replace `your_gemini_api_key_here` with your actual Gemini API key when prompted

6. **Start using:**
```bash
python universal_client.py "database status"
python universal_client.py "show me subscription summary for last 30 days"
python universal_client.py "compare 7 days vs 14 days performance with graphs"
```

### Option 2: Deploy Your Own Server

1. **Fork the repository:**
   - Go to your repository URL
   - Click "Fork"

2. **Deploy to Render:**
   - Connect your GitHub repo to Render
   - Use the included `render.yaml` configuration
   - Set environment variables (see Configuration section)
   - Render will auto-deploy using the `Dockerfile`

3. **Set up your database:**
   - Add PostgreSQL service in Render
   - Migrate your data using provided scripts
   - Update client configuration

## 🔧 Configuration

### Live Server Access
```env
SUBSCRIPTION_API_URL=https://subscription-analytics.onrender.com
SUBSCRIPTION_API_KEY=sub_analytics_mhHT-jo1FcowxIKbqf3hAAMyUrRHKODxXhcd_PCHT5k
GEMINI_API_KEY=your_gemini_api_key_here
```

### Your Own Server Deployment
```env
# Render Environment Variables
DATABASE_URL=postgresql://username:password@hostname:port/database
API_KEY_1=your_secure_api_key_1
API_KEY_2=your_secure_api_key_2
GEMINI_API_KEY=your_gemini_api_key
```

## 📊 Features

- **🤖 Natural Language Queries**: Ask questions in plain English using Gemini AI
- **📈 Subscription Analytics**: Track new subscriptions, churn, retention rates
- **💳 Payment Analytics**: Monitor success rates, revenue, failed transactions
- **📊 Auto Graph Generation**: Automatic visualization in `generated_graphs/` folder
- **⏱️ Time Period Comparisons**: Compare metrics across different timeframes
- **🗄️ Database Monitoring**: Check connection status and basic statistics
- **🔗 AI Integration**: Works with Claude Desktop via MCP protocol
- **🎨 Beautiful Formatting**: Human-readable output with emojis and tables
- **🔒 Secure Access**: API key authentication
- **🌐 Remote Access**: No local server setup required
- **🐳 Docker Ready**: Containerized deployment with `Dockerfile`

## 🤖 Example Queries

```bash
# Subscription analysis
python universal_client.py "Compare subscription performance for 7 days vs 30 days"
python universal_client.py "How many new subscriptions did we get this month?"
python universal_client.py "What's our retention rate for the last 2 weeks?"

# Payment analysis with graphs
python universal_client.py "What's our payment success rate for the last 2 weeks with chart"
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
        "SUBSCRIPTION_API_URL": "https://subscription-analytics.onrender.com",
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
- ✅ Reliable Render hosting
- ✅ Just need client setup

### Option 2: Render Deployment
1. Fork repository
2. Connect to Render
3. Use included `render.yaml` configuration
4. Set environment variables
5. Auto-deployment from GitHub using `Dockerfile`

### Option 3: Docker
```bash
# Build and run with Docker
docker build -t subscription-analytics .
docker run -p 8000:8000 subscription-analytics

# Or use docker-compose if available
docker-compose up -d
```

## 🛠️ Development Setup

```bash
# Clone repository
git clone https://github.com/Abhyuday-Gupta912/subscription-analytics.git
cd subscription-analytics

# Server setup
pip install -r requirements.txt
python api_server.py

# Client setup
cd client
pip install -r requirements.txt
python universal_client.py
```

## 📁 Project Structure

```
subscription-analytics/
├── api_server.py              # Main FastAPI application
├── requirements.txt           # Server dependencies
├── Dockerfile                 # Docker configuration
├── render.yaml               # Render deployment config
├── client/                   # Client Applications
│   ├── universal_client.py   # Main client with Gemini AI
│   ├── mcp_client.py        # MCP wrapper for Claude Desktop
│   ├── config_manager.py    # Configuration management
│   ├── requirements.txt     # Client dependencies
│   ├── config.json          # Client configuration
│   ├── .env                 # Environment variables
│   └── generated_graphs/    # Auto-generated visualization graphs
├── model/                   # Data models and schemas
├── How everything works.md  # Technical documentation
├── README.md
├── .gitignore
├── .dockerignore
└── .env                     # Server environment variables
```

## 🔒 Security & Access

- **🔑 API Key Authentication**: All requests require valid API key
- **🔐 SSL/TLS Encryption**: All communications encrypted via Render
- **🚫 No Data Storage**: Client doesn't store sensitive data locally
- **⚡ Rate Limiting**: API includes rate limiting for stability
- **🛡️ Input Validation**: All inputs validated and sanitized
- **🐳 Secure Deployment**: Containerized deployment with Render

## 🌍 Access from Anywhere

**Your analytics are now accessible from:**
- ✅ Any computer with Python
- ✅ Claude Desktop (via MCP)
- ✅ Command line interface
- ✅ Interactive mode
- ✅ Custom applications (via API)
- ✅ Auto-generated graphs and visualizations

## 📞 API Endpoints

Base URL: `https://subscription-analytics.onrender.com`

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
curl https://subscription-analytics.onrender.com/health

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

### Graph Generation Issues
- Check that `generated_graphs/` folder exists in client directory
- Ensure proper write permissions
- Verify matplotlib and visualization dependencies are installed

## 📖 Additional Documentation

- **Technical Details**: See `How everything works.md`
- **API Documentation**: Visit `https://subscription-analytics.onrender.com/docs`
- **Configuration Guide**: Check `client/config.json` for client settings
