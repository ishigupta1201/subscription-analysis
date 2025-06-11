# Subscription Analytics Platform

A comprehensive subscription analytics platform with API server, universal client, and MCP integration for AI assistants.

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   API Server    │    │ Universal Client│    │   MCP Client    │
│                 │    │                 │    │                 │
│ • FastAPI       │◄───┤ • HTTP Client   │◄───┤ • MCP Wrapper   │
│ • MySQL         │    │ • Gemini NLP    │    │ • Claude Server │
│ • Analytics     │    │ • Formatting    │    │ • Tool Bridge   │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## 🚀 Components

### 1. API Server (`server/`)

- **FastAPI** backend with MySQL database
- **Authentication** with API keys
- **Analytics tools** for subscription and payment data
- **RESTful API** endpoints

### 2. Universal Client (`client/`)

- **HTTP client** for API server
- **Gemini AI integration** for natural language queries
- **Beautiful formatting** of results
- **Interactive and CLI modes**

### 3. MCP Client (`client/`)

- **MCP (Model Context Protocol)** wrapper
- **Claude Desktop integration**
- **Tool server** functionality
- **Bridge** between AI assistants and analytics

## 📦 Quick Start

### Server Deployment (Railway)

1. **Deploy to Railway:**

   ```bash
   # Connect your GitHub repo to Railway
   # Railway will auto-detect and deploy the server
   ```

2. **Environment Variables:**
   ```env
   DB_HOST=your_mysql_host
   DB_NAME=SUBS_STAGING
   DB_USER=your_db_user
   DB_PASSWORD=your_db_password
   API_KEY_1=your_secure_api_key
   API_KEY_2=your_second_api_key
   ```

### Client Usage

1. **Install dependencies:**

   ```bash
   cd client
   pip install -r requirements.txt
   ```

2. **Configure environment:**

   ```bash
   cp .env.example .env
   # Edit .env with your API endpoint and keys
   ```

3. **Run interactive client:**

   ```bash
   python universal_client.py
   ```

4. **Run MCP server for Claude:**
   ```bash
   python mcp_client.py --mcp
   ```

## 🔧 Configuration

### Server Environment

```env
DB_HOST=localhost
DB_NAME=SUBS_STAGING
DB_USER=root
DB_PASSWORD=your_password
API_KEY_1=sub_analytics_secure_key_1
API_KEY_2=sub_analytics_secure_key_2
```

### Client Environment

```env
SUBSCRIPTION_API_URL=https://your-railway-app.railway.app
SUBSCRIPTION_API_KEY=your_api_key_here
GEMINI_API_KEY=your_gemini_api_key
```

## 📊 Features

- **Natural Language Queries**: Ask questions in plain English
- **Subscription Analytics**: Track new subscriptions, churn, retention
- **Payment Analytics**: Monitor success rates, revenue, failures
- **Time Period Comparisons**: Compare metrics across different timeframes
- **Database Monitoring**: Check connection status and basic stats
- **AI Integration**: Works with Claude Desktop via MCP
- **Beautiful Formatting**: Human-readable output with emojis and tables

## 🤖 Example Queries

```
"Compare subscription performance for 7 days vs 30 days"
"What's our payment success rate for the last 2 weeks?"
"Show me database status and recent subscription summary"
"How many new subscriptions did we get this month?"
```

## 🔌 MCP Integration

Add to your Claude Desktop config:

```json
{
  "servers": {
    "subscription-analytics": {
      "command": "python",
      "args": ["/path/to/client/mcp_client.py", "--mcp"]
    }
  }
}
```

## 🚀 Deployment

### Railway (Recommended for Server)

1. Connect GitHub repository
2. Set environment variables
3. Deploy automatically

### Docker

```bash
docker-compose up -d
```

## 📁 Project Structure

```
subscription-analytics/
├── server/              # API Server
│   ├── api_server.py   # Main FastAPI application
│   ├── requirements.txt
│   └── .env.example
├── client/              # Client Applications
│   ├── universal_client.py  # Main client with Gemini
│   ├── mcp_client.py       # MCP wrapper for AI assistants
│   ├── requirements.txt
│   └── .env.example
├── docs/               # Documentation
├── README.md
└── .gitignore
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📄 License

MIT License - see LICENSE file for details

## 🆘 Support

- **Issues**: GitHub Issues
- **Documentation**: `/docs` folder
- **Examples**: See example queries above
