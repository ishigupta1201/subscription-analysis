# Subscription Analytics Platform

A modern, AI-powered analytics tool for subscription and payment data. Use natural language to get insights, graphs, and reports from your subscription business—either from the command line or via Claude Desktop (MCP protocol).

---

## Features

- Natural language analytics (Gemini AI)
- Context-aware follow-up queries
- Feedback-driven improvement (CLI)
- Auto-generated graphs (PNG)
- Secure, remote API access
- Claude Desktop (MCP) integration

---

## Requirements

- Python 3.8+
- [Gemini API key](https://ai.google.dev/)
- API key for the subscription analytics server
- Internet connection

---

## Quick Start (Command Line)

1. **Clone or download the client code:**

   - Place the `client/` folder on your machine.

2. **Install dependencies:**

   ```bash
   cd client
   pip install -r requirements.txt
   ```

3. **Configure your credentials:**

   - Edit `client/config.json` (create if missing):
     ```json
     {
       "GOOGLE_API_KEY": "your-gemini-api-key",
       "API_KEY_1": "your-subscription-analytics-api-key",
       "SUBSCRIPTION_API_URL": "https://subscription-analytics.onrender.com"
     }
     ```
   - Get your Gemini API key from https://ai.google.dev/
   - Get your API key from your admin or the server owner.

4. **Run a query:**

   ```bash
   python universal_client.py "How many new subscriptions this month?"
   python universal_client.py "Show payment success rate for last 30 days with graph"
   ```

   - You will be prompted for feedback after each result. Your suggestions help improve future answers.

5. **Interactive mode:**
   ```bash
   python universal_client.py
   ```
   - Type queries interactively. Type `exit` to quit.

---

## Quick Start (Claude Desktop / MCP Client)

1. **Install dependencies (if not already):**

   ```bash
   cd client
   pip install -r requirements.txt
   ```

2. **Configure your credentials:**

   - As above, ensure `client/config.json` is set up.

3. **Run the MCP server:**

   ```bash
   python mcp_client.py
   ```

   - This will start the MCP server and wait for connections from Claude Desktop or any MCP-compatible client.

4. **Configure Claude Desktop:**
   - Add to your Claude Desktop config (usually at `~/Library/Application Support/Claude/claude_desktop_config.json`):
     ```json
     {
       "servers": {
         "subscription-analytics": {
           "command": "python",
           "args": ["/absolute/path/to/client/mcp_client.py"],
           "env": {
             "SUBSCRIPTION_API_URL": "https://subscription-analytics.onrender.com",
             "API_KEY_1": "your-subscription-analytics-api-key",
             "GOOGLE_API_KEY": "your-gemini-api-key"
           }
         }
       }
     }
     ```
   - Restart Claude Desktop if needed.
   - Now you can ask Claude: _"Show me our subscription analytics for the last month"_

---

## Example Queries (CLI or Claude)

```bash
python universal_client.py "Compare subscription performance for 7 days vs 30 days"
python universal_client.py "How many new subscriptions did we get this month?"
python universal_client.py "What's our retention rate for the last 2 weeks?"
python universal_client.py "Show me database status and recent subscription summary"
```

---

## Project Structure

```
subscription_analysis/
├── client/
│   ├── universal_client.py   # Main CLI client
│   ├── mcp_client.py         # MCP/Claude Desktop server
│   ├── config_manager.py     # Config loader
│   ├── requirements.txt      # Client dependencies
│   ├── config.json           # Your credentials
│   └── generated_graphs/     # Auto-generated graphs
├── api_server.py             # (Server, not needed for client use)
├── requirements.txt          # (Server requirements)
├── ...
```

---

## Troubleshooting

- **Missing API key or Gemini key:** Edit `client/config.json` and add the correct values.
- **SSL errors on Mac:** Run `/Applications/Python\ 3.x/Install\ Certificates.command`.
- **Cannot connect to server:** Check your internet, API URL, and API key.
- **Graphs not generated:** Ensure `matplotlib` is installed and `generated_graphs/` is writable.

---

## Support

- For technical details, see `How everything works.md`.
- For API docs, visit `https://subscription-analytics.onrender.com/docs`.
