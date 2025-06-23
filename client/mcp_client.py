#!/usr/bin/env python3
# full_tools_mcp_client.py - MCP client exposing all backend tools individually

import asyncio
import aiohttp
import logging
import sys
import os
import json
import ssl
import certifi
from typing import Any, Dict, List
from pathlib import Path

# MCP Imports
try:
    from mcp.server import Server, InitializationOptions
    from mcp.types import TextContent, Tool
    import mcp.server.stdio
except ImportError:
    print("❌ MCP not installed. Run: pip install mcp", file=sys.stderr)
    sys.exit(1)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("full-tools-mcp")

class FullToolsMCPClient:
    def __init__(self):
        print("🔧 Initializing Full Tools MCP Client...", file=sys.stderr)
        self.server = Server("subscription-analytics")
        self.config = None
        self.session = None
        
        # Define ALL backend tools for Claude Desktop
        self.tools = [
            Tool(
                name="get_subscriptions_in_last_days",
                description="Get subscription statistics for the last N days",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "days": {
                            "type": "integer",
                            "description": "Number of days to look back (1-365)"
                        }
                    },
                    "required": ["days"]
                }
            ),
            Tool(
                name="get_payment_success_rate_in_last_days",
                description="Get payment success rate and revenue statistics for the last N days",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "days": {
                            "type": "integer",
                            "description": "Number of days to look back (1-365)"
                        }
                    },
                    "required": ["days"]
                }
            ),
            Tool(
                name="get_user_payment_history",
                description="Get payment history for a specific user by merchant_user_id",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "merchant_user_id": {
                            "type": "string",
                            "description": "The merchant user ID to query"
                        },
                        "days": {
                            "type": "integer",
                            "description": "Number of days to look back (default: 90)"
                        }
                    },
                    "required": ["merchant_user_id"]
                }
            ),
            Tool(
                name="get_database_status",
                description="Check database connection and get basic statistics",
                inputSchema={
                    "type": "object",
                    "properties": {}
                }
            ),
            Tool(
                name="execute_dynamic_sql",
                description="""Execute a custom SELECT SQL query for complex analytics.
                
Database Schema:
- subscription_contract_v2: subscription_id, merchant_user_id, status, subcription_start_date
- subscription_payment_details: subscription_id, status, trans_amount_decimal, created_date

Use MySQL syntax, always start with SELECT, use proper JOINs on subscription_id.""",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "sql_query": {
                            "type": "string",
                            "description": "SELECT SQL query to execute (MySQL syntax)"
                        }
                    },
                    "required": ["sql_query"]
                }
            ),
            Tool(
                name="record_query_feedback",
                description="Record feedback on a dynamic query result (for system learning)",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "original_question": {
                            "type": "string",
                            "description": "The original question that was asked"
                        },
                        "sql_query": {
                            "type": "string", 
                            "description": "The SQL query that was generated"
                        },
                        "was_helpful": {
                            "type": "boolean",
                            "description": "Whether the result was helpful and accurate"
                        }
                    },
                    "required": ["original_question", "sql_query", "was_helpful"]
                }
            ),
            Tool(
                name="get_query_suggestions",
                description="Get suggestions based on similar queries in memory",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "original_question": {
                            "type": "string",
                            "description": "The question to find similar queries for"
                        }
                    },
                    "required": ["original_question"]
                }
            )
        ]
        
        print(f"✅ Defined {len(self.tools)} tools", file=sys.stderr)
        self._register_handlers()
        
    def _load_config(self):
        """Load configuration from config.json"""
        if self.config:
            return  # Already loaded
            
        # Try multiple locations for config.json
        config_paths = [
            Path(__file__).parent / 'config.json',
            Path.cwd() / 'config.json',
            Path.cwd() / 'client' / 'config.json'
        ]
        
        for path in config_paths:
            try:
                if path.exists():
                    with open(path, 'r') as f:
                        self.config = json.load(f)
                        print(f"✅ Found config at: {path}", file=sys.stderr)
                        
                        # Validate required fields
                        required_fields = ['API_KEY_1', 'SUBSCRIPTION_API_URL']
                        missing_fields = [field for field in required_fields if not self.config.get(field)]
                        if missing_fields:
                            raise Exception(f"Missing required config fields: {missing_fields}")
                        
                        return
            except Exception as e:
                print(f"⚠️ Could not read config from {path}: {e}", file=sys.stderr)
                continue
        
        raise Exception(f"config.json not found. Tried: {[str(p) for p in config_paths]}")

    async def _init_session(self):
        """Initialize HTTP session if not already done"""
        if not self.session:
            connector = aiohttp.TCPConnector(
                ssl=ssl.create_default_context(cafile=certifi.where()),
                limit=10,
                limit_per_host=5,
                enable_cleanup_closed=True,
                force_close=True,
                ttl_dns_cache=300
            )
            self.session = aiohttp.ClientSession(
                connector=connector,
                timeout=aiohttp.ClientTimeout(total=120),
                headers={'Connection': 'close'}
            )

    def _register_handlers(self):
        print("🔧 Registering MCP handlers...", file=sys.stderr)
        
        @self.server.list_tools()
        async def handle_list_tools() -> List[Tool]:
            print("📋 MCP: list_tools called", file=sys.stderr)
            return self.tools
        
        @self.server.call_tool()
        async def handle_tool_call(name: str, args: Dict[str, Any]) -> List[TextContent]:
            print(f"🔧 MCP: tool_call '{name}' with args: {args}", file=sys.stderr)
            
            try:
                # Load config and init session
                self._load_config()
                await self._init_session()
                
                # Call the backend API tool directly
                result_data = await self._call_api_tool(name, args)
                
                # Format the result
                formatted_output = self._format_result(result_data, name)
                
                return [TextContent(text=formatted_output)]
                
            except Exception as e:
                print(f"❌ Error in tool call '{name}': {e}", file=sys.stderr)
                import traceback
                traceback.print_exc(file=sys.stderr)
                return [TextContent(text=f"❌ Error: {str(e)}")]
        
        print("✅ MCP handlers registered", file=sys.stderr)

    async def _call_api_tool(self, tool_name: str, parameters: Dict = None) -> Dict:
        """Call the subscription analytics API"""
        headers = {
            "Authorization": f"Bearer {self.config['API_KEY_1']}",
            "Connection": "close"
        }
        payload = {"tool_name": tool_name, "parameters": parameters or {}}
        server_url = self.config['SUBSCRIPTION_API_URL']
        
        try:
            async with self.session.post(f"{server_url}/execute", json=payload, headers=headers) as response:
                if response.status != 200:
                    error_text = await response.text()
                    return {
                        "success": False,
                        "error": f"HTTP {response.status}: {error_text}"
                    }
                
                return await response.json()
                
        except Exception as e:
            return {
                "success": False,
                "error": f"Network error: {str(e)}"
            }

    def _format_result(self, result_data: Dict, tool_name: str) -> str:
        """Format API result for display"""
        if not result_data.get('success', False):
            return f"❌ ERROR: {result_data.get('error', 'Unknown error')}"
        
        if result_data.get('message'):
            return f"ℹ️ {result_data['message']}"
        
        data = result_data.get('data')
        if data is None:
            return "✅ Query succeeded, but no data returned."
        
        output = []
        is_dynamic = tool_name == 'execute_dynamic_sql'
        header = f"📊 DYNAMIC QUERY RESULT" if is_dynamic else f"📊 RESULT FROM TOOL: {tool_name.upper()}"
        output.append(header)
        output.append("=" * len(header))
        
        if isinstance(data, list) and len(data) > 0:
            # Table formatting
            headers = list(data[0].keys())
            col_widths = {h: len(str(h)) for h in headers}
            
            # Calculate column widths
            for row in data:
                for h in headers:
                    col_widths[h] = max(col_widths[h], len(str(row.get(h, ''))))
            
            # Create table
            header_line = " | ".join(h.ljust(col_widths[h]) for h in headers)
            output.append(header_line)
            output.append("-" * len(header_line))
            
            for row in data:
                output.append(" | ".join(str(row.get(h, '')).ljust(col_widths[h]) for h in headers))
            
            output.append("")
            output.append(f"📈 Total rows: {len(data)}")
            
            # Add helpful note for dynamic queries
            if is_dynamic:
                output.append("\n💡 This was a custom SQL query. You can use record_query_feedback to help improve the system.")
            
        elif isinstance(data, dict):
            for key, value in data.items():
                output.append(f"{key}: {value}")
        else:
            output.append(str(data))
        
        return "\n".join(output)

    async def cleanup(self):
        print("🧹 Cleaning up full tools MCP client...", file=sys.stderr)
        if self.session:
            await self.session.close()

async def main():
    """Main entry point for full tools MCP server"""
    print(f"🚀 Full Tools MCP Client starting from: {Path(__file__).parent.absolute()}", file=sys.stderr)
    print(f"🔧 This version exposes all 7 backend tools individually", file=sys.stderr)
    
    try:
        mcp_client = FullToolsMCPClient()
        print("✅ Full tools MCP client instance created", file=sys.stderr)
        
        async with mcp.server.stdio.stdio_server() as (reader, writer):
            print("🚀 Full Tools MCP Server ready for Claude Desktop", file=sys.stderr)
            initialization_options = InitializationOptions(
                server_name="subscription-analytics",
                server_version="1.0.0",
                capabilities={}
            )
            await mcp_client.server.run(reader, writer, initialization_options)
    except Exception as e:
        print(f"❌ Full tools MCP server error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc(file=sys.stderr)
        raise
    finally:
        await mcp_client.cleanup()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("🛑 Full tools MCP server stopped by user", file=sys.stderr)
    except Exception as e:
        print(f"💥 Fatal error: {e}", file=sys.stderr)
        sys.exit(1)