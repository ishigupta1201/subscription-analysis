#!/usr/bin/env python3
# standalone_mcp_client.py - Self-contained MCP client with no external dependencies

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

# Google AI imports
try:
    import google.generativeai as genai
except ImportError:
    print("❌ Google AI not installed. Run: pip install google-generativeai", file=sys.stderr)
    sys.exit(1)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("standalone-mcp")

class StandaloneMCPClient:
    def __init__(self):
        print("🔧 Initializing Standalone MCP Client...", file=sys.stderr)
        self.server = Server("subscription-analytics")
        self.config = None
        self.session = None
        self.last_query_info = None  # Store info for feedback
        
        # Define tools for Claude Desktop
        self.tools = [
            Tool(
                name="natural_language_query",
                description="""
Analyze subscription and payment data using natural language queries.

Examples:
• "What's the payment success rate for the last month?"
• "How many users have success rate above 15%?"
• "Show me payment history for user abc123"
• "Which users have the highest success rates?"

The system intelligently chooses between pre-built tools or generates custom SQL.
""",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Natural language question about subscription/payment analytics"
                        }
                    },
                    "required": ["query"]
                }
            ),
            Tool(
                name="record_feedback",
                description="Record feedback on dynamic query results to improve the system",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "was_helpful": {
                            "type": "boolean",
                            "description": "Whether the previous answer was helpful and accurate"
                        }
                    },
                    "required": ["was_helpful"]
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
                        required_fields = ['GOOGLE_API_KEY', 'API_KEY_1', 'SUBSCRIPTION_API_URL']
                        missing_fields = [field for field in required_fields if not self.config.get(field)]
                        if missing_fields:
                            raise Exception(f"Missing required config fields: {missing_fields}")
                        
                        # Configure Google AI
                        genai.configure(api_key=self.config['GOOGLE_API_KEY'])
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
                
                if name == "natural_language_query":
                    return await self._handle_query(args)
                elif name == "record_feedback":
                    return await self._handle_feedback(args)
                else:
                    return [TextContent(text=f"❌ Unknown tool: {name}")]
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
            
        elif isinstance(data, dict):
            for key, value in data.items():
                output.append(f"{key}: {value}")
        else:
            output.append(str(data))
        
        return "\n".join(output)

    async def _choose_tool_and_params(self, user_query: str) -> tuple:
        """Use AI to choose the best tool and parameters"""
        model = genai.GenerativeModel('gemini-1.5-flash-latest')
        
        prompt = f"""
You are a subscription analytics assistant. For the user query: "{user_query}"

Choose the best tool and parameters from these options:

1. get_subscriptions_in_last_days(days: int) - For subscription overview questions
2. get_payment_success_rate_in_last_days(days: int) - For payment performance questions  
3. get_user_payment_history(merchant_user_id: str, days: int) - For specific user history
4. get_database_status() - For health checks
5. execute_dynamic_sql(sql_query: str) - For complex analytics requiring custom SQL

Database schema:
- subscription_contract_v2 (subscription_id, merchant_user_id, status, subcription_start_date)
- subscription_payment_details (subscription_id, status, trans_amount_decimal, created_date)

For dynamic SQL:
- Use MySQL syntax
- Always start with SELECT
- Use proper JOINs between tables on subscription_id
- Use 'ACTIVE' status for successful payments

Return your choice in this exact format:
TOOL: tool_name
PARAMETERS: {{"param1": "value1", "param2": "value2"}}

If generating SQL, make it valid MySQL that starts with SELECT.
"""

        try:
            response = model.generate_content(prompt)
            response_text = response.text.strip()
            
            # Parse the response
            lines = response_text.split('\n')
            tool_name = None
            parameters = {}
            
            for line in lines:
                if line.startswith('TOOL:'):
                    tool_name = line.split(':', 1)[1].strip()
                elif line.startswith('PARAMETERS:'):
                    param_text = line.split(':', 1)[1].strip()
                    try:
                        parameters = json.loads(param_text)
                    except:
                        parameters = {}
            
            if not tool_name:
                # Fallback
                tool_name = 'get_database_status'
                parameters = {}
            
            print(f"🤖 AI chose tool: {tool_name} with params: {parameters}", file=sys.stderr)
            return tool_name, parameters
            
        except Exception as e:
            print(f"⚠️ AI tool selection failed: {e}, using fallback", file=sys.stderr)
            return 'get_database_status', {}

    async def _handle_query(self, args: Dict[str, Any]) -> List[TextContent]:
        query = args.get("query")
        if not query:
            return [TextContent(text="❌ Query parameter missing")]
        
        print(f"🔍 Processing query: {query[:50]}...", file=sys.stderr)
        
        try:
            # Use AI to choose tool and parameters
            tool_name, parameters = await self._choose_tool_and_params(query)
            
            # Call the API
            result_data = await self._call_api_tool(tool_name, parameters)
            
            # Format the result
            formatted_output = self._format_result(result_data, tool_name)
            
            # Store for potential feedback
            if tool_name == 'execute_dynamic_sql' and result_data.get('success'):
                self.last_query_info = {
                    'original_question': query,
                    'sql_query': parameters.get('sql_query', ''),
                    'tool_name': tool_name
                }
                formatted_output += """

---
**📝 Feedback Opportunity**
This was a custom generated query. If you'd like to help improve the system, let me know if this result was helpful and I can record your feedback.
"""
            
            print(f"✅ Query processed successfully", file=sys.stderr)
            return [TextContent(text=formatted_output)]
            
        except Exception as e:
            print(f"❌ Query processing error: {e}", file=sys.stderr)
            import traceback
            traceback.print_exc(file=sys.stderr)
            return [TextContent(text=f"❌ Error processing query: {e}")]

    async def _handle_feedback(self, args: Dict[str, Any]) -> List[TextContent]:
        was_helpful = args.get("was_helpful")
        if was_helpful is None:
            return [TextContent(text="❌ was_helpful parameter missing")]
        
        if not self.last_query_info:
            return [TextContent(text="⚠️ No recent dynamic query to provide feedback for")]
        
        print(f"📝 Recording feedback: {'positive' if was_helpful else 'negative'}", file=sys.stderr)
        
        try:
            # Submit feedback
            feedback_params = {
                'original_question': self.last_query_info['original_question'],
                'sql_query': self.last_query_info['sql_query'],
                'was_helpful': was_helpful
            }
            
            result = await self._call_api_tool('record_query_feedback', feedback_params)
            self.last_query_info = None  # Clear after feedback
            
            if result.get('success'):
                if was_helpful:
                    return [TextContent(text="✅ Positive feedback recorded! This will help improve future queries.")]
                else:
                    return [TextContent(text="📝 Negative feedback recorded! The system will avoid similar approaches.")]
            else:
                return [TextContent(text=f"⚠️ Feedback noted but couldn't sync: {result.get('error', 'Unknown error')}")]
                
        except Exception as e:
            print(f"❌ Feedback error: {e}", file=sys.stderr)
            return [TextContent(text=f"⚠️ Feedback noted but couldn't sync: {e}")]

    async def cleanup(self):
        print("🧹 Cleaning up standalone MCP client...", file=sys.stderr)
        if self.session:
            await self.session.close()

async def main():
    """Main entry point for standalone MCP server"""
    print(f"🚀 Standalone MCP Client starting from: {Path(__file__).parent.absolute()}", file=sys.stderr)
    print(f"🐍 Python executable: {sys.executable}", file=sys.stderr)
    print(f"🐍 Python version: {sys.version}", file=sys.stderr)
    
    try:
        mcp_client = StandaloneMCPClient()
        print("✅ Standalone MCP client instance created", file=sys.stderr)
        
        async with mcp.server.stdio.stdio_server() as (reader, writer):
            print("🚀 Standalone MCP Server ready for Claude Desktop", file=sys.stderr)
            initialization_options = InitializationOptions(
                server_name="subscription-analytics",
                server_version="1.0.0",
                capabilities={}
            )
            await mcp_client.server.run(reader, writer, initialization_options)
    except Exception as e:
        print(f"❌ Standalone MCP server error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc(file=sys.stderr)
        raise
    finally:
        await mcp_client.cleanup()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("🛑 Standalone MCP server stopped by user", file=sys.stderr)
        logger.info("Standalone MCP server stopped")
    except Exception as e:
        print(f"💥 Fatal error: {e}", file=sys.stderr)
        logger.error(f"Fatal error: {e}")
        sys.exit(1)