#!/usr/bin/env python3
"""
MCP Client for Universal Client - FIXED VERSION
Provides an MCP-compatible interface for the universal client
"""

import asyncio
import json
import logging
import os
import sys
import inspect
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Callable, Awaitable

# MCP imports
try:
    import mcp
    from mcp import types
    from mcp.server import Server
    from mcp.types import Tool, TextContent
    import mcp.server.stdio
    import mcp.types as types
except ImportError:
    print("❌ MCP not installed. Run: pip install mcp")
    sys.exit(1)

# Add parent directory to path to allow importing from the client module
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import your existing universal client
try:
    from universal_client import UniversalClient, QueryResult
except ImportError:
    print("❌ Universal client not found. Make sure it's in your PYTHONPATH")
    sys.exit(1)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("mcp-universal-client")

class UniversalClientMCPServer:
    """MCP Server wrapper for your Universal Client - FIXED"""
    
    def __init__(self):
        self.server = Server("subscription-analytics")
        self.universal_client = None
        
        # Load configuration for the universal client
        self.config = self._load_config()
        
        # Define MCP tools that map to your universal client capabilities
        self.tools = [
            Tool(
                name="natural_language_query",
                description="Process any natural language query about subscription analytics using AI",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Natural language query about subscription data (e.g., 'compare 7 days vs 30 days performance')"
                        }
                    },
                    "required": ["query"]
                }
            ),
            Tool(
                name="get_subscription_summary",
                description="Get comprehensive subscription and payment summary for a specific period",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "days": {
                            "type": "integer",
                            "description": "Number of days to look back (default: 30)",
                            "minimum": 1,
                            "maximum": 365,
                            "default": 30
                        }
                    },
                    "required": []
                }
            ),
            Tool(
                name="get_subscriptions_in_last_days",
                description="Get subscription statistics for the last N days",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "days": {
                            "type": "integer",
                            "description": "Number of days to look back",
                            "minimum": 1,
                            "maximum": 365
                        }
                    },
                    "required": ["days"]
                }
            ),
            Tool(
                name="get_payment_success_rate_in_last_days",
                description="Get payment success rate statistics for the last N days",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "days": {
                            "type": "integer",
                            "description": "Number of days to look back",
                            "minimum": 1,
                            "maximum": 365
                        }
                    },
                    "required": ["days"]
                }
            ),
            Tool(
                name="get_database_status",
                description="Check database connection and get basic statistics",
                inputSchema={
                    "type": "object",
                    "properties": {},
                    "required": []
                }
            ),
            Tool(
                name="get_user_payment_history",
                description="Get payment history for a specific user",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "merchant_user_id": {
                            "type": "string",
                            "description": "The merchant user ID to query"
                        },
                        "days": {
                            "type": "integer",
                            "description": "Number of days to look back (default: 90)",
                            "minimum": 1,
                            "maximum": 365,
                            "default": 90
                        }
                    },
                    "required": ["merchant_user_id"]
                }
            ),
            Tool(
                name="compare_periods",
                description="Compare analytics across multiple time periods",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "periods": {
                            "type": "array",
                            "items": {"type": "integer"},
                            "description": "List of day periods to compare (e.g., [7, 14, 30])",
                            "minItems": 2,
                            "maxItems": 5
                        }
                    },
                    "required": ["periods"]
                }
            ),
            Tool(
                name="get_subscriptions_by_date_range",
                description="Get subscription statistics for a specific date range",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "start_date": {
                            "type": "string",
                            "description": "Start date in YYYY-MM-DD format"
                        },
                        "end_date": {
                            "type": "string",
                            "description": "End date in YYYY-MM-DD format"
                        }
                    },
                    "required": ["start_date", "end_date"]
                }
            ),
            Tool(
                name="get_payments_by_date_range",
                description="Get payment statistics for a specific date range",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "start_date": {
                            "type": "string",
                            "description": "Start date in YYYY-MM-DD format"
                        },
                        "end_date": {
                            "type": "string",
                            "description": "End date in YYYY-MM-DD format"
                        }
                    },
                    "required": ["start_date", "end_date"]
                }
            ),
            Tool(
                name="get_analytics_by_date_range",
                description="Get comprehensive analytics for a specific date range",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "start_date": {
                            "type": "string",
                            "description": "Start date in YYYY-MM-DD format"
                        },
                        "end_date": {
                            "type": "string",
                            "description": "End date in YYYY-MM-DD format"
                        }
                    },
                    "required": ["start_date", "end_date"]
                }
            )
        ]
        
        # Register MCP handlers
        self._register_handlers()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration for universal client"""
        from dotenv import load_dotenv
        load_dotenv()
        
        config = {
            'server_url': os.getenv('SUBSCRIPTION_API_URL', 'http://localhost:8000'),
            'api_key': os.getenv('API_KEY_1') or os.getenv('SUBSCRIPTION_API_KEY'),
            'gemini_api_key': os.getenv('GEMINI_API_KEY'),
            'timeout': int(os.getenv('SUBSCRIPTION_API_TIMEOUT', '30')),
            'retry_attempts': int(os.getenv('SUBSCRIPTION_API_RETRIES', '3'))
        }
        
        # Validate required config
        required_keys = ['server_url', 'api_key']
        missing_keys = []
        for key in required_keys:
            if not config.get(key):
                missing_keys.append(key)
        
        if missing_keys:
            logger.error(f"Missing required configuration: {missing_keys}")
            logger.error("Please set environment variables: SUBSCRIPTION_API_URL, API_KEY_1")
            raise ValueError(f"Missing configuration: {missing_keys}")
        
        logger.info(f"✅ Configuration loaded - Server: {config['server_url']}")
        return config
    
    def _register_handlers(self):
        """Register MCP protocol handlers"""
        logger.info("🔄 Registering MCP handlers...")
        
        # Create a dictionary to map tool names to handlers
        self._tool_handlers = {}
        
        # Create a single tool call handler
        @self.server.call_tool()
        async def handle_tool_call(name: str, arguments: Dict[str, Any]) -> List[types.TextContent]:
            logger.info(f"🔧 Handling tool call: {name} with args: {arguments}")
            if name in self._tool_handlers:
                try:
                    # Get the handler function
                    handler = self._tool_handlers[name]
                    
                    # Call the handler with the arguments
                    logger.debug(f"🔄 Executing handler for {name}")
                    
                    # Call the handler and get the coroutine
                    coro = handler(**arguments)
                    
                    # If it's a coroutine, await it
                    if asyncio.iscoroutine(coro):
                        result = await coro
                    else:
                        result = coro
                        
                    # Format the result as TextContent
                    if isinstance(result, (str, int, float, bool)):
                        return [types.TextContent(type="text", text=str(result))]
                    elif isinstance(result, dict) or isinstance(result, list):
                        return [types.TextContent(type="text", text=json.dumps(result, indent=2))]
                    else:
                        return [types.TextContent(type="text", text=str(result))]
                        
                except Exception as e:
                    error_msg = f"❌ Error in tool handler for {name}: {str(e)}"
                    logger.error(error_msg, exc_info=True)
                    return [types.TextContent(type="text", text=error_msg)]
                    
            error_msg = f"No handler registered for tool: {name}"
            logger.error(error_msg)
            return [types.TextContent(type="text", text=error_msg)]
        
        # Register the list_tools handler
        @self.server.list_tools()
        async def handle_list_tools() -> List[Tool]:
            """List available tools"""
            logger.info("🔧 MCP client requesting tool list")
            # Log the tools being returned for debugging
            for tool in self.tools:
                logger.info(f"🔧 Registered tool: {tool.name} - {tool.description}")
            return self.tools
        
        # Create and register tool handlers
        for tool in self.tools:
            logger.debug(f"🔧 Registering tool handler for: {tool.name}")
            
            # Create a unique handler for this tool
            tool_name = tool.name  # Capture the tool name in the closure
            
            # Create a closure to capture the tool name
            def create_tool_handler(tool_name):
                async def tool_handler(**kwargs):
                    return await self.handle_call_tool(tool_name, kwargs)
                return tool_handler
            
            # Store the handler
            self._tool_handlers[tool_name] = create_tool_handler(tool_name)
            logger.debug(f"✅ Registered handler for {tool_name}")
        
        logger.info("✅ MCP handlers registered successfully")
    
    async def handle_call_tool(self, name: str, arguments: Dict[str, Any]) -> str:
        """
        Handle tool calls from MCP clients
        
        Args:
            name: Name of the tool to call
            arguments: Dictionary of arguments for the tool
            
        Returns:
            str: The result of the tool execution as a string
        """
        logger.info(f"🎯 MCP tool call received: {name} with arguments: {arguments}")
        
        try:
            # Ensure universal client is initialized
            if not self.universal_client:
                logger.info("🔌 Initializing UniversalClient...")
                self.universal_client = UniversalClient(**self.config)
                await self.universal_client.__aenter__()
            
            logger.debug(f"🛠️  Executing tool: {name}")
            
            # Handle different tool calls
            if name == "natural_language_query":
                result = await self._handle_natural_language_query(arguments)
            elif name == "get_subscription_summary":
                result = await self._handle_direct_tool_call(name, arguments)
            elif name == "get_subscriptions_in_last_days":
                result = await self._handle_direct_tool_call(name, arguments)
            elif name == "get_payment_success_rate_in_last_days":
                result = await self._handle_direct_tool_call(name, arguments)
            elif name == "get_database_status":
                result = await self._handle_direct_tool_call(name, arguments)
            elif name == "get_user_payment_history":
                result = await self._handle_direct_tool_call(name, arguments)
            elif name == "compare_periods":
                result = await self._handle_compare_periods(arguments)
            else:
                raise ValueError(f"Unknown tool: {name}")
            
            # Format the result as a string
            if isinstance(result, dict) or isinstance(result, list):
                formatted_result = json.dumps(result, indent=2)
            else:
                formatted_result = str(result)
                
            logger.debug(f"✅ Tool {name} executed successfully")
            return formatted_result
            
        except Exception as e:
            error_msg = f"❌ Tool call failed: {str(e)}"
            logger.error(error_msg, exc_info=True)
            
            # Include more detailed error information for debugging
            import traceback
            error_details = traceback.format_exc()
            logger.debug(f"Error details: {error_details}")
            
            # Format a user-friendly error message
            error_message = (
                f"❌ Error executing {name}: {str(e)}\n\n"
                "Please check:\n"
                f"- Error details: {error_details}\n"
                "- Is the API server running?\n"
                "- Is the configuration correct?\n"
                "- Is there network connectivity?"
            )
            return error_message
    
    async def _handle_natural_language_query(self, arguments: Dict[str, Any]) -> str:
        """Handle natural language queries using your universal client"""
        query = arguments.get("query", "")
        logger.info(f"🤖 Processing natural language query: '{query}'")
        
        try:
            # Use your universal client's query_formatted method
            result = await self.universal_client.query_formatted(query)
            return result
        except Exception as e:
            logger.error(f"Natural language query failed: {e}")
            return f"❌ Query failed: {str(e)}"
    
    async def _handle_compare_periods(self, arguments: Dict[str, Any]) -> str:
        """Handle period comparison queries"""
        periods = arguments.get("periods", [])
        logger.info(f"📊 Comparing periods: {periods}")
        
        try:
            # Create a natural language query for comparison
            periods_str = " vs ".join([f"{p} days" for p in periods])
            query = f"compare the statistics for {periods_str}"
            
            # Use your universal client
            result = await self.universal_client.query_formatted(query)
            return result
        except Exception as e:
            logger.error(f"Period comparison failed: {e}")
            return f"❌ Comparison failed: {str(e)}"
    
    async def _handle_direct_tool_call(self, tool_name: str, arguments: Dict[str, Any]) -> Any:
        """
        Handle direct tool calls to your backend
        
        Args:
            tool_name: Name of the tool/method to call on the universal client
            arguments: Dictionary of arguments to pass to the method
            
        Returns:
            The result of the method call
            
        Raises:
            ValueError: If the tool/method is not found
            Exception: Any exception raised by the called method
        """
        logger.info(f"🔧 Direct tool call: {tool_name} with args: {arguments}")
        
        # Clean up the arguments (remove None values)
        clean_args = {k: v for k, v in arguments.items() if v is not None}
        
        # Log the cleaned arguments
        logger.debug(f"🔧 Cleaned arguments for {tool_name}: {clean_args}")
        
        try:
            # Get the method from the universal client
            method = getattr(self.universal_client, tool_name, None)
            if not method or not callable(method):
                error_msg = f"Tool/method not found: {tool_name}"
                logger.error(error_msg)
                raise ValueError(error_msg)
            
            # Log the method being called
            logger.debug(f"🔧 Calling method: {tool_name} with args: {clean_args}")
            
            # Call the method with the provided arguments
            if asyncio.iscoroutinefunction(method):
                result = await method(**clean_args)
            else:
                result = method(**clean_args)
                
            logger.debug(f"✅ Method {tool_name} executed successfully")
            return result
            
        except Exception as e:
            error_msg = f"❌ Error in direct tool call {tool_name}: {str(e)}"
            logger.error(error_msg, exc_info=True)
            
            # Include more context in the error message
            error_details = f"Error calling {tool_name} with args {clean_args}: {str(e)}"
            logger.debug(f"Error details: {error_details}")
            
            # Re-raise with more context
            raise Exception(f"Failed to execute {tool_name}: {str(e)}") from e
    
    def _format_for_mcp(self, result: Any, tool_name: str, arguments: Dict[str, Any]) -> str:
        """Format results for MCP clients with metadata"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        output = []
        output.append(f"🎯 **Subscription Analytics Result**")
        output.append(f"⏰ Generated: {timestamp}")
        output.append(f"🔧 Tool: {tool_name}")
        if arguments:
            output.append(f"📋 Parameters: {json.dumps(arguments, indent=2)}")
        output.append("")
        output.append("---")
        output.append("")
        output.append(str(result))
        output.append("")
        output.append("---")
        output.append("💡 *Powered by Universal Analytics Client*")
        return "\n".join(output)
    
    async def cleanup(self):
        """Cleanup resources"""
        if self.universal_client:
            await self.universal_client.__aexit__(None, None, None)

# FIXED: Proper MCP server mode
async def run_mcp_server():
    """Run the MCP server - THIS IS THE KEY FIX"""
    logger.info("🌉 Starting MCP Server for Claude Desktop")
    
    mcp_server = UniversalClientMCPServer()
    
    # Log all registered tools at startup
    logger.info("📋 Registered tools:")
    for tool in mcp_server.tools:
        logger.info(f"  - {tool.name}: {tool.description}")
    
    try:
        # This is what Claude Desktop expects - stdio server
        logger.info("🔌 Starting MCP server on stdio...")
        async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
            logger.info("🚀 MCP Server is running and ready to accept connections")
            await mcp_server.server.run(
                read_stream,
                write_stream,
                mcp_server.server.create_initialization_options()
            )
    except KeyboardInterrupt:
        logger.info("🛑 MCP Server interrupted")
    except Exception as e:
        logger.error(f"❌ MCP Server error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise
    finally:
        logger.info("🧹 Cleaning up MCP server resources...")
        await mcp_server.cleanup()
        logger.info("👋 MCP Server shutdown complete")
        logger.info("🧹 MCP Server cleanup complete")

# FIXED: Interactive mode
async def run_interactive():
    """Run interactive mode"""
    try:
        from universal_client import interactive_mode
        await interactive_mode()
    except ImportError:
        print("❌ Could not import interactive_mode from universal_client")
        print("Running basic interactive mode...")
        
        # Basic fallback interactive mode
        from universal_client import UniversalClient
        config = {
            'server_url': os.getenv('SUBSCRIPTION_API_URL', 'http://localhost:8000'),
            'api_key': os.getenv('API_KEY_1'),
            'gemini_api_key': os.getenv('GEMINI_API_KEY')
        }
        
        async with UniversalClient(**config) as client:
            while True:
                try:
                    query = input("\n🎯 Query: ").strip()
                    if query.lower() in ['quit', 'exit']:
                        break
                    result = await client.query_formatted(query)
                    print(result)
                except KeyboardInterrupt:
                    break

# FIXED: Single query mode  
async def run_single_query(query: str):
    """Run single query mode"""
    try:
        from universal_client import single_query_mode
        exit_code = await single_query_mode(query)
        sys.exit(exit_code)
    except ImportError:
        print("❌ Could not import single_query_mode from universal_client")
        sys.exit(1)

# FIXED: Main function
async def main():
    """Main entry point - COMPLETELY FIXED"""
    # Configure logging to show all levels
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Check command line arguments
    if len(sys.argv) > 1:
        if sys.argv[1] in ['--mcp', '-m']:
            # MCP server mode - this is what Claude Desktop calls
            logger.info("🚀 Starting in MCP server mode (--mcp flag detected)")
            await run_mcp_server()
            return
        elif sys.argv[1] in ['--help', '-h']:
            print("""
🎯 Universal Client MCP Bridge

Usage:
  python mcp_client.py           # Interactive mode
  python mcp_client.py --mcp     # MCP server mode (for Claude Desktop)
  python mcp_client.py "query"   # Single query mode
  python mcp_client.py --debug    # Enable debug logging

Examples:
  python mcp_client.py --mcp
  python mcp_client.py "show me database status"
            """)
            return
        elif sys.argv[1] == '--debug':
            logging.getLogger().setLevel(logging.DEBUG)
            logger.debug("🔍 Debug logging enabled")
            if len(sys.argv) > 2 and sys.argv[2] == '--mcp':
                await run_mcp_server()
                return
            else:
                await run_interactive()
                return
        else:
            # Single query mode
            query = " ".join(sys.argv[1:])
            await run_single_query(query)
            return
    
    # Check if being called by MCP client (Claude Desktop)
    if not sys.stdin.isatty():
        # Being called by Claude Desktop - run MCP server
        logger.info("🚀 Starting in MCP server mode (non-interactive mode detected)")
        await run_mcp_server()
    else:
        # Interactive mode
        logger.info("💻 Starting in interactive mode")
        await run_interactive()

if __name__ == "__main__":
    asyncio.run(main())