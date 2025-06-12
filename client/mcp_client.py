#!/usr/bin/env python3
"""
MCP Client for Universal Client - ENHANCED FIXED VERSION
Provides an MCP-compatible interface for the universal client with full date range support
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
sys.path.insert(0, str(Path(__file__).parent))

# Import your existing universal client
try:
    from universal_client import UniversalClient, QueryResult
except ImportError:
    try:
        # Try alternative import paths
        sys.path.insert(0, os.getcwd())
        from universal_client import UniversalClient, QueryResult
    except ImportError:
        print("❌ Universal client not found. Make sure universal_client.py is in the same directory or PYTHONPATH")
        print(f"Current working directory: {os.getcwd()}")
        print(f"Script directory: {Path(__file__).parent}")
        sys.exit(1)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("mcp-universal-client")

class UniversalClientMCPServer:
    """MCP Server wrapper for your Universal Client - ENHANCED FIXED VERSION"""
    
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
        """Load configuration for universal client with enhanced error handling"""
        from dotenv import load_dotenv
        
        # Try to load .env from multiple locations
        env_paths = [
            '.env',
            '../.env',
            os.path.join(os.getcwd(), '.env'),
            os.path.join(Path(__file__).parent, '.env'),
            os.path.join(Path(__file__).parent.parent, '.env')
        ]
        
        env_loaded = False
        for env_path in env_paths:
            if os.path.exists(env_path):
                load_dotenv(env_path)
                env_loaded = True
                logger.info(f"Loaded .env from: {env_path}")
                break
        
        if not env_loaded:
            logger.warning("No .env file found, using environment variables only")
        
        config = {
            'server_url': os.getenv('SUBSCRIPTION_API_URL', 'https://subscription-analysis-production.up.railway.app'),
            'api_key': os.getenv('API_KEY_1') or os.getenv('SUBSCRIPTION_API_KEY'),
            'gemini_api_key': os.getenv('GEMINI_API_KEY'),
            'timeout': int(os.getenv('SUBSCRIPTION_API_TIMEOUT', '30')),
            'retry_attempts': int(os.getenv('SUBSCRIPTION_API_RETRIES', '3'))
        }
        
        # DEBUG: Print configuration (sanitized)
        logger.info(f"🔍 MCP Client Configuration:")
        logger.info(f"  Working Directory: {os.getcwd()}")
        logger.info(f"  Script Directory: {Path(__file__).parent}")
        logger.info(f"  URL: {config['server_url']}")
        logger.info(f"  API Key: {'SET (' + config['api_key'][:20] + '...)' if config['api_key'] else 'MISSING'}")
        logger.info(f"  Gemini Key: {'SET' if config['gemini_api_key'] else 'MISSING'}")
        
        # Validate required config
        required_keys = ['server_url', 'api_key']
        missing_keys = []
        for key in required_keys:
            if not config.get(key):
                missing_keys.append(key)
        
        if missing_keys:
            logger.error(f"Missing required configuration: {missing_keys}")
            logger.error("Please set environment variables:")
            logger.error("  - SUBSCRIPTION_API_URL (or use default)")
            logger.error("  - API_KEY_1 or SUBSCRIPTION_API_KEY")
            logger.error("  - GEMINI_API_KEY (optional but recommended)")
            raise ValueError(f"Missing configuration: {missing_keys}")
        
        logger.info(f"✅ Configuration loaded successfully")
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
            
            try:
                # Execute the tool
                result = await self.handle_call_tool(name, arguments)
                
                # Return as TextContent
                return [types.TextContent(type="text", text=str(result))]
                        
            except Exception as e:
                error_msg = f"❌ Error in tool handler for {name}: {str(e)}"
                logger.error(error_msg, exc_info=True)
                return [types.TextContent(type="text", text=error_msg)]
        
        # Register the list_tools handler
        @self.server.list_tools()
        async def handle_list_tools() -> List[Tool]:
            """List available tools"""
            logger.info("🔧 MCP client requesting tool list")
            # Log the tools being returned for debugging
            for tool in self.tools:
                logger.debug(f"🔧 Available tool: {tool.name} - {tool.description}")
            return self.tools
        
        logger.info("✅ MCP handlers registered successfully")
    
    async def handle_call_tool(self, name: str, arguments: Dict[str, Any]) -> str:
        """
        Handle tool calls from MCP clients - ENHANCED VERSION
        
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
                logger.info("✅ UniversalClient initialized successfully")
            
            logger.debug(f"🛠️ Executing tool: {name}")
            
            # Handle different tool calls with enhanced routing
            if name == "natural_language_query":
                result = await self._handle_natural_language_query(arguments)
            elif name == "compare_periods":
                result = await self._handle_compare_periods(arguments)
            elif name in ["get_subscriptions_by_date_range", "get_payments_by_date_range", "get_analytics_by_date_range"]:
                # Handle date range tools - try direct API first, then natural language
                result = await self._handle_date_range_tool(name, arguments)
            elif name in ["get_subscription_summary", "get_subscriptions_in_last_days", 
                          "get_payment_success_rate_in_last_days", "get_database_status", 
                          "get_user_payment_history"]:
                # Handle standard API tools
                result = await self._handle_api_tool_call(name, arguments)
            else:
                # Unknown tool - try direct API call as fallback
                logger.warning(f"Unknown tool {name}, trying direct API call")
                result = await self._handle_api_tool_call(name, arguments)
            
            logger.debug(f"✅ Tool {name} executed successfully")
            return result
            
        except Exception as e:
            error_msg = f"❌ Tool call failed: {str(e)}"
            logger.error(error_msg, exc_info=True)
            
            # Include more detailed error information for debugging
            import traceback
            error_details = traceback.format_exc()
            logger.debug(f"Full error details: {error_details}")
            
            # Format a user-friendly error message
            error_message = (
                f"❌ Error executing {name}: {str(e)}\n\n"
                "Troubleshooting:\n"
                f"- Tool: {name}\n"
                f"- Arguments: {arguments}\n"
                f"- API URL: {self.config.get('server_url', 'Not set')}\n"
                f"- API Key: {'Present' if self.config.get('api_key') else 'Missing'}\n"
                "\nPlease check:\n"
                "- Is the API server running?\n"
                "- Are the environment variables correct?\n"
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
            return f"❌ Natural language query failed: {str(e)}"
    
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
    
    async def _handle_date_range_tool(self, tool_name: str, arguments: Dict[str, Any]) -> str:
        """Handle date range tools with fallback strategy"""
        start_date = arguments.get("start_date", "")
        end_date = arguments.get("end_date", "")
        
        logger.info(f"📅 Date range tool: {tool_name} from {start_date} to {end_date}")
        
        try:
            # First, try direct API call
            logger.debug(f"Trying direct API call for {tool_name}")
            result = await self._handle_api_tool_call(tool_name, arguments)
            
            # Check if result looks like an error
            if "❌" in result or "Error" in result or "Failed" in result:
                logger.warning(f"Direct API call failed, trying natural language approach")
                # Fallback to natural language
                if tool_name == "get_analytics_by_date_range":
                    query = f"get comprehensive analytics from {start_date} to {end_date}"
                elif tool_name == "get_subscriptions_by_date_range":
                    query = f"get subscription statistics from {start_date} to {end_date}"
                elif tool_name == "get_payments_by_date_range":
                    query = f"get payment statistics from {start_date} to {end_date}"
                else:
                    return result  # Return the error from direct API call
                
                # Try natural language approach
                result = await self._handle_natural_language_query({"query": query})
            
            return result
            
        except Exception as e:
            logger.error(f"Date range tool failed: {e}")
            return f"❌ Date range query failed: {str(e)}"
    
    async def _handle_api_tool_call(self, tool_name: str, arguments: Dict[str, Any]) -> str:
        """
        Handle tool calls that go to the API server via universal client's call_tool method
        """
        logger.info(f"🔧 API tool call: {tool_name} with args: {arguments}")
        
        # Clean up the arguments (remove None values)
        clean_args = {k: v for k, v in arguments.items() if v is not None}
        
        try:
            # Use universal client's call_tool method to hit the API
            result = await self.universal_client.call_tool(tool_name, clean_args)
            
            if result.success:
                # Format the result using the universal client's formatter
                formatted_result = self.universal_client.formatter.format_single_result(result)
                return formatted_result
            else:
                return f"❌ API Error: {result.error}"
                
        except AttributeError as e:
            logger.error(f"❌ Universal client method not found: {e}")
            return f"❌ Client error: Universal client missing required method or attribute"
        except Exception as e:
            logger.error(f"❌ API tool call failed: {e}")
            return f"❌ API tool call failed: {str(e)}"
    
    async def cleanup(self):
        """Cleanup resources"""
        if self.universal_client:
            try:
                await self.universal_client.__aexit__(None, None, None)
                logger.info("✅ Universal client cleaned up")
            except Exception as e:
                logger.error(f"Error during cleanup: {e}")

# MCP Server Mode
async def run_mcp_server():
    """Run the MCP server - ENHANCED VERSION"""
    logger.info("🌉 Starting MCP Server for Claude Desktop")
    
    try:
        mcp_server = UniversalClientMCPServer()
        
        # Log all registered tools at startup
        logger.info("📋 Registered tools:")
        for tool in mcp_server.tools:
            logger.info(f"  - {tool.name}: {tool.description}")
        
        # This is what Claude Desktop expects - stdio server
        logger.info("🔌 Starting MCP server on stdio...")
        async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
            logger.info("🚀 MCP Server is running and ready to accept connections from Claude Desktop")
            await mcp_server.server.run(
                read_stream,
                write_stream,
                mcp_server.server.create_initialization_options()
            )
    except KeyboardInterrupt:
        logger.info("🛑 MCP Server interrupted by user")
    except Exception as e:
        logger.error(f"❌ MCP Server error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise
    finally:
        logger.info("🧹 Cleaning up MCP server resources...")
        if 'mcp_server' in locals():
            await mcp_server.cleanup()
        logger.info("👋 MCP Server shutdown complete")

# Interactive Mode
async def run_interactive():
    """Run interactive mode with enhanced error handling"""
    print("🎯 Subscription Analytics - Interactive Mode")
    print("=" * 50)
    
    try:
        # Try to import and use the universal client's interactive mode
        try:
            from universal_client import interactive_mode
            await interactive_mode()
        except ImportError:
            logger.warning("Could not import interactive_mode, using fallback")
            await _fallback_interactive()
    except Exception as e:
        logger.error(f"Interactive mode failed: {e}")
        print(f"❌ Interactive mode failed: {e}")

async def _fallback_interactive():
    """Fallback interactive mode"""
    print("Running basic interactive mode...")
    
    # Load config
    try:
        from dotenv import load_dotenv
        load_dotenv()
        
        config = {
            'server_url': os.getenv('SUBSCRIPTION_API_URL', 'https://subscription-analysis-production.up.railway.app'),
            'api_key': os.getenv('API_KEY_1') or os.getenv('SUBSCRIPTION_API_KEY'),
            'gemini_api_key': os.getenv('GEMINI_API_KEY')
        }
        
        missing = []
        if not config['api_key']:
            missing.append('API_KEY_1')
        if not config['gemini_api_key']:
            missing.append('GEMINI_API_KEY')
        
        if missing:
            print(f"❌ Missing environment variables: {missing}")
            return
        
    except Exception as e:
        print(f"❌ Configuration error: {e}")
        return
    
    # Run interactive session
    try:
        async with UniversalClient(**config) as client:
            print("\n💬 Enter your queries (or 'quit' to exit):")
            while True:
                try:
                    query = input("\n🎯 Query: ").strip()
                    if query.lower() in ['quit', 'exit', 'q']:
                        break
                    if not query:
                        continue
                    
                    result = await client.query_formatted(query)
                    print(result)
                except KeyboardInterrupt:
                    break
                except Exception as e:
                    print(f"❌ Query failed: {e}")
    except Exception as e:
        print(f"❌ Client initialization failed: {e}")

# Single Query Mode
async def run_single_query(query: str):
    """Run single query mode with enhanced error handling"""
    try:
        # Try to use the universal client's single query mode
        try:
            from universal_client import single_query_mode
            exit_code = await single_query_mode(query)
            sys.exit(exit_code)
        except ImportError:
            logger.warning("Could not import single_query_mode, using fallback")
            await _fallback_single_query(query)
    except Exception as e:
        print(f"❌ Single query mode failed: {e}")
        sys.exit(1)

async def _fallback_single_query(query: str):
    """Fallback single query implementation"""
    try:
        from dotenv import load_dotenv
        load_dotenv()
        
        config = {
            'server_url': os.getenv('SUBSCRIPTION_API_URL', 'https://subscription-analysis-production.up.railway.app'),
            'api_key': os.getenv('API_KEY_1') or os.getenv('SUBSCRIPTION_API_KEY'),
            'gemini_api_key': os.getenv('GEMINI_API_KEY')
        }
        
        missing = []
        if not config['api_key']:
            missing.append('API_KEY_1')
        if not config['gemini_api_key']:
            missing.append('GEMINI_API_KEY')
        
        if missing:
            print(f"❌ Missing environment variables: {missing}")
            sys.exit(1)
        
        async with UniversalClient(**config) as client:
            result = await client.query_formatted(query)
            print(result)
            sys.exit(0)
    except Exception as e:
        print(f"❌ Query execution failed: {e}")
        sys.exit(1)

# Main Function
async def main():
    """Main entry point - ENHANCED VERSION"""
    # Configure logging
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
🎯 Universal Client MCP Bridge - ENHANCED VERSION

Usage:
  python mcp_client.py           # Interactive mode
  python mcp_client.py --mcp     # MCP server mode (for Claude Desktop)
  python mcp_client.py "query"   # Single query mode
  python mcp_client.py --debug   # Enable debug logging

Examples:
  python mcp_client.py --mcp
  python mcp_client.py "show me database status"
  python mcp_client.py "analytics from 2024-06-01 to 2024-12-11"
  python mcp_client.py "compare 7 days vs 30 days performance"

Features:
  ✅ Natural language queries with Gemini AI
  ✅ Date range analytics (specific dates)
  ✅ Period comparisons (last N days)
  ✅ Beautiful formatted output
  ✅ Claude Desktop integration via MCP
  ✅ Enhanced error handling and debugging
  ✅ Fallback mechanisms for reliability
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