# client/mcp_client.py

import asyncio
import logging
import sys
import os
from typing import Any, Dict, List

# --- MCP Imports (v1.9.4 Compatible) ---
try:
    from mcp.server import Server, InitializationOptions
    from mcp.types import TextContent, Tool
    import mcp.server.stdio
    print("✅ MCP v1.9.4 imports successful", file=sys.stderr)
except ImportError as e:
    print(f"❌ MCP import failed: {e}", file=sys.stderr)
    print("Available MCP components:", file=sys.stderr)
    try:
        import mcp.types
        print(f"mcp.types contents: {[x for x in dir(mcp.types) if not x.startswith('_')]}", file=sys.stderr)
    except:
        pass
    sys.exit(1)

# --- Local Package Imports ---
try:
    from .universal_client import UniversalClient, QueryResult
    from .config_manager import ConfigManager
    print("✅ Project imports successful", file=sys.stderr)
except ImportError as e:
    print(f"❌ Project import failed: {e}", file=sys.stderr)
    print(f"Current working directory: {os.getcwd()}", file=sys.stderr)
    sys.exit(1)

# --- Basic Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("mcp-server-v1.9.4")

# --- The Enhanced MCP Server Class ---
class UniversalClientMCPServer:
    """Enhanced MCP Server wrapper compatible with MCP v1.9.4."""
    
    def __init__(self):
        self.server = Server("subscription-analytics-v1.9.4")
        self.universal_client: UniversalClient | None = None
        self.last_dynamic_result: QueryResult | None = None
        self.config = {}
        self.learning_stats = {"positive_feedback": 0, "negative_feedback": 0, "total_queries": 0}
        
        # Enhanced tool definitions compatible with v1.9.4
        self.tools = [
            Tool(
                name="natural_language_query",
                description="""
The primary tool for subscription analytics. Processes any natural language query about subscription data.

Features:
- Uses pre-built functions for common queries (fast, reliable)
- Generates custom SQL for complex analytics (flexible, powerful) 
- Context-aware: understands follow-up questions like "show me users higher than that"
- Auto-fixes SQL syntax errors (SQLite → MySQL conversion)
- Learns from both positive and negative feedback

Examples:
• "What's the payment success rate for the last month?"
• "How many users have success rate above 15%?"
• "Show me their individual failure rates"
• "How many payments in the same timeframe?"

The system will intelligently choose between pre-built tools or generate custom SQL as needed.
""",
                inputSchema={
                    "type": "object", 
                    "properties": {
                        "query": {
                            "type": "string", 
                            "description": "The user's question in plain English about subscription or payment analytics."
                        }
                    }, 
                    "required": ["query"]
                }
            ),
            Tool(
                name="record_feedback",
                description="""
Record user feedback on a dynamically generated query result.

IMPORTANT: Only call this tool AFTER the user explicitly provides feedback (yes/no/good/bad) 
on an answer that was generated using custom SQL (marked as "DYNAMIC QUERY RESULT").

The system learns from BOTH positive and negative feedback:
- Positive feedback: Stores successful (question, SQL) patterns for future use
- Negative feedback: Remembers failed approaches to avoid similar mistakes

This enhances the AI's ability to generate better queries over time.
""",
                inputSchema={
                    "type": "object", 
                    "properties": {
                        "was_helpful": {
                            "type": "boolean",
                            "description": "True if the user found the answer helpful/accurate, False if not helpful/inaccurate"
                        }
                    }, 
                    "required": ["was_helpful"]
                }
            ),
            Tool(
                name="get_learning_stats",
                description="""
Get statistics about the system's learning progress and feedback history.

Shows:
- Total queries processed
- Number of positive vs negative feedback examples
- Learning system status
- Recent feedback trends

Useful for understanding how the system is improving over time.
""",
                inputSchema={
                    "type": "object",
                    "properties": {},
                    "required": []
                }
            )
        ]
        self._register_handlers()
    
    async def _initialize_client(self):
        """Initializes the config and the client on first use."""
        if not self.universal_client:
            logger.info("Initializing enhanced configuration and UniversalClient for MCP server...")
            config_manager = ConfigManager()
            self.config = config_manager.get_config()
            
            # Set the environment variable before initializing the client
            os.environ['GOOGLE_API_KEY'] = self.config['GOOGLE_API_KEY']
            
            self.universal_client = UniversalClient(config=self.config)
            await self.universal_client.__aenter__()
            logger.info("Enhanced UniversalClient with negative feedback learning initialized.")

    def _register_handlers(self):
        """Registers handlers for listing and calling tools."""
        @self.server.list_tools()
        async def handle_list_tools() -> List[Tool]:
            return self.tools
        
        @self.server.call_tool()
        async def handle_tool_call(name: str, args: Dict[str, Any]) -> List[TextContent]:
            logger.info(f"🔧 Agent called tool: {name}")
            await self._initialize_client()
            handler_name = f"_handle_{name}"
            try:
                if hasattr(self, handler_name):
                    handler = getattr(self, handler_name)
                    result_text = await handler(args)
                else:
                    result_text = f"❌ Error: Unknown tool '{name}'."
                return [TextContent(text=result_text)]
            except Exception as e:
                logger.error(f"Error handling tool '{name}': {e}", exc_info=True)
                return [TextContent(text=f"❌ An internal error occurred: {e}")]

    async def _handle_natural_language_query(self, args: Dict[str, Any]) -> str:
        """Handles the main query tool call from the agent with enhanced features."""
        query = args.get("query")
        if not query: 
            return "❌ Error: 'query' parameter is missing."
        
        # Increment query counter
        self.learning_stats["total_queries"] += 1
        
        logger.info(f"🤔 Processing query: {query[:50]}...")
        result_obj = await self.universal_client.query(query)
        formatter = self.universal_client.formatter
        
        # Format the result
        if isinstance(result_obj, list):
            formatted_data = formatter.format_multi_result(result_obj, query)
        else:
            formatted_data = formatter.format_single_result(result_obj)
        
        # Handle dynamic query results with enhanced feedback instructions
        if isinstance(result_obj, QueryResult) and result_obj.is_dynamic and result_obj.success and result_obj.data is not None:
            self.last_dynamic_result = result_obj
            
            # Enhanced feedback instructions for dynamic queries
            feedback_instructions = """

---
**🧠 LEARNING OPPORTUNITY**

This answer was generated using a custom SQL query. The system can learn from your feedback:

**To help the AI improve:**
- If the answer is **helpful and accurate**: Please ask the user if it was helpful, then call `record_feedback` with `was_helpful: true`
- If the answer is **wrong or unhelpful**: Please ask the user if it was helpful, then call `record_feedback` with `was_helpful: false`

**The system learns from BOTH positive and negative feedback** to:
✅ Remember successful query patterns
❌ Avoid failed approaches for similar questions
🎯 Generate better SQL over time

**Example user interaction:**
User: "Was this answer helpful and accurate?"
If user says yes → call record_feedback(was_helpful=true)
If user says no → call record_feedback(was_helpful=false)"""
            
            return formatted_data + feedback_instructions
        else:
            # For pre-built tool results, no feedback needed
            self.last_dynamic_result = None
            return formatted_data

    async def _handle_record_feedback(self, args: Dict[str, Any]) -> str:
        """Handles the enhanced feedback tool call from the agent."""
        was_helpful = args.get("was_helpful")
        if was_helpful is None: 
            return "❌ Error: 'was_helpful' parameter is missing."
        
        if not self.last_dynamic_result: 
            return "⚠️ No recent dynamic query found to record feedback for. Feedback can only be recorded for custom SQL queries."
        
        # Update learning stats
        if was_helpful:
            self.learning_stats["positive_feedback"] += 1
        else:
            self.learning_stats["negative_feedback"] += 1
        
        # Submit feedback to the backend
        try:
            await self.universal_client.submit_feedback(self.last_dynamic_result, was_helpful)
            self.last_dynamic_result = None
            
            if was_helpful:
                return """✅ **Positive feedback recorded!** 

The system has learned from this successful query. It will:
- Remember this (question → SQL) pattern for future similar questions
- Use this as a positive example when generating new queries
- Help improve accuracy for related analytics requests

Thank you for helping the AI learn! 🎓"""
            else:
                return """📝 **Negative feedback recorded!**

The system has learned from this unsuccessful query. It will:
- Remember to avoid this SQL pattern for similar questions  
- Use this as a negative example to prevent similar mistakes
- Generate alternative approaches for related queries in the future

Your feedback helps the AI improve - thank you! 🔄"""
                
        except Exception as e:
            logger.error(f"Error submitting feedback: {e}")
            return f"⚠️ Feedback noted locally but couldn't sync with server: {e}"

    async def _handle_get_learning_stats(self, args: Dict[str, Any]) -> str:
        """Handles the learning statistics tool call."""
        try:
            # Get server-side learning stats if available
            if self.universal_client:
                # Try to get health info from server which includes learning stats
                health_result = await self.universal_client.call_tool('get_database_status', {})
                
                stats_output = ["📊 **LEARNING SYSTEM STATISTICS**", "=" * 40]
                
                # Local session stats
                stats_output.extend([
                    f"**This Session:**",
                    f"• Total queries processed: {self.learning_stats['total_queries']}",
                    f"• Positive feedback given: {self.learning_stats['positive_feedback']}",
                    f"• Negative feedback given: {self.learning_stats['negative_feedback']}",
                    ""
                ])
                
                # Server-side stats if available
                if health_result.success and health_result.data:
                    server_data = health_result.data
                    if 'learning_stats' in server_data:
                        learning_stats = server_data['learning_stats']
                        stats_output.extend([
                            f"**Overall System Learning:**",
                            f"• Total learned queries: {learning_stats.get('total_learned_queries', 'N/A')}",
                            f"• Positive examples stored: {learning_stats.get('positive_examples', 'N/A')}",
                            f"• Negative examples stored: {learning_stats.get('negative_examples', 'N/A')}",
                            f"• Semantic learning: {server_data.get('semantic_learning', 'Unknown')}",
                            ""
                        ])
                    
                    stats_output.extend([
                        f"**System Status:**",
                        f"• Server status: {server_data.get('status', 'Unknown')}",
                        f"• Available tools: {server_data.get('available_tools', 'N/A')}",
                        f"• Stability level: {server_data.get('stability', 'Unknown')}",
                        f"• Last updated: {server_data.get('timestamp', 'Unknown')}"
                    ])
                
                return "\n".join(stats_output)
            else:
                return "⚠️ Learning statistics not available - client not initialized."
                
        except Exception as e:
            logger.error(f"Error getting learning stats: {e}")
            return f"❌ Error retrieving learning statistics: {e}"

    async def cleanup(self):
        """Enhanced cleanup with learning statistics summary."""
        if self.universal_client:
            await self.universal_client.__aexit__(None, None, None)
            logger.info("✅ Enhanced Universal client cleaned up.")
        
        # Log session learning summary
        total_feedback = self.learning_stats['positive_feedback'] + self.learning_stats['negative_feedback']
        logger.info(f"📊 Session Summary: {self.learning_stats['total_queries']} queries, "
                   f"{total_feedback} feedback responses ({self.learning_stats['positive_feedback']} positive, "
                   f"{self.learning_stats['negative_feedback']} negative)")

# --- Main Execution Block ---
async def main():
    print("🌉 Starting Enhanced MCP Server (v1.9.4 Compatible) with Negative Feedback Learning", file=sys.stderr)
    print(f"🐍 Python: {sys.executable}", file=sys.stderr)
    print(f"📁 Working dir: {os.getcwd()}", file=sys.stderr)
    
    server_instance = UniversalClientMCPServer()
    try:
        async with mcp.server.stdio.stdio_server() as (reader, writer):
            print("🚀 Enhanced MCP Server is running and ready for Claude to connect.", file=sys.stderr)
            print("🧠 Features: Context awareness, auto-fix, positive & negative feedback learning", file=sys.stderr)
            
            # MCP v1.9.4 requires InitializationOptions with capabilities
            initialization_options = InitializationOptions(
                server_name="subscription-analytics-v1.9.4",
                server_version="2.0.0",
                capabilities={}  # Empty capabilities for basic server
            )
            
            await server_instance.server.run(reader, writer, initialization_options)
    except Exception as e:
        print(f"❌ Critical error in Enhanced MCP Server: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc(file=sys.stderr)
    finally:
        await server_instance.cleanup()
        print("👋 Enhanced MCP Server shutdown complete.", file=sys.stderr)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Enhanced MCP Server shut down by user.", file=sys.stderr)
    except Exception as e:
        print(f"Fatal error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)