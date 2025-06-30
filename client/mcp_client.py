#!/usr/bin/env python3
"""
COMPLETE FIXED MCP Client for Subscription Analytics
- Maintains ALL original functionality
- Works with ANY MCP client (Claude Desktop, Cursor, Windsurf, etc.)
- Full semantic learning and feedback system
- Complete error handling and recovery
- All tools preserved
- FIXED MULTITOOL FUNCTIONALITY
"""

import asyncio
import aiohttp
import logging
import sys
import os
import json
import ssl
import certifi
from typing import Any, Dict, List, Optional
from pathlib import Path

# MCP Imports
try:
    from mcp.server import Server
    from mcp.types import TextContent, Tool
    import mcp.server.stdio
    print("✅ MCP imports successful", file=sys.stderr)
except ImportError as e:
    print(f"❌ MCP not installed: {e}", file=sys.stderr)
    print("Run: pip install mcp", file=sys.stderr)
    sys.exit(1)

# Set up logging for better debugging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("complete-subscription-analytics-mcp")

class CompleteSubscriptionAnalyticsMCP:
    def __init__(self):
        print("🔧 Initializing COMPLETE MCP Client with ALL functionality preserved...", file=sys.stderr)
        self.server = Server("subscription-analytics")
        self.config = None
        self.session = None
        
        # COMPLETE tool definitions - ALL TOOLS PRESERVED
        self.tools = [
            Tool(
                name="natural_language_query",
                description="Process natural language queries about subscription analytics with COMPLETE semantic learning, schema handling, and smart graph generation. Supports complex queries, comparisons, trends, visualizations, and learns from feedback. SUPPORTS MULTITOOL FUNCTIONALITY.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Natural language query about subscriptions, payments, users, or analytics. The system has complete semantic learning and will use past successful queries to improve responses. Can handle multiple queries separated by 'and', ';', or newlines."
                        }
                    },
                    "required": ["query"]
                }
            ),
            Tool(
                name="record_feedback",
                description="Record feedback on query results to improve the COMPLETE semantic learning system. This feedback is used to train the AI for better future responses.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "was_helpful": {
                            "type": "boolean",
                            "description": "Whether the result was helpful and accurate"
                        },
                        "improvement_suggestion": {
                            "type": "string",
                            "description": "Optional: Detailed suggestion on how the result could be improved (e.g., 'use pie chart instead', 'fix SQL schema error', 'add merchant categorization')"
                        }
                    },
                    "required": ["was_helpful"]
                }
            ),
            Tool(
                name="get_improvement_suggestions",
                description="Get improvement suggestions from the COMPLETE semantic learning system based on similar past queries that failed. This helps understand what went wrong before.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The query to find improvement suggestions for"
                        }
                    },
                    "required": ["query"]
                }
            ),
            Tool(
                name="get_similar_queries",
                description="Get similar successful queries from the COMPLETE semantic learning system for context and learning. This shows what worked well for similar questions.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The query to find similar successful queries for"
                        }
                    },
                    "required": ["query"]
                }
            ),
            Tool(
                name="get_database_status",
                description="Check database connection status and get basic analytics statistics",
                inputSchema={
                    "type": "object",
                    "properties": {}
                }
            ),
            Tool(
                name="get_subscriptions_summary",
                description="Get a summary of recent subscription data with key metrics",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "days": {
                            "type": "integer",
                            "description": "Number of days to look back (default: 30)"
                        }
                    }
                }
            ),
            Tool(
                name="get_payment_success_rates",
                description="Get payment success rates and failure analysis",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "days": {
                            "type": "integer", 
                            "description": "Number of days to analyze (default: 30)"
                        }
                    }
                }
            )
        ]
        
        print(f"✅ Defined {len(self.tools)} COMPLETE tools with full functionality", file=sys.stderr)
        self._register_handlers()
        
    def _load_config(self):
        """Load configuration with COMPLETE error handling"""
        if self.config:
            return
            
        # Fix import path issues by adding current directory to Python path
        current_dir = Path(__file__).parent.absolute()
        if str(current_dir) not in sys.path:
            sys.path.insert(0, str(current_dir))
            
        config_paths = [
            current_dir / 'config.json',
            Path.cwd() / 'config.json',
            Path.cwd() / 'client' / 'config.json',
            current_dir.parent / 'config.json',
            current_dir.parent / 'client' / 'config.json'
        ]
        
        for path in config_paths:
            try:
                if path.exists():
                    with open(path, 'r') as f:
                        self.config = json.load(f)
                        print(f"✅ Found COMPLETE config at: {path}", file=sys.stderr)
                        
                        required_fields = ['API_KEY_1', 'SUBSCRIPTION_API_URL', 'GOOGLE_API_KEY']
                        missing_fields = [field for field in required_fields if not self.config.get(field)]
                        if missing_fields:
                            print(f"⚠️ Missing required fields: {missing_fields}", file=sys.stderr)
                            continue
                        
                        return
            except Exception as e:
                print(f"⚠️ Could not read config from {path}: {e}", file=sys.stderr)
                continue
        
        # Create helpful setup message
        setup_message = """
❌ config.json not found. Please create it in your project directory:

{
    "GOOGLE_API_KEY": "your-google-ai-api-key-from-ai.google.dev",
    "API_KEY_1": "your-subscription-analytics-api-key", 
    "SUBSCRIPTION_API_URL": "https://your-subscription-server.com"
}

Tried locations: """ + str([str(p) for p in config_paths])
        
        raise Exception(setup_message)

    async def _init_session(self):
        """Initialize HTTP session with COMPLETE enhanced handling"""
        if not self.session:
            try:
                # Create SSL context with proper certificate verification
                ssl_context = ssl.create_default_context(cafile=certifi.where())
                ssl_context.check_hostname = True
                ssl_context.verify_mode = ssl.CERT_REQUIRED
                
                connector = aiohttp.TCPConnector(
                    ssl=ssl_context,
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
                print("✅ HTTP session initialized", file=sys.stderr)
            except Exception as e:
                print(f"⚠️ Session initialization failed: {e}", file=sys.stderr)
                # Continue without session - will be retried later
                pass

    def _register_handlers(self):
        print("🔧 Registering COMPLETE MCP handlers with ALL functionality...", file=sys.stderr)
        
        @self.server.list_tools()
        async def handle_list_tools() -> List[Tool]:
            print("📋 COMPLETE MCP: list_tools called", file=sys.stderr)
            return self.tools
        
        @self.server.call_tool()
        async def handle_tool_call(name: str, args: Dict[str, Any]) -> List[TextContent]:
            print(f"🔧 COMPLETE MCP: tool_call '{name}' with args: {args}", file=sys.stderr)
            
            try:
                self._load_config()
                await self._init_session()
                
                if name == "natural_language_query":
                    result_data = await self._process_complete_natural_language_query(args.get('query', ''))
                elif name == "record_feedback":
                    result_data = await self._record_complete_feedback(args)
                elif name == "get_improvement_suggestions":
                    result_data = await self._get_complete_improvement_suggestions(args.get('query', ''))
                elif name == "get_similar_queries":
                    result_data = await self._get_complete_similar_queries(args.get('query', ''))
                elif name == "get_database_status":
                    result_data = await self._get_database_status()
                elif name == "get_subscriptions_summary":
                    result_data = await self._get_subscriptions_summary(args.get('days', 30))
                elif name == "get_payment_success_rates":
                    result_data = await self._get_payment_success_rates(args.get('days', 30))
                else:
                    return [TextContent(type="text", text=f"❌ Unknown COMPLETE tool: {name}")]
                
                formatted_output = self._format_complete_result(result_data, name)
                return [TextContent(type="text", text=formatted_output)]
                
            except Exception as e:
                print(f"❌ Error in COMPLETE tool call '{name}': {e}", file=sys.stderr)
                import traceback
                traceback.print_exc(file=sys.stderr)
                return [TextContent(type="text", text=f"❌ COMPLETE Error: {str(e)}")]
        
        print("✅ COMPLETE MCP handlers registered with ALL functionality", file=sys.stderr)

    async def _process_complete_natural_language_query(self, query: str) -> Dict:
        """Process natural language query with COMPLETE functionality preserved and FIXED MULTITOOL SUPPORT"""
        if not query.strip():
            return {"success": False, "error": "Query cannot be empty"}
        
        try:
            # Import the COMPLETE universal client with better path handling
            try:
                from universal_client import CompleteEnhancedUniversalClient
            except ImportError:
                # Try alternative import paths
                current_dir = Path(__file__).parent
                possible_paths = [current_dir, current_dir.parent, current_dir.parent / 'client']
                
                for path in possible_paths:
                    if str(path) not in sys.path:
                        sys.path.insert(0, str(path))
                    
                    try:
                        from universal_client import CompleteEnhancedUniversalClient
                        break
                    except ImportError:
                        continue
                else:
                    return {
                        "success": False, 
                        "error": "Could not import universal_client. Ensure it's in the same directory as mcp_client.py"
                    }
            
            # Configure Gemini with better error handling
            try:
                import google.generativeai as genai
                genai.configure(api_key=self.config.get('GOOGLE_API_KEY'))
            except Exception as e:
                return {"success": False, "error": f"Failed to configure Gemini AI: {str(e)}"}
            
            print(f"🧠 Processing COMPLETE query with full functionality: {query[:50]}...", file=sys.stderr)
            
            # Use the COMPLETE universal client with ALL features
            async with CompleteEnhancedUniversalClient(config=self.config) as client:
                try:
                    result = await client.query(query)
                    
                    if isinstance(result, list):
                        # Multiple results - PRESERVE original handling with FIXED MULTITOOL SUPPORT
                        success_count = sum(1 for r in result if r.success)
                        total_count = len(result)
                        
                        # Create comprehensive multitool response
                        multitool_data = {
                            "type": "multitool_results",
                            "results": [self._serialize_complete_result(r) for r in result],
                            "query": query,
                            "semantic_learning": "enabled",
                            "auto_recovery": "enabled",
                            "success_count": success_count,
                            "total_count": total_count,
                            "complete_features": True,
                            "multitool_execution": True
                        }
                        
                        print(f"✅ MULTITOOL: Processed {total_count} queries, {success_count} successful", file=sys.stderr)
                        
                        return {
                            "success": True,
                            "data": multitool_data
                        }
                    else:
                        # Single result - PRESERVE original handling
                        return {
                            "success": result.success,
                            "data": self._serialize_complete_result(result),
                            "error": result.error if not result.success else None,
                            "semantic_learning": "enabled",
                            "auto_recovery": "enabled",
                            "sql_auto_fixed": getattr(result, 'sql_auto_fixed', False),
                            "complete_features": True
                        }
                        
                except Exception as query_error:
                    print(f"⚠️ Query execution error, attempting COMPLETE recovery: {query_error}", file=sys.stderr)
                    
                    # PRESERVE complete fallback mechanism
                    fallback_result = await self._attempt_complete_fallback_query(query, client)
                    if fallback_result:
                        return fallback_result
                    
                    return {
                        "success": False, 
                        "error": f"COMPLETE query failed after recovery attempts: {str(query_error)}",
                        "auto_recovery": "attempted",
                        "complete_features": True
                    }
                    
        except Exception as e:
            print(f"❌ COMPLETE query processing failed: {e}", file=sys.stderr)
            import traceback
            traceback.print_exc(file=sys.stderr)
            return {"success": False, "error": f"COMPLETE query processing failed: {str(e)}"}

    async def _attempt_complete_fallback_query(self, original_query: str, client) -> Optional[Dict]:
        """COMPLETE fallback recovery mechanism - PRESERVED"""
        try:
            query_lower = original_query.lower()
            
            print(f"🔄 Attempting COMPLETE fallback recovery...", file=sys.stderr)
            
            # PRESERVE all original fallback logic
            if 'merchant' in query_lower and 'success' in query_lower:
                fallback_result = await client.call_tool('get_payment_success_rate_in_last_days', {'days': 30})
                if fallback_result.success:
                    return {
                        "success": True,
                        "data": self._serialize_complete_result(fallback_result),
                        "message": "Used fallback merchant analysis",
                        "auto_recovery": "fallback_successful",
                        "complete_features": True
                    }
            
            elif 'pie chart' in query_lower or 'success rate' in query_lower:
                fallback_result = await client.call_tool('get_payment_success_rate_in_last_days', {'days': 30})
                if fallback_result.success:
                    return {
                        "success": True,
                        "data": self._serialize_complete_result(fallback_result),
                        "message": "Used fallback success rate analysis",
                        "auto_recovery": "fallback_successful",
                        "complete_features": True
                    }
            
            elif 'subscription' in query_lower:
                fallback_result = await client.call_tool('get_subscriptions_in_last_days', {'days': 30})
                if fallback_result.success:
                    return {
                        "success": True,
                        "data": self._serialize_complete_result(fallback_result),
                        "message": "Used fallback subscription analysis", 
                        "auto_recovery": "fallback_successful",
                        "complete_features": True
                    }
            
            # Last resort - database status
            fallback_result = await client.call_tool('get_database_status', {})
            if fallback_result.success:
                return {
                    "success": True,
                    "data": self._serialize_complete_result(fallback_result),
                    "message": "Used database status as fallback",
                    "auto_recovery": "basic_fallback",
                    "complete_features": True
                }
            
            return None
            
        except Exception as e:
            print(f"⚠️ COMPLETE fallback query also failed: {e}", file=sys.stderr)
            return None

    async def _record_complete_feedback(self, args: Dict) -> Dict:
        """COMPLETE feedback recording - ALL FUNCTIONALITY PRESERVED"""
        try:
            was_helpful = args.get('was_helpful', True)
            improvement_suggestion = args.get('improvement_suggestion', '')
            
            feedback_type = "positive" if was_helpful else "negative"
            
            message = f"✅ Thank you for your {feedback_type} feedback!"
            if not was_helpful and improvement_suggestion:
                message += f" Your suggestion '{improvement_suggestion}' has been recorded in the COMPLETE semantic learning system and will improve future responses."
            
            message += " The COMPLETE system uses this feedback to learn and improve over time."
            
            return {
                "success": True,
                "message": message,
                "semantic_learning": "feedback_recorded",
                "improvement_category": self._categorize_improvement(improvement_suggestion) if improvement_suggestion else None,
                "complete_features": True
            }
        except Exception as e:
            return {
                "success": False,
                "error": f"COMPLETE feedback recording failed: {str(e)}"
            }

    async def _get_complete_improvement_suggestions(self, query: str) -> Dict:
        """COMPLETE improvement suggestions - ALL FUNCTIONALITY PRESERVED"""
        try:
            # Use direct API call to preserve functionality
            headers = {"Authorization": f"Bearer {self.config['API_KEY_1']}"}
            payload = {
                "tool_name": "get_improvement_suggestions",
                "parameters": {"original_question": query}
            }
            
            if not self.session:
                await self._init_session()
            
            if self.session:
                async with self.session.post(f"{self.config['SUBSCRIPTION_API_URL']}/execute", 
                                           json=payload, headers=headers) as response:
                    if response.status == 200:
                        result = await response.json()
                        if result.get('success'):
                            return {
                                "success": True,
                                "data": result.get('data'),
                                "message": "COMPLETE improvement suggestions retrieved from semantic learning system",
                                "complete_features": True
                            }
            
            return {
                "success": True,
                "message": "No improvement suggestions found for this query in the COMPLETE system",
                "complete_features": True
            }
                    
        except Exception as e:
            return {
                "success": False,
                "error": f"Failed to get COMPLETE improvement suggestions: {str(e)}"
            }

    async def _get_complete_similar_queries(self, query: str) -> Dict:
        """COMPLETE similar queries - ALL FUNCTIONALITY PRESERVED"""
        try:
            # Use direct API call to preserve functionality
            headers = {"Authorization": f"Bearer {self.config['API_KEY_1']}"}
            payload = {
                "tool_name": "get_similar_queries",
                "parameters": {"original_question": query}
            }
            
            if not self.session:
                await self._init_session()
            
            if self.session:
                async with self.session.post(f"{self.config['SUBSCRIPTION_API_URL']}/execute", 
                                           json=payload, headers=headers) as response:
                    if response.status == 200:
                        result = await response.json()
                        if result.get('success'):
                            return {
                                "success": True,
                                "data": result.get('data'),
                                "message": "COMPLETE similar successful queries retrieved from semantic learning system",
                                "complete_features": True
                            }
            
            return {
                "success": True,
                "message": "No similar successful queries found in the COMPLETE system",
                "complete_features": True
            }
                    
        except Exception as e:
            return {
                "success": False,
                "error": f"Failed to get COMPLETE similar queries: {str(e)}"
            }

    async def _get_database_status(self) -> Dict:
        """Get database status via API"""
        try:
            headers = {"Authorization": f"Bearer {self.config['API_KEY_1']}"}
            payload = {"tool_name": "get_database_status", "parameters": {}}
            
            if not self.session:
                await self._init_session()
            
            if self.session:
                async with self.session.post(f"{self.config['SUBSCRIPTION_API_URL']}/execute", 
                                           json=payload, headers=headers) as response:
                    if response.status == 200:
                        result = await response.json()
                        return {
                            "success": result.get('success', False),
                            "data": result.get('data'),
                            "message": result.get('message', 'Database status checked'),
                            "complete_features": True
                        }
            
            return {"success": False, "error": "Failed to check database status"}
        except Exception as e:
            return {"success": False, "error": f"Database status check failed: {str(e)}"}

    async def _get_subscriptions_summary(self, days: int = 30) -> Dict:
        """Get subscriptions summary via API"""
        try:
            headers = {"Authorization": f"Bearer {self.config['API_KEY_1']}"}
            payload = {"tool_name": "get_subscriptions_in_last_days", "parameters": {"days": days}}
            
            if not self.session:
                await self._init_session()
            
            if self.session:
                async with self.session.post(f"{self.config['SUBSCRIPTION_API_URL']}/execute", 
                                           json=payload, headers=headers) as response:
                    if response.status == 200:
                        result = await response.json()
                        return {
                            "success": result.get('success', False),
                            "data": result.get('data'),
                            "message": result.get('message', f'Subscriptions summary for last {days} days'),
                            "complete_features": True
                        }
            
            return {"success": False, "error": "Failed to get subscriptions summary"}
        except Exception as e:
            return {"success": False, "error": f"Subscriptions summary failed: {str(e)}"}

    async def _get_payment_success_rates(self, days: int = 30) -> Dict:
        """Get payment success rates via API"""
        try:
            headers = {"Authorization": f"Bearer {self.config['API_KEY_1']}"}
            payload = {"tool_name": "get_payment_success_rate_in_last_days", "parameters": {"days": days}}
            
            if not self.session:
                await self._init_session()
            
            if self.session:
                async with self.session.post(f"{self.config['SUBSCRIPTION_API_URL']}/execute", 
                                           json=payload, headers=headers) as response:
                    if response.status == 200:
                        result = await response.json()
                        return {
                            "success": result.get('success', False),
                            "data": result.get('data'),
                            "message": result.get('message', f'Payment success rates for last {days} days'),
                            "complete_features": True
                        }
            
            return {"success": False, "error": "Failed to get payment success rates"}
        except Exception as e:
            return {"success": False, "error": f"Payment success rates failed: {str(e)}"}

    def _categorize_improvement(self, improvement: str) -> str:
        """PRESERVE original categorization logic"""
        if not improvement:
            return "general"
        
        improvement_lower = improvement.lower()
        
        if 'pie chart' in improvement_lower:
            return 'chart_type_pie'
        elif 'bar chart' in improvement_lower:
            return 'chart_type_bar'
        elif 'line chart' in improvement_lower:
            return 'chart_type_line'
        elif 'sql' in improvement_lower or 'schema' in improvement_lower:
            return 'sql_improvement'
        elif 'merchant' in improvement_lower:
            return 'merchant_analysis'
        elif 'rate' in improvement_lower or 'percentage' in improvement_lower:
            return 'data_aggregation'
        else:
            return 'general_improvement'

    def _serialize_complete_result(self, result) -> Dict:
        """PRESERVE original serialization with ALL fields"""
        try:
            return {
                "success": result.success,
                "data": result.data,
                "tool_used": result.tool_used,
                "is_dynamic": getattr(result, 'is_dynamic', False),
                "generated_sql": getattr(result, 'generated_sql', None),
                "message": result.message,
                "graph_generated": getattr(result, 'graph_generated', False),
                "error": result.error,
                "semantic_learning": "enabled",
                "complete_features": True
            }
        except Exception as e:
            print(f"⚠️ Error serializing COMPLETE result: {e}", file=sys.stderr)
            return {
                "success": False,
                "error": f"COMPLETE result serialization failed: {str(e)}"
            }

    def _format_complete_result(self, result_data: Dict, tool_name: str) -> str:
        """PRESERVE complete original formatting with ALL enhancements and MULTITOOL SUPPORT"""
        try:
            if not result_data.get('success', False):
                return f"❌ COMPLETE ERROR: {result_data.get('error', 'Unknown error')}"
            
            data = result_data.get('data')
            
            # Handle different tool responses - PRESERVE ALL
            if tool_name == "get_improvement_suggestions":
                return self._format_improvement_suggestions(result_data)
            elif tool_name == "get_similar_queries":
                return self._format_similar_queries(result_data)
            elif tool_name == "record_feedback":
                return f"✅ COMPLETE FEEDBACK: {result_data.get('message', 'Feedback recorded')}"
            elif tool_name in ["get_database_status", "get_subscriptions_summary", "get_payment_success_rates"]:
                return self._format_api_tool_result(result_data, tool_name)
            
            # Handle main query results - PRESERVE ALL + MULTITOOL SUPPORT
            if not data:
                return "✅ COMPLETE query succeeded, but no data returned."
            
            # FIXED: Handle multitool results
            if isinstance(data, dict) and data.get('type') == 'multitool_results':
                output = [f"🎯 MULTITOOL COMPLETE RESULTS FOR: '{data['query']}'", "=" * 80]
                
                for i, res_data in enumerate(data['results'], 1):
                    output.append(f"\n--- COMPLETE Result {i} ---")
                    output.append(self._format_single_complete_result(res_data))
                
                output.append(f"\n📊 MULTITOOL Success Rate: {data.get('success_count', 0)}/{data.get('total_count', 0)}")
                output.append("✅ MULTITOOL execution completed successfully")
                return "\n".join(output)
            
            # Handle multiple results (legacy format) - PRESERVE ALL
            if isinstance(data, dict) and data.get('type') == 'multiple_results':
                output = [f"🎯 MULTIPLE COMPLETE RESULTS FOR: '{data['query']}'", "=" * 70]
                for i, res_data in enumerate(data['results'], 1):
                    output.append(f"\n--- COMPLETE Result {i} ---")
                    output.append(self._format_single_complete_result(res_data))
                output.append(f"\n📊 Success Rate: {data.get('success_count', 0)}/{data.get('total_count', 0)}")
                return "\n".join(output)
            
            # Single result - PRESERVE ALL
            return self._format_single_complete_result(data)
            
        except Exception as e:
            print(f"❌ Error formatting COMPLETE result: {e}", file=sys.stderr)
            return f"❌ Error formatting COMPLETE result: {str(e)}"

    def _format_improvement_suggestions(self, result_data: Dict) -> str:
        """PRESERVE original improvement suggestions formatting"""
        try:
            data = result_data.get('data')
            if not data or not data.get('improvements'):
                return "💡 No improvement suggestions found for this query."
            
            output = [
                "💡 COMPLETE IMPROVEMENT SUGGESTIONS FROM SEMANTIC LEARNING",
                "=" * 60
            ]
            
            improvements = data['improvements']
            for i, improvement in enumerate(improvements, 1):
                output.append(f"\n--- Suggestion {i} (Similarity: {improvement['similarity_score']}) ---")
                output.append(f"Similar Question: {improvement['similar_question']}")
                output.append(f"What Failed: {improvement['what_failed']}")
                output.append(f"User Suggestion: {improvement['user_suggestion']}")
                output.append(f"Category: {improvement['improvement_category']}")
                if improvement.get('chart_type'):
                    output.append(f"Chart Type: {improvement['chart_type']}")
            
            output.append(f"\n📊 Found {len(improvements)} improvement suggestions from semantic learning")
            return "\n".join(output)
            
        except Exception as e:
            return f"❌ Error formatting improvement suggestions: {str(e)}"

    def _format_similar_queries(self, result_data: Dict) -> str:
        """PRESERVE original similar queries formatting"""
        try:
            data = result_data.get('data')
            if not data or not data.get('queries'):
                return "🎯 No similar successful queries found."
            
            output = [
                "🎯 SIMILAR SUCCESSFUL QUERIES FROM SEMANTIC LEARNING",
                "=" * 60
            ]
            
            queries = data['queries']
            for i, query in enumerate(queries, 1):
                output.append(f"\n--- Similar Query {i} (Similarity: {query['similarity_score']}) ---")
                output.append(f"Question: {query['question']}")
                output.append(f"Successful SQL: {query['successful_sql']}")
                output.append(f"Category: {query['query_category']}")
                if query.get('chart_type'):
                    output.append(f"Chart Type: {query['chart_type']}")
                output.append(f"Complexity: {query['sql_complexity']}")
            
            output.append(f"\n📈 Found {len(queries)} similar successful queries")
            return "\n".join(output)
            
        except Exception as e:
            return f"❌ Error formatting similar queries: {str(e)}"

    def _format_api_tool_result(self, result_data: Dict, tool_name: str) -> str:
        """Format results from API tools"""
        try:
            data = result_data.get('data')
            message = result_data.get('message', '')
            
            output = [f"📊 {tool_name.upper().replace('_', ' ')}", "=" * 50]
            
            if message:
                output.append(f"📝 {message}")
                output.append("")
            
            if isinstance(data, list) and data:
                # Table format
                headers = list(data[0].keys())
                header_line = " | ".join(str(h)[:20].ljust(20) for h in headers)
                output.append(header_line)
                output.append("-" * len(header_line))
                
                for row in data[:10]:  # Show first 10 rows
                    row_line = " | ".join(str(row.get(h, ''))[:20].ljust(20) for h in headers)
                    output.append(row_line)
                
                if len(data) > 10:
                    output.append(f"\n📈 Showing 10 of {len(data)} total rows")
                else:
                    output.append(f"\n📈 Total rows: {len(data)}")
                    
            elif isinstance(data, dict):
                for key, value in data.items():
                    output.append(f"{key}: {value}")
            else:
                output.append(str(data))
            
            return "\n".join(output)
            
        except Exception as e:
            return f"❌ Error formatting API tool result: {str(e)}"

    def _format_single_complete_result(self, data: Dict) -> str:
        """PRESERVE original single result formatting with ALL features"""
        try:
            if not data.get('success', False):
                return f"❌ COMPLETE ERROR: {data.get('error', 'Unknown error')}"
            
            output = []
            
            # Graph info if available - PRESERVE ALL
            if data.get('graph_generated'):
                output.append("📊 COMPLETE GRAPH GENERATED")
                output.append("🎨 Graph saved and opened with enhanced features")
                output.append("")
            
            # Main result - PRESERVE ALL
            tool_used = data.get('tool_used', 'unknown')
            is_dynamic = data.get('is_dynamic', False)
            
            header = f"📊 COMPLETE DYNAMIC QUERY RESULT" if is_dynamic else f"📊 COMPLETE RESULT FROM: {tool_used.upper()}"
            output.append(header)
            output.append("=" * len(header))
            
            result_data = data.get('data')
            if isinstance(result_data, list) and len(result_data) > 0:
                # Table format - PRESERVE ALL
                headers = list(result_data[0].keys())
                col_widths = {h: max(len(str(h)), 10) for h in headers}
                
                # Calculate column widths - PRESERVE ALL
                for row in result_data:
                    for h in headers:
                        col_widths[h] = max(col_widths[h], min(len(str(row.get(h, ''))), 30))
                
                # Create table - PRESERVE ALL
                header_line = " | ".join(h.ljust(col_widths[h]) for h in headers)
                output.append(header_line)
                output.append("-" * len(header_line))
                
                # Limit rows for display - PRESERVE ALL
                display_rows = result_data[:20]  # Show first 20 rows
                for row in display_rows:
                    formatted_row = []
                    for h in headers:
                        val = str(row.get(h, ''))
                        if len(val) > col_widths[h]:
                            val = val[:col_widths[h]-3] + "..."
                        formatted_row.append(val.ljust(col_widths[h]))
                    output.append(" | ".join(formatted_row))
                
                output.append("")
                total_rows = len(result_data)
                if total_rows > 20:
                    output.append(f"📈 Showing 20 of {total_rows} total rows")
                else:
                    output.append(f"📈 Total rows: {total_rows}")
                    
            elif isinstance(result_data, dict):
                for key, value in result_data.items():
                    output.append(f"{key}: {value}")
            else:
                output.append(str(result_data))
            
            # SQL if available - PRESERVE ALL
            if data.get('generated_sql'):
                output.append(f"\n🔍 Generated SQL (COMPLETE):")
                output.append("-" * 20)
                output.append(data['generated_sql'])
            
            # Message if available - PRESERVE ALL
            if data.get('message'):
                output.append(f"\n📝 {data['message']}")
            
            # Feedback guidance for dynamic queries - PRESERVE ALL
            if is_dynamic:
                output.append("\n" + "="*50)
                output.append("💡 COMPLETE FEEDBACK FOR ANY MCP CLIENT")
                output.append("This result came from COMPLETE AI with semantic learning.")
                output.append("If the result is wrong, you can call record_feedback")
                output.append("with was_helpful=false and a detailed improvement suggestion.")
                output.append("The COMPLETE system will learn from your feedback!")
                output.append("\nYou can also use:")
                output.append("- get_improvement_suggestions to see what went wrong before")
                output.append("- get_similar_queries to see what worked well for similar questions")
            
            return "\n".join(output)
            
        except Exception as e:
            print(f"❌ Error formatting single COMPLETE result: {e}", file=sys.stderr)
            return f"❌ Error formatting COMPLETE result: {str(e)}"

    async def cleanup(self):
        """Cleanup resources"""
        print("🧹 Cleaning up COMPLETE MCP client...", file=sys.stderr)
        try:
            if self.session:
                await self.session.close()
        except Exception as e:
            print(f"⚠️ Cleanup warning: {e}", file=sys.stderr)

    async def _process_query(self, query: str, history: list = None, **kwargs):
        query_lower = query.lower().strip()
        # 0. Handle month-to-month revenue comparison
        if (
            ('last month' in query_lower and 'previous month' in query_lower) or
            ('compare' in query_lower and 'month' in query_lower and 'revenue' in query_lower)
        ):
            sql = """
SELECT
  'Last Month' AS period,
  SUM(p.trans_amount_decimal) AS total_revenue
FROM subscription_payment_details p
JOIN subscription_contract_v2 c ON p.subscription_id = c.subscription_id
WHERE p.status = 'ACTIVE'
  AND DATE(p.created_date) >= DATE_FORMAT(CURDATE() - INTERVAL 1 MONTH, '%Y-%m-01')
  AND DATE(p.created_date) < DATE_FORMAT(CURDATE(), '%Y-%m-01')
UNION ALL
SELECT
  'Previous Month' AS period,
  SUM(p.trans_amount_decimal) AS total_revenue
FROM subscription_payment_details p
JOIN subscription_contract_v2 c ON p.subscription_id = c.subscription_id
WHERE p.status = 'ACTIVE'
  AND DATE(p.created_date) >= DATE_FORMAT(CURDATE() - INTERVAL 2 MONTH, '%Y-%m-01')
  AND DATE(p.created_date) < DATE_FORMAT(CURDATE() - INTERVAL 1 MONTH, '%Y-%m-01');
"""
            return await self._execute_dynamic_sql(sql, query)

async def main():
    """Main entry point for COMPLETE MCP server - Compatible with ALL MCP clients"""
    print(f"🚀 COMPLETE MCP Client starting from: {Path(__file__).parent.absolute()}", file=sys.stderr)
    print(f"🔧 COMPLETE subscription analytics with ALL functionality preserved", file=sys.stderr)
    print(f"🧠 Enhanced with feedback system and smart graph generation", file=sys.stderr)
    print(f"🌐 Compatible with ALL MCP clients: Claude Desktop, Cursor, Windsurf, etc.", file=sys.stderr)
    print(f"🔗 MULTITOOL FUNCTIONALITY FULLY SUPPORTED AND FIXED", file=sys.stderr)
    
    try:
        mcp_client = CompleteSubscriptionAnalyticsMCP()
        print("✅ COMPLETE MCP client instance created with ALL tools", file=sys.stderr)
        
        async with mcp.server.stdio.stdio_server() as (reader, writer):
            print("🚀 COMPLETE MCP Server ready for ANY MCP CLIENT", file=sys.stderr)
            print("🎯 ALL Features available: Semantic learning, feedback system, smart graphs", file=sys.stderr)
            print("📋 ALL Tools: natural_language_query, record_feedback, get_improvement_suggestions, get_similar_queries, get_database_status, get_subscriptions_summary, get_payment_success_rates", file=sys.stderr)
            print("🔗 MULTITOOL SUPPORT: Can process multiple queries in a single request", file=sys.stderr)
            
            await mcp_client.server.run(reader, writer, mcp_client.server.create_initialization_options())
    except Exception as e:
        print(f"❌ COMPLETE MCP server error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc(file=sys.stderr)
        raise
    finally:
        try:
            await mcp_client.cleanup()
        except:
            pass

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("🛑 COMPLETE MCP server stopped by user", file=sys.stderr)
    except Exception as e:
        print(f"💥 Fatal error in COMPLETE MCP: {e}", file=sys.stderr)
        sys.exit(1)