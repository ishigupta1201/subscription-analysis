#!/usr/bin/env python3
"""
Complete Fixed Universal Client with SSL fixes and proper async context management
"""
import asyncio
import aiohttp
import os
import json
import sys
import ssl
import certifi
from typing import Dict, List, Optional, Union
from dataclasses import dataclass
from google import genai
from google.genai import types

# Set up logging
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class QueryResult:
    """Result of a query execution"""
    success: bool
    data: Optional[Dict] = None
    error: Optional[str] = None
    tool_used: Optional[str] = None
    parameters: Optional[Dict] = None

class ResultFormatter:
    """Format results in a beautiful, user-friendly way"""
    
    @staticmethod
    def safe_int(value) -> int:
        """Safely convert any value to int"""
        if isinstance(value, int):
            return value
        if isinstance(value, str):
            # Remove commas and convert
            cleaned = value.replace(',', '').replace('$', '').strip()
            try:
                return int(float(cleaned))
            except (ValueError, TypeError):
                return 0
        return 0
    
    @staticmethod
    def safe_float(value) -> float:
        """Safely convert any value to float"""
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str):
            # Remove commas, dollar signs and convert
            cleaned = value.replace(',', '').replace('$', '').strip()
            try:
                return float(cleaned)
            except (ValueError, TypeError):
                return 0.0
        return 0.0
    
    @staticmethod
    def format_single_result(result: QueryResult) -> str:
        """Format a single result for display"""
        if not result.success:
            return f"❌ ERROR: {result.error}"
        
        data = result.data
        if not data:
            return f"❌ No data returned from {result.tool_used}"
        
        output = []
        
        # Format based on tool type
        if 'subscription' in result.tool_used and 'summary' not in result.tool_used:
            # This is get_subscriptions_in_last_days
            period = data.get('period_days', 'unknown')
            date_range = data.get('date_range', {})
            start_date = date_range.get('start', 'unknown')
            end_date = date_range.get('end', 'unknown')
            
            # Safely convert to integers
            new_subs = ResultFormatter.safe_int(data.get('new_subscriptions', 0))
            active = ResultFormatter.safe_int(data.get('active_subscriptions', 0))
            cancelled = ResultFormatter.safe_int(data.get('cancelled_subscriptions', 0))
            
            output.append(f"📈 SUBSCRIPTION METRICS ({period} days)")
            output.append(f"📅 Period: {start_date} to {end_date}")
            output.append(f"🆕 New Subscriptions: {new_subs:,}")
            output.append(f"✅ Currently Active: {active:,}")
            output.append(f"❌ Cancelled: {cancelled:,}")
            
            # Calculate percentages
            if new_subs > 0:
                retention_rate = (active / new_subs) * 100
                churn_rate = (cancelled / new_subs) * 100
                output.append(f"📊 Retention Rate: {retention_rate:.1f}%")
                output.append(f"📉 Churn Rate: {churn_rate:.1f}%")
        
        elif 'payment' in result.tool_used and 'summary' not in result.tool_used:
            # This is get_payment_success_rate_in_last_days
            period = data.get('period_days', 'unknown')
            date_range = data.get('date_range', {})
            start_date = date_range.get('start', 'unknown')
            end_date = date_range.get('end', 'unknown')
            
            # Safely convert to integers
            total_payments = ResultFormatter.safe_int(data.get('total_payments', 0))
            successful = ResultFormatter.safe_int(data.get('successful_payments', 0))
            failed = ResultFormatter.safe_int(data.get('failed_payments', 0))
            
            output.append(f"💳 PAYMENT METRICS ({period} days)")
            output.append(f"📅 Period: {start_date} to {end_date}")
            output.append(f"📊 Total Payments: {total_payments:,}")
            output.append(f"✅ Successful: {successful:,}")
            output.append(f"❌ Failed: {failed:,}")
            output.append(f"📈 Success Rate: {data.get('success_rate', '0%')}")
            output.append(f"📉 Failure Rate: {data.get('failure_rate', '0%')}")
            
            # Handle revenue safely
            total_revenue = ResultFormatter.safe_float(data.get('total_revenue', '$0.00'))
            lost_revenue = ResultFormatter.safe_float(data.get('lost_revenue', '$0.00'))
            
            output.append(f"💰 Total Revenue: ${total_revenue:,.2f}")
            output.append(f"💸 Lost Revenue: ${lost_revenue:,.2f}")
            
            # Calculate average transaction
            if successful > 0:
                avg_transaction = total_revenue / successful
                output.append(f"📊 Average Transaction: ${avg_transaction:.2f}")
        
        elif 'database' in result.tool_used:
            # Safely convert database metrics
            unique_users = ResultFormatter.safe_int(data.get('unique_users', 0))
            total_subs = ResultFormatter.safe_int(data.get('total_subscriptions', 0))
            total_payments = ResultFormatter.safe_int(data.get('total_payments', 0))
            
            output.append(f"🗄️ DATABASE STATUS")
            output.append(f"📊 Connection: {data.get('status', 'Unknown').upper()}")
            output.append(f"👥 Total Users: {unique_users:,}")
            output.append(f"📝 Total Subscriptions: {total_subs:,}")
            output.append(f"💳 Total Payments: {total_payments:,}")
            output.append(f"📈 Overall Success Rate: {data.get('overall_success_rate', '0%')}")
            
            if data.get('latest_subscription'):
                output.append(f"📅 Latest Subscription: {data['latest_subscription']}")
            if data.get('latest_payment'):
                output.append(f"💰 Latest Payment: {data['latest_payment']}")
        
        elif 'summary' in result.tool_used:
            # This is get_subscription_summary
            period = data.get('period_days', 'unknown')
            output.append(f"📋 COMPREHENSIVE SUMMARY ({period} days)")
            
            if 'summary' in data:
                output.append(f"📝 {data['summary']}")
            
            # Show subscription data if available
            if 'subscriptions' in data:
                subs = data['subscriptions']
                new_subs = ResultFormatter.safe_int(subs.get('new_subscriptions', 0))
                active_subs = ResultFormatter.safe_int(subs.get('active_subscriptions', 0))
                cancelled_subs = ResultFormatter.safe_int(subs.get('cancelled_subscriptions', 0))
                
                output.append(f"\n📈 SUBSCRIPTION DETAILS:")
                output.append(f"   🆕 New Subscriptions: {new_subs:,}")
                output.append(f"   ✅ Active: {active_subs:,}")
                output.append(f"   ❌ Cancelled: {cancelled_subs:,}")
                
                if new_subs > 0:
                    retention_rate = (active_subs / new_subs) * 100
                    output.append(f"   📊 Retention Rate: {retention_rate:.1f}%")
            
            # Show payment data if available
            if 'payments' in data:
                payments = data['payments']
                total_payments = ResultFormatter.safe_int(payments.get('total_payments', 0))
                successful_payments = ResultFormatter.safe_int(payments.get('successful_payments', 0))
                
                output.append(f"\n💳 PAYMENT DETAILS:")
                output.append(f"   📊 Total Payments: {total_payments:,}")
                output.append(f"   ✅ Successful: {successful_payments:,}")
                output.append(f"   📈 Success Rate: {payments.get('success_rate', '0%')}")
                
                revenue = payments.get('total_revenue', '$0.00')
                if isinstance(revenue, str) and revenue.startswith('$'):
                    output.append(f"   💰 Revenue: {revenue}")
                else:
                    revenue_float = ResultFormatter.safe_float(revenue)
                    output.append(f"   💰 Revenue: ${revenue_float:,.2f}")
        
        else:
            # Unknown tool type - show raw data
            output.append(f"📊 RESULTS FROM {result.tool_used.upper()}")
            for key, value in data.items():
                output.append(f"   {key}: {value}")
        
        return "\n".join(output)
    
    @staticmethod
    def format_multi_result(results: List[QueryResult], original_query: str) -> str:
        """Format multiple results for display"""
        output = []
        output.append(f"🎯 RESULTS FOR: '{original_query}'")
        output.append("=" * 80)
        
        for i, result in enumerate(results, 1):
            output.append(f"\n📊 RESULT {i}/{len(results)}")
            output.append("-" * 40)
            
            if result.success:
                formatted = ResultFormatter.format_single_result(result)
                output.append(formatted)
            else:
                output.append(f"❌ ERROR: {result.error}")
        
        # Add comparison summary for multi-results
        if len(results) > 1:
            output.append(f"\n🔍 SIDE-BY-SIDE COMPARISON")
            output.append("=" * 50)
            
            # Create a comparison table
            comparison_data = []
            for i, result in enumerate(results, 1):
                if result.success and result.data:
                    data = result.data
                    days = result.parameters.get('days', 'unknown') if result.parameters else 'unknown'
                    
                    # Extract key metrics based on tool type
                    if 'subscription' in result.tool_used and 'summary' not in result.tool_used:
                        new_subs = ResultFormatter.safe_int(data.get('new_subscriptions', 0))
                        active = ResultFormatter.safe_int(data.get('active_subscriptions', 0))
                        comparison_data.append({
                            'period': f"{days} days",
                            'new_subs': new_subs,
                            'active': active,
                            'type': 'subscription'
                        })
                    elif 'payment' in result.tool_used and 'summary' not in result.tool_used:
                        total_payments = ResultFormatter.safe_int(data.get('total_payments', 0))
                        success_rate = data.get('success_rate', '0%')
                        comparison_data.append({
                            'period': f"{days} days",
                            'total_payments': total_payments,
                            'success_rate': success_rate,
                            'type': 'payment'
                        })
                    elif 'summary' in result.tool_used:
                        # Extract from summary data
                        if 'subscriptions' in data:
                            new_subs = ResultFormatter.safe_int(data['subscriptions'].get('new_subscriptions', 0))
                            active = ResultFormatter.safe_int(data['subscriptions'].get('active_subscriptions', 0))
                        else:
                            new_subs = active = 0
                        
                        if 'payments' in data:
                            total_payments = ResultFormatter.safe_int(data['payments'].get('total_payments', 0))
                            success_rate = data['payments'].get('success_rate', '0%')
                        else:
                            total_payments = 0
                            success_rate = '0%'
                        
                        comparison_data.append({
                            'period': f"{days} days",
                            'new_subs': new_subs,
                            'active': active,
                            'total_payments': total_payments,
                            'success_rate': success_rate,
                            'type': 'summary'
                        })
            
            # Display comparison
            if comparison_data:
                if comparison_data[0]['type'] == 'subscription':
                    output.append("📈 SUBSCRIPTION COMPARISON:")
                    for item in comparison_data:
                        output.append(f"   {item['period']}: {item['new_subs']:,} new, {item['active']:,} active")
                elif comparison_data[0]['type'] == 'payment':
                    output.append("💳 PAYMENT COMPARISON:")
                    for item in comparison_data:
                        output.append(f"   {item['period']}: {item['total_payments']:,} payments, {item['success_rate']} success")
                elif comparison_data[0]['type'] == 'summary':
                    output.append("📋 COMPREHENSIVE COMPARISON:")
                    for item in comparison_data:
                        output.append(f"   {item['period']}:")
                        output.append(f"     📈 Subscriptions: {item['new_subs']:,} new, {item['active']:,} active")
                        output.append(f"     💳 Payments: {item['total_payments']:,} total, {item['success_rate']} success")
        
        return "\n".join(output)

class GeminiNLPProcessor:
    """Uses Gemini API for natural language understanding"""
    
    def __init__(self, api_key: str):
        self.client = genai.Client(api_key=api_key)
        
        # Define available tools for Gemini
        self.available_tools = [
            {
                "name": "get_subscriptions_in_last_days",
                "description": "Get subscription statistics for the last N days",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "days": {
                            "type": "integer",
                            "description": "Number of days to look back (1-365)"
                        }
                    },
                    "required": ["days"]
                }
            },
            {
                "name": "get_payment_success_rate_in_last_days",
                "description": "Get payment success rate statistics for the last N days",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "days": {
                            "type": "integer",
                            "description": "Number of days to look back (1-365)"
                        }
                    },
                    "required": ["days"]
                }
            },
            {
                "name": "get_subscription_summary",
                "description": "Get comprehensive subscription and payment summary",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "days": {
                            "type": "integer",
                            "description": "Number of days to look back (default: 30)"
                        }
                    },
                    "required": []
                }
            },
            {
                "name": "get_database_status",
                "description": "Check database connection and get basic statistics",
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "required": []
                }
            },
            {
                "name": "get_user_payment_history",
                "description": "Get payment history for a specific user",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "merchant_user_id": {
                            "type": "string",
                            "description": "The user ID to query"
                        },
                        "days": {
                            "type": "integer",
                            "description": "Number of days to look back (default: 90)"
                        }
                    },
                    "required": ["merchant_user_id"]
                }
            }
        ]
    
    def parse_query(self, user_query: str) -> List[Dict]:
        """Use Gemini to understand the query and return tool calls"""
        print(f"🤖 Asking Gemini to understand: '{user_query}'")
        
        # Convert tools to Gemini format
        gemini_tools = [
            types.Tool(
                function_declarations=[
                    {
                        "name": tool["name"],
                        "description": tool["description"],
                        "parameters": tool["parameters"]
                    }
                ]
            )
            for tool in self.available_tools
        ]
        
        try:
            # Enhanced prompt for better understanding
            enhanced_query = f"""
            You are an expert in subscription analytics. Analyze this user query: "{user_query}"
            
            IMPORTANT RULES:
            1. For comparison queries with multiple time periods, ALWAYS call get_subscription_summary for each period
            2. If the user wants to "compare" different time periods, use get_subscription_summary for comprehensive data
            3. Extract ALL numbers mentioned in the query (like "10 days", "7 days")
            4. Convert time periods: "1 week" = 7 days, "2 weeks" = 14 days, "1 month" = 30 days
            
            QUERY ANALYSIS:
            - Query: "{user_query}"
            - Contains "compare": {"yes" if "compar" in user_query.lower() else "no"}
            - Time periods mentioned: Look for numbers followed by "day", "week", "month"
            
            TOOL SELECTION LOGIC:
            - For "compare X days vs Y days" or "compare X and Y" → call get_subscription_summary(days=X) AND get_subscription_summary(days=Y)
            - For "subscription performance" or "statistics" → use get_subscription_summary
            - For specific metrics only → use the specific tool
            
            EXAMPLES:
            - "compare 10 days vs 7 days" → get_subscription_summary(days=10) AND get_subscription_summary(days=7)
            - "compare the statistics for 10 days vs 7 days" → get_subscription_summary(days=10) AND get_subscription_summary(days=7)
            - "show me subscription performance for 10 days and 7 days" → get_subscription_summary(days=10) AND get_subscription_summary(days=7)
            
            Please call the appropriate tool(s) for this query. Make sure to use get_subscription_summary for comparison queries.
            """
            
            response = self.client.models.generate_content(
                model="gemini-2.0-flash",
                contents=enhanced_query,
                config=types.GenerateContentConfig(
                    temperature=0,
                    tools=gemini_tools,
                ),
            )
            
            # Extract tool calls from response
            tool_calls = []
            
            if response.candidates and response.candidates[0].content.parts:
                for part in response.candidates[0].content.parts:
                    if hasattr(part, 'function_call') and part.function_call:
                        function_call = part.function_call
                        tool_calls.append({
                            'tool': function_call.name,
                            'parameters': dict(function_call.args),
                            'original_query': user_query
                        })
                        print(f"🔧 Gemini suggests: {function_call.name} with {dict(function_call.args)}")
            
            # If no tool calls, try to extract from text response
            if not tool_calls and response.text:
                print(f"🤖 Gemini response: {response.text}")
                # Fallback to improved parsing
                tool_calls = self._improved_fallback_parse(user_query)
            
            if not tool_calls:
                print("⚠️ Gemini couldn't understand the query, using improved default")
                tool_calls = self._improved_fallback_parse(user_query)
            
            return tool_calls
            
        except Exception as e:
            print(f"❌ Gemini API error: {e}")
            print("🔄 Falling back to improved parsing...")
            return self._improved_fallback_parse(user_query)
    
    def _improved_fallback_parse(self, query: str) -> List[Dict]:
        """Improved fallback parsing"""
        import re
        
        query_lower = query.lower()
        
        # Extract ALL numbers
        numbers = []
        # Look for patterns like "10 days", "7 days", "2 weeks", etc.
        day_matches = re.findall(r'(\d+)\s*days?', query_lower)
        week_matches = re.findall(r'(\d+)\s*weeks?', query_lower)
        month_matches = re.findall(r'(\d+)\s*months?', query_lower)
        
        # Convert to days
        for day in day_matches:
            numbers.append(int(day))
        for week in week_matches:
            numbers.append(int(week) * 7)
        for month in month_matches:
            numbers.append(int(month) * 30)
        
        # Remove duplicates and sort
        numbers = sorted(list(set(numbers)))
        
        print(f"🔍 Extracted time periods: {numbers} days")
        
        # Check for keywords
        has_compare = any(word in query_lower for word in ['compare', 'comparison', 'vs', 'versus', 'against'])
        has_subscription = any(word in query_lower for word in ['subscription', 'subs', 'sub'])
        has_payment = any(word in query_lower for word in ['payment', 'pay', 'rate', 'success'])
        has_summary = any(word in query_lower for word in ['summary', 'overview', 'performance', 'stats', 'statistics'])
        has_database = any(word in query_lower for word in ['database', 'db', 'status'])
        
        results = []
        
        # For comparison queries, use summary tool
        if has_compare or (len(numbers) > 1):
            print("🔍 Detected comparison query - using summary tool")
            if numbers:
                for day_count in numbers:
                    results.append({
                        'tool': 'get_subscription_summary', 
                        'parameters': {'days': day_count}, 
                        'original_query': query
                    })
            else:
                # Default comparison
                results.extend([
                    {'tool': 'get_subscription_summary', 'parameters': {'days': 10}, 'original_query': query},
                    {'tool': 'get_subscription_summary', 'parameters': {'days': 7}, 'original_query': query}
                ])
        else:
            # Single metric queries
            if has_database:
                results.append({'tool': 'get_database_status', 'parameters': {}, 'original_query': query})
            elif has_summary or has_subscription:
                days = numbers[0] if numbers else 30
                results.append({'tool': 'get_subscription_summary', 'parameters': {'days': days}, 'original_query': query})
            elif has_payment:
                days = numbers[0] if numbers else 7
                results.append({'tool': 'get_payment_success_rate_in_last_days', 'parameters': {'days': days}, 'original_query': query})
            else:
                # Default to summary
                days = numbers[0] if numbers else 30
                results.append({'tool': 'get_subscription_summary', 'parameters': {'days': days}, 'original_query': query})
        
        print(f"🔧 Fallback parser suggests: {[r['tool'] + str(r['parameters']) for r in results]}")
        return results

class UniversalClient:
    """Universal Client with SSL fixes and proper async context management"""
    
    def __init__(self, config_path: Optional[str] = None, **kwargs):
        self.config = self._load_config(config_path, **kwargs)
        self.session: Optional[aiohttp.ClientSession] = None
        self.formatter = ResultFormatter()
        
        # Initialize Gemini NLP processor
        gemini_api_key = self.config.get('gemini_api_key')
        if not gemini_api_key:
            raise ValueError("Missing required configuration: gemini_api_key")
        
        self.nlp = GeminiNLPProcessor(gemini_api_key)
        
        # Validate required config
        required_keys = ['server_url', 'api_key', 'gemini_api_key']
        for key in required_keys:
            if key not in self.config:
                raise ValueError(f"Missing required configuration: {key}")
    
    def _load_config(self, config_path: Optional[str] = None, **kwargs) -> Dict:
        """Load configuration from file or environment variables"""
        config = {}
        
        # Override with environment variables
        env_config = {
            'server_url': os.getenv('SUBSCRIPTION_API_URL'),
            'api_key': os.getenv('SUBSCRIPTION_API_KEY') or os.getenv('API_KEY_1'),
            'gemini_api_key': os.getenv('GEMINI_API_KEY'),
            'timeout': int(os.getenv('SUBSCRIPTION_API_TIMEOUT', '30')),
            'retry_attempts': int(os.getenv('SUBSCRIPTION_API_RETRIES', '3'))
        }
        
        for key, value in env_config.items():
            if value is not None:
                config[key] = value
        
        # Override with direct kwargs
        config.update(kwargs)
        
        # Set defaults
        config.setdefault('timeout', 30)
        config.setdefault('retry_attempts', 3)
        config.setdefault('server_url', 'http://localhost:8000')
        
        return config
    
    async def __aenter__(self):
        """Async context manager entry with SSL fix"""
        # Create SSL context with proper certificates
        ssl_context = ssl.create_default_context(cafile=certifi.where())
        connector = aiohttp.TCPConnector(ssl=ssl_context)
        
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=self.config.get('timeout', 30)),
            connector=connector
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.session:
            await self.session.close()
    
    async def call_tool(self, tool_name: str, parameters: Dict = None) -> QueryResult:
        """Execute a tool on the HTTP API server"""
        if not self.session:
            raise RuntimeError("Client not initialized. Use 'async with' context manager.")
        
        api_key = self.config['api_key'].strip()
        headers = {"Authorization": f"Bearer {api_key}"}
        payload = {
            "tool_name": tool_name,
            "parameters": parameters or {}
        }
        
        print(f"🔧 Calling API: {tool_name} with {parameters}")
        print(f"🔗 URL: {self.config['server_url']}/execute")
        
        for attempt in range(self.config.get('retry_attempts', 3)):
            try:
                async with self.session.post(
                    f"{self.config['server_url']}/execute",
                    json=payload,
                    headers=headers
                ) as response:
                    
                    print(f"📡 Response status: {response.status}")
                    data = await response.json()
                    print(f"📊 Response data keys: {list(data.keys()) if isinstance(data, dict) else 'Not a dict'}")
                    
                    if response.status == 200 and data.get('success'):
                        print(f"✅ Tool {tool_name} succeeded")
                        return QueryResult(
                            success=True,
                            data=data['data'],
                            tool_used=tool_name,
                            parameters=parameters
                        )
                    else:
                        error_msg = data.get('error', f"HTTP {response.status}")
                        print(f"❌ Tool failed: {error_msg}")
                        if attempt == self.config.get('retry_attempts', 3) - 1:
                            return QueryResult(
                                success=False,
                                error=f"Tool execution failed: {error_msg}",
                                tool_used=tool_name,
                                parameters=parameters
                            )
                        await asyncio.sleep(2 ** attempt)
                        
            except aiohttp.ClientError as e:
                print(f"🌐 Network error: {e}")
                if attempt == self.config.get('retry_attempts', 3) - 1:
                    return QueryResult(
                        success=False,
                        error=f"Network error: {str(e)}",
                        tool_used=tool_name,
                        parameters=parameters
                    )
                await asyncio.sleep(2 ** attempt)
    
    async def query(self, natural_language_query: str) -> Union[QueryResult, List[QueryResult]]:
        """Process natural language query using Gemini NLP"""
        print(f"\n🎯 Processing: '{natural_language_query}'")
        
        # Use Gemini to understand the query
        parsed_queries = self.nlp.parse_query(natural_language_query)
        
        if len(parsed_queries) == 1:
            # Single query
            query = parsed_queries[0]
            return await self.call_tool(query['tool'], query['parameters'])
        else:
            # Multi-query execution
            print(f"🔄 Executing {len(parsed_queries)} operations...")
            results = []
            for i, query in enumerate(parsed_queries, 1):
                print(f"📊 Operation {i}/{len(parsed_queries)}: {query['tool']} with {query['parameters']}")
                result = await self.call_tool(query['tool'], query['parameters'])
                results.append(result)
            return results
    
    async def query_formatted(self, natural_language_query: str) -> str:
        """Query and return beautifully formatted output"""
        result = await self.query(natural_language_query)
        
        if isinstance(result, list):
            return self.formatter.format_multi_result(result, natural_language_query)
        else:
            return self.formatter.format_single_result(result)

# ==================================================
# COMMAND LINE AND INTERACTIVE MODES
# ==================================================

async def interactive_mode():
    """Interactive mode for users"""
    print("✨ SUBSCRIPTION ANALYTICS - GEMINI POWERED (COMPLETE FIXED VERSION)")
    print("=" * 60)
    
    # Try to load configuration
    try:
        from dotenv import load_dotenv
        load_dotenv()
        
        config = {
            'server_url': os.getenv('SUBSCRIPTION_API_URL', 'http://localhost:8000'),
            'api_key': os.getenv('API_KEY_1') or os.getenv('SUBSCRIPTION_API_KEY'),
            'gemini_api_key': os.getenv('GEMINI_API_KEY')
        }
        
        if not config['api_key']:
            print("⚠️ No API key found in environment variables.")
            config['api_key'] = input("🔑 Enter your API key: ").strip()
        
        if not config['gemini_api_key']:
            print("⚠️ No Gemini API key found in environment variables.")
            config['gemini_api_key'] = input("🤖 Enter your Gemini API key: ").strip()
        
        if not config['api_key'] or not config['gemini_api_key']:
            print("❌ Both API keys are required.")
            return
            
    except ImportError:
        print("📝 Environment configuration not available. Please enter manually:")
        config = {}
        config['server_url'] = input("🔗 Enter API server URL (default: http://localhost:8000): ").strip()
        if not config['server_url']:
            config['server_url'] = 'http://localhost:8000'
        
        config['api_key'] = input("🔑 Enter your API key: ").strip()
        config['gemini_api_key'] = input("🤖 Enter your Gemini API key: ").strip()
        
        if not config['api_key'] or not config['gemini_api_key']:
            print("❌ Both API keys are required.")
            return
    
    print(f"🔗 Connected to: {config['server_url']}")
    print(f"🔑 Using API key: {config['api_key'][:20]}...")
    print(f"🤖 Using Gemini API: {config['gemini_api_key'][:20]}...")
    
    async with UniversalClient(**config) as client:
        print("\n💬 Enter your queries in natural language (or 'quit' to exit):")
        print("\n📋 Example queries:")
        print("  • 'compare the statistics for 10 days vs 7 days'")
        print("  • 'Compare our 1-day and 7-day performance metrics'")
        print("  • 'How is our database doing and what's the 30-day summary?'")
        print("  • 'Give me payment analytics for the past 2 weeks'")
        print("  • 'What are our subscription numbers versus payment rates?'")
        
        while True:
            try:
                query = input("\n🎯 Your query: ").strip()
                
                if query.lower() in ['quit', 'exit', 'q', 'bye']:
                    print("👋 Goodbye!")
                    break
                
                if not query:
                    continue
                
                if query.lower() in ['help', 'h']:
                    print("\n📚 You can ask anything in natural language!")
                    print("  • Time periods: 'last 7 days', 'past 2 weeks', 'this month'")
                    print("  • Metrics: 'subscriptions', 'payments', 'success rates'")
                    print("  • Comparisons: 'compare X and Y', 'X versus Y'")
                    print("  • Multiple queries: 'show me X and also Y'")
                    continue
                
                if query.lower() in ['debug', 'test']:
                    print("🔍 Running debug test...")
                    debug_result = await client.query_formatted("database status")
                    print(debug_result)
                    continue
                
                print("\n🔄 Processing with Gemini AI...")
                formatted_output = await client.query_formatted(query)
                print("\n" + formatted_output)
                    
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")
                logger.exception("Error in interactive mode")

async def single_query_mode(query: str):
    """Single query mode"""
    try:
        from dotenv import load_dotenv
        load_dotenv()
        
        config = {
            'server_url': os.getenv('SUBSCRIPTION_API_URL', 'http://localhost:8000'),
            'api_key': os.getenv('API_KEY_1') or os.getenv('SUBSCRIPTION_API_KEY'),
            'gemini_api_key': os.getenv('GEMINI_API_KEY')
        }
        
        missing_keys = []
        if not config['api_key']:
            missing_keys.append('API_KEY_1 or SUBSCRIPTION_API_KEY')
        if not config['gemini_api_key']:
            missing_keys.append('GEMINI_API_KEY')
        
        if missing_keys:
            print(f"❌ Missing environment variables: {', '.join(missing_keys)}")
            return 1
            
    except ImportError:
        print("❌ python-dotenv not available. Please set environment variables manually.")
        return 1
    
    try:
        async with UniversalClient(**config) as client:
            formatted_output = await client.query_formatted(query)
            print(formatted_output)
            return 0
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1

def main():
    """Main function"""
    if len(sys.argv) > 1:
        if sys.argv[1] in ['--help', '-h', 'help']:
            print("Complete Fixed Universal Client - SSL and async context manager fixes")
            print("\nUsage:")
            print("  python universal_client.py                    # Interactive mode")
            print("  python universal_client.py 'your query'       # Single query mode")
            print("  python universal_client.py --help             # Show help")
            print("\nExample queries:")
            print("  python universal_client.py 'database status'")
            print("  python universal_client.py 'compare 7 vs 30 days'")
            return
        
        # Single query mode
        query = " ".join(sys.argv[1:])
        exit_code = asyncio.run(single_query_mode(query))
        sys.exit(exit_code)
    else:
        # Interactive mode
        asyncio.run(interactive_mode())

if __name__ == "__main__":
    main()