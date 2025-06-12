#!/usr/bin/env python3
"""
Complete Fixed Universal Client with SSL fixes and proper async context management
Enhanced version with better error handling and date range support
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
    """Format results in a beautiful, user-friendly way - FIXED VERSION"""
    
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
        if 'subscription' in result.tool_used and 'summary' not in result.tool_used and 'date_range' not in result.tool_used:
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
        
        elif 'payment' in result.tool_used and 'summary' not in result.tool_used and 'date_range' not in result.tool_used:
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
        """Format multiple results for display - FIXED VERSION"""
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
            
            # Create a comparison table - FIXED TO HANDLE MISSING DATA
            comparison_data = []
            for i, result in enumerate(results, 1):
                if result.success and result.data:
                    data = result.data
                    days = result.parameters.get('days', 'unknown') if result.parameters else 'unknown'
                    
                    # Initialize comparison item with defaults
                    comparison_item = {
                        'period': f"{days} days",
                        'tool_type': result.tool_used,
                        'new_subs': 0,
                        'active': 0,
                        'total_payments': 0,
                        'success_rate': '0%'
                    }
                    
                    # Extract data based on tool type - with safe handling
                    if 'subscription' in result.tool_used and 'summary' not in result.tool_used:
                        # get_subscriptions_in_last_days - only has subscription data
                        comparison_item.update({
                            'new_subs': ResultFormatter.safe_int(data.get('new_subscriptions', 0)),
                            'active': ResultFormatter.safe_int(data.get('active_subscriptions', 0)),
                            'type': 'subscription_only'
                        })
                    elif 'payment' in result.tool_used and 'summary' not in result.tool_used:
                        # get_payment_success_rate_in_last_days - only has payment data
                        comparison_item.update({
                            'total_payments': ResultFormatter.safe_int(data.get('total_payments', 0)),
                            'success_rate': data.get('success_rate', '0%'),
                            'type': 'payment_only'
                        })
                    elif 'summary' in result.tool_used:
                        # get_subscription_summary - has both subscription and payment data
                        if 'subscriptions' in data:
                            comparison_item.update({
                                'new_subs': ResultFormatter.safe_int(data['subscriptions'].get('new_subscriptions', 0)),
                                'active': ResultFormatter.safe_int(data['subscriptions'].get('active_subscriptions', 0))
                            })
                        
                        if 'payments' in data:
                            comparison_item.update({
                                'total_payments': ResultFormatter.safe_int(data['payments'].get('total_payments', 0)),
                                'success_rate': data['payments'].get('success_rate', '0%')
                            })
                        
                        comparison_item['type'] = 'summary'
                    
                    comparison_data.append(comparison_item)
            
            # Display comparison - group by data type available
            subscription_data = [item for item in comparison_data if item.get('new_subs', 0) > 0 or item['type'] in ['subscription_only', 'summary']]
            payment_data = [item for item in comparison_data if item.get('total_payments', 0) > 0 or item['type'] in ['payment_only', 'summary']]
            
            if subscription_data:
                output.append("📈 SUBSCRIPTION COMPARISON:")
                for item in subscription_data:
                    if item['type'] == 'subscription_only':
                        output.append(f"   {item['period']}: {item['new_subs']:,} new, {item['active']:,} active")
                    elif item['type'] == 'summary':
                        output.append(f"   {item['period']}: {item['new_subs']:,} new, {item['active']:,} active")
            
            if payment_data:
                output.append("💳 PAYMENT COMPARISON:")
                for item in payment_data:
                    if item['type'] == 'payment_only':
                        output.append(f"   {item['period']}: {item['total_payments']:,} payments, {item['success_rate']} success")
                    elif item['type'] == 'summary':
                        output.append(f"   {item['period']}: {item['total_payments']:,} payments, {item['success_rate']} success")
            
            # If we have mixed data types, show a combined summary
            if subscription_data and payment_data:
                output.append("📊 COMBINED OVERVIEW:")
                # Create a table showing all metrics side by side
                for item in comparison_data:
                    if item['type'] == 'summary':
                        output.append(f"   {item['period']} (Complete):")
                        output.append(f"     📈 {item['new_subs']:,} new subs, {item['active']:,} active")
                        output.append(f"     💳 {item['total_payments']:,} payments, {item['success_rate']} success")
                    elif item['type'] == 'subscription_only':
                        output.append(f"   {item['period']} (Subscriptions Only):")
                        output.append(f"     📈 {item['new_subs']:,} new subs, {item['active']:,} active")
                    elif item['type'] == 'payment_only':
                        output.append(f"   {item['period']} (Payments Only):")
                        output.append(f"     💳 {item['total_payments']:,} payments, {item['success_rate']} success")
        
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
            },
            {
                "name": "get_subscriptions_by_date_range",
                "description": "Get subscription statistics for a specific date range",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "start_date": {"type": "string", "description": "Start date (YYYY-MM-DD)"},
                        "end_date": {"type": "string", "description": "End date (YYYY-MM-DD)"}
                    },
                    "required": ["start_date", "end_date"]
                }
            },
            {
                "name": "get_payments_by_date_range", 
                "description": "Get payment statistics for a specific date range",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "start_date": {"type": "string", "description": "Start date (YYYY-MM-DD)"},
                        "end_date": {"type": "string", "description": "End date (YYYY-MM-DD)"}
                    },
                    "required": ["start_date", "end_date"]
                }
            },
            {
                "name": "get_analytics_by_date_range",
                "description": "Get comprehensive analytics for a specific date range", 
                "parameters": {
                    "type": "object",
                    "properties": {
                        "start_date": {"type": "string", "description": "Start date (YYYY-MM-DD)"},
                        "end_date": {"type": "string", "description": "End date (YYYY-MM-DD)"}
                    },
                    "required": ["start_date", "end_date"]
                }
            }
        ]
    
    def parse_query(self, user_query: str) -> List[Dict]:
        """Use Gemini to understand the query and return tool calls"""
        print(f"🤖 Asking Gemini to understand: '{user_query}'")
        
        # Get current date for context
        from datetime import datetime
        today = datetime.now().strftime("%Y-%m-%d")
        current_year = datetime.now().year
        
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
            # Enhanced prompt with current date context
            enhanced_query = f"""
            You are an expert in subscription analytics. Today's date is {today}.
            
            Analyze this user query: "{user_query}"
            
            IMPORTANT RULES:
            1. For date range queries, use get_analytics_by_date_range or get_subscriptions_by_date_range
            2. Convert relative dates like "1st June" to "2024-06-01" (assume current year {current_year} if not specified)
            3. "today" means {today}
            4. For comparison queries with multiple time periods, use get_subscription_summary for each period
            5. Extract ALL numbers and dates mentioned
            
            DATE PARSING EXAMPLES:
            - "1st June to today" → start_date: "2024-06-01", end_date: "{today}"
            - "between May 15 and June 30" → start_date: "2024-05-15", end_date: "2024-06-30"
            - "from January 1st to March 31st" → start_date: "2024-01-01", end_date: "2024-03-31"
            
            TOOL SELECTION:
            - For date ranges: use get_analytics_by_date_range(start_date, end_date)
            - For comparisons: use get_subscription_summary for each period
            - For general queries: use appropriate specific tools
            
            Current query: "{user_query}"
            Today's date: {today}
            
            Please call the appropriate tool(s) for this query.
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
        """Improved fallback parsing with date range support"""
        import re
        from datetime import datetime
        
        query_lower = query.lower()
        today = datetime.now().strftime("%Y-%m-%d")
        current_year = datetime.now().year
        
        print(f"🔍 Fallback parsing query: '{query}'")
        print(f"🗓️ Today's date: {today}")
        
        # Check for date range indicators
        has_date_range = any(phrase in query_lower for phrase in [
            'between', 'from', 'to', 'june', 'may', 'april', 'march', 
            'january', 'february', 'july', 'august', 'september', 
            'october', 'november', 'december', 'today', 'yesterday',
            'this month', 'last month', 'this year', 'last year'
        ])
        
        # Month name to number mapping
        months = {
            'january': '01', 'jan': '01',
            'february': '02', 'feb': '02', 
            'march': '03', 'mar': '03',
            'april': '04', 'apr': '04',
            'may': '05',
            'june': '06', 'jun': '06',
            'july': '07', 'jul': '07',
            'august': '08', 'aug': '08',
            'september': '09', 'sep': '09', 'sept': '09',
            'october': '10', 'oct': '10',
            'november': '11', 'nov': '11',
            'december': '12', 'dec': '12'
        }
        
        results = []
        
        # Try to handle date range queries
        if has_date_range:
            print("🗓️ Detected date range query")
            
            # Pattern 1: "between X and Y" or "from X to Y"
            date_patterns = [
                r'between\s+(.+?)\s+and\s+(.+?)(?:\s|$)',
                r'from\s+(.+?)\s+to\s+(.+?)(?:\s|$)',
                r'(.+?)\s+to\s+(.+?)(?:\s|$)',
                r'(.+?)\s+and\s+(.+?)(?:\s|$)'
            ]
            
            for pattern in date_patterns:
                match = re.search(pattern, query_lower)
                if match:
                    start_phrase = match.group(1).strip()
                    end_phrase = match.group(2).strip()
                    print(f"🔍 Found date pattern: '{start_phrase}' to '{end_phrase}'")
                    
                    # Parse start date
                    start_date = self._parse_date_phrase(start_phrase, current_year, months)
                    # Parse end date
                    end_date = self._parse_date_phrase(end_phrase, current_year, months, today)
                    
                    if start_date and end_date:
                        print(f"📅 Parsed dates: {start_date} to {end_date}")
                        return [{
                            'tool': 'get_analytics_by_date_range',
                            'parameters': {
                                'start_date': start_date,
                                'end_date': end_date
                            },
                            'original_query': query
                        }]
            
            # Pattern 2: Single month mentions
            for month_name, month_num in months.items():
                if month_name in query_lower:
                    if 'today' in query_lower or 'now' in query_lower:
                        # "june to today" or "since june"
                        start_date = f"{current_year}-{month_num}-01"
                        end_date = today
                        print(f"📅 Month to today: {start_date} to {end_date}")
                        return [{
                            'tool': 'get_analytics_by_date_range',
                            'parameters': {
                                'start_date': start_date,
                                'end_date': end_date
                            },
                            'original_query': query
                        }]
                    elif 'this month' in query_lower and month_name == datetime.now().strftime('%B').lower():
                        # "this month" and it matches current month
                        start_date = f"{current_year}-{month_num}-01"
                        end_date = today
                        return [{
                            'tool': 'get_analytics_by_date_range',
                            'parameters': {
                                'start_date': start_date,
                                'end_date': end_date
                            },
                            'original_query': query
                        }]
        
        # Extract ALL numbers for "last X days" queries
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

    def _parse_date_phrase(self, phrase: str, current_year: int, months: dict, today: str = None) -> str:
        """Parse a date phrase into YYYY-MM-DD format"""
        import re
        from datetime import datetime, timedelta
        
        phrase = phrase.strip()
        
        # Handle "today"
        if phrase == 'today' or phrase == 'now':
            return today if today else datetime.now().strftime("%Y-%m-%d")
        
        # Handle "1st june", "june 1st", "june 1", etc.
        for month_name, month_num in months.items():
            if month_name in phrase:
                # Look for day number
                day_match = re.search(r'(\d+)', phrase)
                if day_match:
                    day = int(day_match.group(1))
                    if 1 <= day <= 31:
                        return f"{current_year}-{month_num}-{day:02d}"
                else:
                    # Just month name, assume 1st
                    return f"{current_year}-{month_num}-01"
        
        # Handle YYYY-MM-DD format
        if re.match(r'\d{4}-\d{2}-\d{2}', phrase):
            return phrase
        
        # Handle MM/DD/YYYY or DD/MM/YYYY
        date_match = re.match(r'(\d{1,2})[/-](\d{1,2})[/-](\d{4})', phrase)
        if date_match:
            # Assume MM/DD/YYYY format (US style)
            month, day, year = date_match.groups()
            return f"{year}-{int(month):02d}-{int(day):02d}"
        
        # Handle relative terms
        if 'yesterday' in phrase:
            yesterday = datetime.now() - timedelta(days=1)
            return yesterday.strftime("%Y-%m-%d")
        
        # If we can't parse it, return None
        print(f"⚠️ Could not parse date phrase: '{phrase}'")
        return None

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
        """Load configuration from file or environment variables with enhanced fallbacks"""
        config = {}
        
        # Try to load from .env file
        try:
            from dotenv import load_dotenv
            
            # Try multiple .env locations
            env_paths = [
                '.env',
                '../.env',
                os.path.join(os.getcwd(), '.env'),
                os.path.join(os.path.dirname(__file__), '.env') if '__file__' in globals() else None
            ]
            
            env_loaded = False
            for env_path in env_paths:
                if env_path and os.path.exists(env_path):
                    load_dotenv(env_path)
                    env_loaded = True
                    logger.debug(f"Loaded .env from: {env_path}")
                    break
            
            if not env_loaded:
                logger.warning("No .env file found, using environment variables only")
                
        except ImportError:
            logger.warning("python-dotenv not available, using environment variables only")
        
        # Override with environment variables
        env_config = {
            'server_url': os.getenv('SUBSCRIPTION_API_URL'),
            'api_key': os.getenv('SUBSCRIPTION_API_KEY') or os.getenv('API_KEY_1'),
            'gemini_api_key': os.getenv('GEMINI_API_KEY'),
            'timeout': int(os.getenv('SUBSCRIPTION_API_TIMEOUT', '30')),
            'retry_attempts': int(os.getenv('SUBSCRIPTION_API_RETRIES', '3')),
            'debug': os.getenv('DEBUG', 'False').lower() in ('true', '1', 't')
        }
        
        for key, value in env_config.items():
            if value is not None:
                config[key] = value
        
        # Override with direct kwargs
        config.update(kwargs)
        
        # Set defaults
        config.setdefault('timeout', 30)
        config.setdefault('retry_attempts', 3)
        config.setdefault('server_url', 'https://subscription-analysis-production.up.railway.app')
        
        return config
    
    async def __aenter__(self):
        """Async context manager entry with enhanced SSL handling"""
        import ssl
        
        # Create SSL context with debug option from config
        ssl_context = ssl.create_default_context()
        
        # Check if we should disable SSL verification (for development only)
        if self.config.get('debug', False):
            logger.warning("⚠️ Running in debug mode with SSL verification disabled")
            ssl_context.check_hostname = False
            ssl_context.verify_mode = ssl.CERT_NONE
        else:
            try:
                # Try to use system CA certificates
                ssl_context = ssl.create_default_context(cafile=certifi.where())
                logger.info("✓ Using system CA certificates for SSL verification")
            except Exception as e:
                logger.warning(f"⚠️ Could not load system CA certificates: {e}")
                logger.warning("⚠️ Falling back to default SSL context (less secure)")
        
        # Configure connection pooling
        connector = aiohttp.TCPConnector(
            ssl=ssl_context,
            limit=10,  # Max number of simultaneous connections
            keepalive_timeout=30  # Keep connections alive for 30 seconds
        )
        
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
        """Execute a tool on the HTTP API server with enhanced error handling"""
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
                    
                    # Handle different response types
                    content_type = response.headers.get('content-type', '').lower()
                    if 'application/json' in content_type:
                        data = await response.json()
                        print(f"📊 Response data keys: {list(data.keys()) if isinstance(data, dict) else 'Not a dict'}")
                    else:
                        # Handle non-JSON responses
                        text_data = await response.text()
                        print(f"📊 Response (text): {text_data[:200]}...")
                        data = {"error": f"Non-JSON response: {text_data[:200]}"}
                    
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
                        
                        # For final attempt, return the error
                        if attempt == self.config.get('retry_attempts', 3) - 1:
                            return QueryResult(
                                success=False,
                                error=f"Tool execution failed: {error_msg}",
                                tool_used=tool_name,
                                parameters=parameters
                            )
                        
                        # Wait before retry
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
            except Exception as e:
                print(f"🚨 Unexpected error: {e}")
                if attempt == self.config.get('retry_attempts', 3) - 1:
                    return QueryResult(
                        success=False,
                        error=f"Unexpected error: {str(e)}",
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
    """Interactive mode for users with enhanced error handling"""
    print("✨ SUBSCRIPTION ANALYTICS - GEMINI POWERED (ENHANCED VERSION)")
    print("=" * 60)
    
    # Try to load configuration
    config = {}
    try:
        from dotenv import load_dotenv
        load_dotenv()
        
        config = {
            'server_url': os.getenv('SUBSCRIPTION_API_URL', 'https://subscription-analysis-production.up.railway.app'),
            'api_key': os.getenv('API_KEY_1') or os.getenv('SUBSCRIPTION_API_KEY'),
            'gemini_api_key': os.getenv('GEMINI_API_KEY')
        }
        
        # Check for missing keys
        missing_keys = []
        if not config['api_key']:
            missing_keys.append('API_KEY_1 or SUBSCRIPTION_API_KEY')
        if not config['gemini_api_key']:
            missing_keys.append('GEMINI_API_KEY')
        
        if missing_keys:
            print(f"⚠️ Missing environment variables: {', '.join(missing_keys)}")
            
            # Interactive fallback
            if not config['api_key']:
                config['api_key'] = input("🔑 Enter your API key: ").strip()
            if not config['gemini_api_key']:
                config['gemini_api_key'] = input("🤖 Enter your Gemini API key: ").strip()
        
        if not config['api_key'] or not config['gemini_api_key']:
            print("❌ Both API keys are required.")
            return
            
    except ImportError:
        print("📝 Environment configuration not available. Please enter manually:")
        config['server_url'] = input("🔗 Enter API server URL (default: https://subscription-analysis-production.up.railway.app): ").strip()
        if not config['server_url']:
            config['server_url'] = 'https://subscription-analysis-production.up.railway.app'
        
        config['api_key'] = input("🔑 Enter your API key: ").strip()
        config['gemini_api_key'] = input("🤖 Enter your Gemini API key: ").strip()
        
        if not config['api_key'] or not config['gemini_api_key']:
            print("❌ Both API keys are required.")
            return
    
    print(f"🔗 Connected to: {config['server_url']}")
    print(f"🔑 Using API key: {config['api_key'][:20]}...")
    print(f"🤖 Using Gemini API: {config['gemini_api_key'][:20]}...")
    
    try:
        async with UniversalClient(**config) as client:
            print("\n💬 Enter your queries in natural language (or 'quit' to exit):")
            print("📚 Examples:")
            print("  • 'database status'")
            print("  • 'subscription performance for last 7 days'")
            print("  • 'compare 7 days vs 30 days'")
            print("  • 'analytics from June 1st to today'")
            
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
                        print("  • Date ranges: 'from June 1st to today', 'between May 15 and June 30'")
                        print("  • Database: 'database status', 'db health'")
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
                    print(f"❌ Error processing query: {e}")
                    logger.exception("Error in interactive mode")
    except Exception as e:
        print(f"❌ Failed to initialize client: {e}")
        logger.exception("Client initialization error")

async def single_query_mode(query: str):
    """Single query mode with enhanced error handling"""
    try:
        from dotenv import load_dotenv
        load_dotenv()
        
        config = {
            'server_url': os.getenv('SUBSCRIPTION_API_URL', 'https://subscription-analysis-production.up.railway.app'),
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
            print("Please set the following environment variables:")
            for key in missing_keys:
                print(f"  - {key}")
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
        logger.exception("Single query mode error")
        return 1

def main():
    """Main function with enhanced help and error handling"""
    if len(sys.argv) > 1:
        if sys.argv[1] in ['--help', '-h', 'help']:
            print("Enhanced Universal Client - Subscription Analytics")
            print("\nUsage:")
            print("  python universal_client.py                    # Interactive mode")
            print("  python universal_client.py 'your query'       # Single query mode")
            print("  python universal_client.py --help             # Show help")
            print("\nExample queries:")
            print("  python universal_client.py 'database status'")
            print("  python universal_client.py 'compare 7 vs 30 days'")
            print("  python universal_client.py 'analytics from June 1st to today'")
            print("\nEnvironment variables needed:")
            print("  - API_KEY_1 or SUBSCRIPTION_API_KEY")
            print("  - GEMINI_API_KEY")
            print("  - SUBSCRIPTION_API_URL (optional, defaults to production)")
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