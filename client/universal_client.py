# client/universal_client.py

import asyncio
import aiohttp
import os
import json
import sys
import ssl
import certifi
import logging
import re
from typing import Dict, List, Optional, Union, Any
from dataclasses import dataclass
import google.generativeai as genai

# Use the ConfigManager from the same package
from .config_manager import ConfigManager

# Basic Setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Data Structures
@dataclass
class QueryResult:
    success: bool
    data: Optional[Any] = None
    error: Optional[str] = None
    tool_used: Optional[str] = None
    parameters: Optional[Dict] = None
    is_dynamic: bool = False
    original_query: Optional[str] = None
    generated_sql: Optional[str] = None
    message: Optional[str] = None

# Formatting Logic
class ResultFormatter:
    @staticmethod
    def format_single_result(result: QueryResult) -> str:
        if not result.success:
            return f"❌ ERROR: {result.error}"
        
        if result.message:
            return f"ℹ️ {result.message}"
        
        if result.data is None:
            return "✅ Query succeeded, but the result set was empty."
        
        output = []
        is_dynamic = result.tool_used == 'execute_dynamic_sql'
        header = f"📊 DYNAMIC QUERY RESULT" if is_dynamic else f"📊 RESULT FROM TOOL: {result.tool_used.upper()}"
        output.append(header)
        output.append("=" * len(header))
        
        if isinstance(result.data, list) and len(result.data) > 0:
            # Check if this looks like a success rate analysis
            headers = list(result.data[0].keys())
            has_success_data = any('success' in str(h).lower() for h in headers) and any('total' in str(h).lower() for h in headers)
            
            # Table formatting
            col_widths = {h: len(str(h)) for h in headers}
            
            # Calculate column widths
            for row in result.data:
                for h in headers:
                    col_widths[h] = max(col_widths[h], len(str(row.get(h, ''))))
            
            # Create table
            header_line = " | ".join(h.ljust(col_widths[h]) for h in headers)
            output.append(header_line)
            output.append("-" * len(header_line))
            
            for row in result.data:
                output.append(" | ".join(str(row.get(h, '')).ljust(col_widths[h]) for h in headers))
            
            output.append("")  # Empty line
            output.append(f"📈 Total rows: {len(result.data)}")
            
            # Add analysis for success rate data
            if has_success_data and len(result.data) > 1:
                output.append("\n💡 **Quick Analysis:**")
                try:
                    for i, row in enumerate(result.data[:3]):  # Top 3
                        merchant_id = row.get('merchant_user_id', 'Unknown')
                        total = row.get('total_payments', 0)
                        successful = row.get('successful_payments', 0)
                        
                        if total and total > 0:
                            rate = (successful / total) * 100
                            volume_desc = "High" if total >= 20 else "Medium" if total >= 5 else "Low"
                            output.append(f"   {i+1}. Merchant {merchant_id}: {rate:.1f}% success rate ({successful}/{total}) - {volume_desc} volume")
                except:
                    pass  # Skip analysis if data structure is unexpected
            
            if result.generated_sql:
                output.append(f"\n🔍 Generated SQL:")
                output.append("-" * 20)
                output.append(result.generated_sql)
        elif isinstance(result.data, dict):
            # Dictionary formatting
            for key, value in result.data.items():
                output.append(f"{key}: {value}")
        else:
            output.append(json.dumps(result.data, indent=2, default=str))
        
        return "\n".join(output)

    @staticmethod
    def format_multi_result(results: List[QueryResult], original_query: str) -> str:
        output = [f"🎯 RESULTS FOR COMPARISON: '{original_query}'", "="*70]
        for i, res in enumerate(results, 1):
            output.append(f"\n--- Result {i}/{len(results)} ---")
            output.append(ResultFormatter.format_single_result(res))
        return "\n".join(output)

# Enhanced AI Logic
class GeminiNLPProcessor:
    def __init__(self):
        self.model = genai.GenerativeModel('gemini-1.5-flash-latest')
        self.db_schema = """
Database Schema:
Tables:
1. subscription_contract_v2:
   - subscription_id (VARCHAR, PRIMARY KEY)
   - merchant_user_id (VARCHAR)
   - status (ENUM: 'ACTIVE', 'INACTIVE')
   - subcription_start_date (DATE)

2. subscription_payment_details:
   - subcription_payment_details_id (VARCHAR, PRIMARY KEY)
   - subscription_id (VARCHAR, FOREIGN KEY)
   - status (ENUM: 'ACTIVE', 'FAILED', 'FAIL', 'INIT')
   - trans_amount_decimal (DECIMAL)
   - created_date (DATE)

Important Notes:
- To get payment success rates by individual users, JOIN the tables on subscription_id
- Use 'ACTIVE' status in subscription_payment_details for successful payments
- Use proper table aliases: subscription_contract_v2 AS sc, subscription_payment_details AS pd
- Always use WHERE clauses for date filtering when analyzing recent data
"""
        
        # Define available tools for the AI
        self.tools = [
            genai.protos.Tool(
                function_declarations=[
                    genai.protos.FunctionDeclaration(
                        name="get_subscriptions_in_last_days",
                        description="Get subscription statistics for the last N days. Use this for general subscription overview questions.",
                        parameters=genai.protos.Schema(
                            type=genai.protos.Type.OBJECT,
                            properties={
                                "days": genai.protos.Schema(
                                    type=genai.protos.Type.INTEGER, 
                                    description="Number of days to look back (1-365)"
                                )
                            },
                            required=["days"]
                        )
                    ),
                    genai.protos.FunctionDeclaration(
                        name="get_payment_success_rate_in_last_days",
                        description="Get payment success rate and revenue statistics for the last N days. Use this for payment performance questions.",
                        parameters=genai.protos.Schema(
                            type=genai.protos.Type.OBJECT,
                            properties={
                                "days": genai.protos.Schema(
                                    type=genai.protos.Type.INTEGER, 
                                    description="Number of days to look back (1-365)"
                                )
                            },
                            required=["days"]
                        )
                    ),
                    genai.protos.FunctionDeclaration(
                        name="get_user_payment_history",
                        description="Get payment history for a specific user by merchant_user_id. Use this when the user asks about a specific user.",
                        parameters=genai.protos.Schema(
                            type=genai.protos.Type.OBJECT,
                            properties={
                                "merchant_user_id": genai.protos.Schema(
                                    type=genai.protos.Type.STRING, 
                                    description="The merchant user ID to query"
                                ),
                                "days": genai.protos.Schema(
                                    type=genai.protos.Type.INTEGER, 
                                    description="Number of days to look back (default: 90)"
                                )
                            },
                            required=["merchant_user_id"]
                        )
                    ),
                    genai.protos.FunctionDeclaration(
                        name="get_database_status",
                        description="Check database connection and get basic statistics. Use this for health checks or general database info.",
                        parameters=genai.protos.Schema(
                            type=genai.protos.Type.OBJECT,
                            properties={}
                        )
                    ),
                    genai.protos.FunctionDeclaration(
                        name="execute_dynamic_sql",
                        description="Generate and execute a custom SQL SELECT query for complex analytics that can't be answered with pre-built tools. Use this for specific analytical questions, rankings, aggregations, or complex filtering.",
                        parameters=genai.protos.Schema(
                            type=genai.protos.Type.OBJECT,
                            properties={
                                "sql_query": genai.protos.Schema(
                                    type=genai.protos.Type.STRING, 
                                    description="The SELECT SQL query to execute. Must start with SELECT. Use proper JOINs between tables when needed."
                                )
                            },
                            required=["sql_query"]
                        )
                    )
                ]
            )
        ]

    def parse_query(self, user_query: str, history: List[str]) -> List[Dict]:
        """Parse user query and determine which tool(s) to use with enhanced context awareness"""
        
        # Build context from conversation history
        history_context = "\n".join(history[-6:]) if history else "No previous context."
        
        # Enhanced context analysis for follow-up questions
        follow_up_indicators = [
            "than that", "greater than that", "higher than that", "lower than that", "above that", "below that",
            "compared to that", "versus that", "against that", "from that result", "from those results",
            "among them", "of those", "from these", "from the previous", "from that list", 
            "out of these", "which one", "who has the", "how many have", "how many people", "how many users",
            "the worst", "the best", "bottom", "top one", "that rate", "that percentage",
            "individual", "personal", "each user", "each merchant", "per user", "per merchant",
            "breakdown", "detailed", "specific users", "more than that", "less than that", 
            "same time", "same period", "same timeframe"
        ]
        
        # Check if this is a contextual follow-up
        is_contextual = any(indicator in user_query.lower() for indicator in follow_up_indicators)
        
        # Enhanced context extraction from history
        context_data = self._extract_context_from_history(history)
        
        prompt = f"""
You are a subscription analytics assistant. Analyze the user's query and choose the most appropriate tool.

CONVERSATION HISTORY:
{history_context}

EXTRACTED CONTEXT DATA:
{context_data}

CURRENT USER QUERY: "{user_query}"

CRITICAL INSTRUCTIONS FOR SQL GENERATION:

1. **ALWAYS START WITH SELECT**: Every SQL query must begin with "SELECT"
2. **USE PROPER TABLE ALIASES**: Always use aliases like "sc" for subscription_contract_v2 and "pd" for subscription_payment_details
3. **QUALIFY ALL COLUMNS**: Always specify table alias (e.g., sc.merchant_user_id, pd.status)
4. **JOIN TABLES CORRECTLY**: Use proper JOIN syntax when needed
5. **USE MYSQL SYNTAX**: Generate MySQL-compatible queries, not SQLite:
   - For date arithmetic: DATE_SUB(CURDATE(), INTERVAL 30 DAY) not DATE('now', '-30 days')
   - For current date: CURDATE() not DATE('now')
   - For current datetime: NOW() not DATETIME('now')

6. **HANDLE SUCCESS RATES PROPERLY**: For individual user success rates, use this pattern:
   ```sql
   SELECT sc.merchant_user_id, 
          COUNT(*) as total_payments,
          SUM(CASE WHEN pd.status = 'ACTIVE' THEN 1 ELSE 0 END) as successful_payments,
          ROUND((SUM(CASE WHEN pd.status = 'ACTIVE' THEN 1 ELSE 0 END) * 100.0 / COUNT(*)), 2) as success_rate_percent
   FROM subscription_contract_v2 AS sc 
   JOIN subscription_payment_details AS pd ON sc.subscription_id = pd.subscription_id 
   GROUP BY sc.merchant_user_id
   HAVING COUNT(*) >= 3
   ORDER BY success_rate_percent DESC
   ```

7. **SUBQUERY COLUMN REFERENCES**: When using subqueries, reference columns correctly:
   - WRONG: `SELECT sc.merchant_user_id FROM (SELECT sc.merchant_user_id ...) AS sub WHERE ...`
   - CORRECT: `SELECT merchant_user_id FROM (SELECT sc.merchant_user_id ...) AS sub WHERE ...`
   - In outer query, use column names as they appear in subquery SELECT, not with inner aliases

8. **AVOID COMPLEX SUBQUERIES**: When possible, use single queries with HAVING clauses instead of subqueries:
   - COMPLEX: `SELECT ... FROM (SELECT ... FROM ... GROUP BY ...) AS sub WHERE ...`
   - SIMPLE: `SELECT ... FROM ... GROUP BY ... HAVING ... AND ...`

9. **SUBQUERY COLUMN SELECTION**: If you must use subqueries, ensure all needed columns are selected:
   - If outer query needs `pd.status`, the subquery must SELECT it
   - If outer query needs aggregate calculations, do them in the subquery, not outside

10. **CONTEXT AWARENESS**: When user refers to "that", "those", etc., extract values from conversation history:
   - If previous result showed success_rate_percent: 14.86, and user asks "how many users have higher than that"
   - Generate SQL with WHERE success_rate_percent > 14.86

{self.db_schema}

TOOL SELECTION LOGIC:

1. **For simple, standalone questions:**
   - "subscriptions in last X days" → get_subscriptions_in_last_days
   - "payment success rate" → get_payment_success_rate_in_last_days  
   - "payment history for user X" → get_user_payment_history
   - "database status" → get_database_status

2. **For complex analytics or context-dependent questions:**
   - Individual comparisons, breakdowns, detailed analysis → execute_dynamic_sql
   - Questions referring to previous results → execute_dynamic_sql with context values
   - "How many users/merchants..." → execute_dynamic_sql with COUNT
   - Any question requiring granular data → execute_dynamic_sql

3. **SPECIFIC EXAMPLES FOR CONTEXT-AWARE QUERIES:**
   
   Example 1: Previous result showed "success_rate_percent: 14.86", user asks "how many users have higher than that"
   Generate SQL:
   ```sql
   SELECT COUNT(*) as users_above_average
   FROM (
       SELECT sc.merchant_user_id, 
              ROUND((SUM(CASE WHEN pd.status = 'ACTIVE' THEN 1 ELSE 0 END) * 100.0 / COUNT(*)), 2) as success_rate_percent
       FROM subscription_contract_v2 AS sc 
       JOIN subscription_payment_details AS pd ON sc.subscription_id = pd.subscription_id 
       GROUP BY sc.merchant_user_id
       HAVING COUNT(*) >= 3
   ) AS user_rates WHERE success_rate_percent > 14.86
   ```

   Example 2: User asks "show me individual user success rates"
   Generate SQL:
   ```sql
   SELECT sc.merchant_user_id, 
          COUNT(*) as total_payments,
          SUM(CASE WHEN pd.status = 'ACTIVE' THEN 1 ELSE 0 END) as successful_payments,
          ROUND((SUM(CASE WHEN pd.status = 'ACTIVE' THEN 1 ELSE 0 END) * 100.0 / COUNT(*)), 2) as success_rate_percent
   FROM subscription_contract_v2 AS sc 
   JOIN subscription_payment_details AS pd ON sc.subscription_id = pd.subscription_id 
   GROUP BY sc.merchant_user_id
   HAVING COUNT(*) >= 3
   ORDER BY success_rate_percent DESC
   LIMIT 20
   ```

   Example 3: User asks "show me their ids" (referring to users above threshold)
   Generate SQL:
   ```sql
   SELECT merchant_user_id, success_rate_percent
   FROM (
       SELECT sc.merchant_user_id, 
              ROUND((SUM(CASE WHEN pd.status = 'ACTIVE' THEN 1 ELSE 0 END) * 100.0 / COUNT(*)), 2) as success_rate_percent
       FROM subscription_contract_v2 AS sc 
       JOIN subscription_payment_details AS pd ON sc.subscription_id = pd.subscription_id 
       GROUP BY sc.merchant_user_id
       HAVING COUNT(*) >= 3
   ) AS user_rates 
   WHERE success_rate_percent > 14.86
   ORDER BY success_rate_percent DESC
   ```

   Example 4: User asks "what's their failure rate?" (referring to users above threshold)
   Generate SQL (single query approach - PREFERRED):
   ```sql
   SELECT sc.merchant_user_id,
          COUNT(*) as total_payments,
          ROUND((SUM(CASE WHEN pd.status = 'ACTIVE' THEN 1 ELSE 0 END) * 100.0 / COUNT(*)), 2) as success_rate_percent,
          ROUND((SUM(CASE WHEN pd.status != 'ACTIVE' THEN 1 ELSE 0 END) * 100.0 / COUNT(*)), 2) as failure_rate_percent
   FROM subscription_contract_v2 AS sc 
   JOIN subscription_payment_details AS pd ON sc.subscription_id = pd.subscription_id 
   GROUP BY sc.merchant_user_id
   HAVING COUNT(*) >= 3 AND ROUND((SUM(CASE WHEN pd.status = 'ACTIVE' THEN 1 ELSE 0 END) * 100.0 / COUNT(*)), 2) > 14.86
   ORDER BY success_rate_percent DESC
   ```

   Example 5: Previous query was "subscriptions in last 30 days", user asks "payments in same time"
   Generate SQL:
   ```sql
   SELECT COUNT(*) AS total_payments 
   FROM subscription_payment_details 
   WHERE created_date >= DATE_SUB(CURDATE(), INTERVAL 30 DAY)
   ```

Choose the most appropriate tool and provide necessary parameters. Make sure all SQL queries are valid MySQL and start with SELECT.
"""

        try:
            response = self.model.generate_content(
                prompt,
                tools=self.tools,
                tool_config={'function_calling_config': {'mode': 'ANY'}}
            )
            
            tool_calls = []
            if response.candidates and response.candidates[0].content.parts:
                for part in response.candidates[0].content.parts:
                    if hasattr(part, 'function_call') and part.function_call:
                        fc = part.function_call
                        
                        # Validate SQL if it's a dynamic query
                        params = dict(fc.args)
                        if fc.name == 'execute_dynamic_sql' and 'sql_query' in params:
                            sql_query = params['sql_query'].strip()
                            
                            # Remove any surrounding quotes that might have been added
                            if sql_query.startswith('"') and sql_query.endswith('"'):
                                sql_query = sql_query[1:-1]
                            if sql_query.startswith("'") and sql_query.endswith("'"):
                                sql_query = sql_query[1:-1]
                            
                            # Clean the query
                            sql_query = sql_query.strip()
                            
                            # Check if it actually starts with SELECT (case insensitive)
                            if not sql_query.upper().startswith('SELECT'):
                                logger.warning(f"Generated SQL doesn't start with SELECT: {sql_query}")
                                # Try to fix simple cases
                                if 'SELECT' in sql_query.upper():
                                    select_pos = sql_query.upper().find('SELECT')
                                    sql_query = sql_query[select_pos:]
                                    logger.info(f"Fixed SQL query: {sql_query}")
                                else:
                                    logger.error(f"Cannot fix SQL query: {sql_query}")
                                    continue
                            
                            # Update the cleaned query
                            params['sql_query'] = sql_query
                        
                        tool_calls.append({
                            'tool': fc.name,
                            'parameters': params,
                            'original_query': user_query
                        })
            
            if not tool_calls:
                logger.warning("AI did not return a valid function call, defaulting to database status")
                return [{
                    'tool': 'get_database_status',
                    'parameters': {},
                    'original_query': user_query
                }]
            
            logger.info(f"AI selected tool(s): {[tc['tool'] for tc in tool_calls]}")
            return tool_calls
            
        except Exception as e:
            logger.error(f"Error in parse_query: {e}", exc_info=True)
            # Fallback to database status
            return [{
                'tool': 'get_database_status',
                'parameters': {},
                'original_query': user_query
            }]

    def _extract_context_from_history(self, history: List[str]) -> str:
        """Extract relevant numerical and contextual data from conversation history."""
        if not history:
            return "No previous context available."
        
        context_data = []
        recent_history = "\n".join(history[-4:])  # Last 2 exchanges
        
        # Extract percentages
        import re
        percentages = re.findall(r'success_rate_percent:\s*(\d+\.?\d*)|(\d+\.?\d*)\s*%', recent_history)
        if percentages:
            # Flatten and filter out empty matches
            flat_percentages = [p for pair in percentages for p in pair if p]
            if flat_percentages:
                context_data.append(f"Recent success rate: {flat_percentages[-1]}%")
        
        # Extract payment counts
        payment_counts = re.findall(r'total_payments:\s*(\d+)|(\d+)\s*total payments', recent_history)
        if payment_counts:
            flat_counts = [p for pair in payment_counts for p in pair if p]
            if flat_counts:
                context_data.append(f"Recent total payments: {flat_counts[-1]}")
        
        # Extract revenue amounts
        revenue = re.findall(r'total_revenue:\s*([\d,]+\.?\d*)|revenue.*?(\d+)', recent_history)
        if revenue:
            flat_revenue = [p for pair in revenue for p in pair if p]
            if flat_revenue:
                context_data.append(f"Recent revenue: {flat_revenue[-1]}")
        
        # Extract days/time periods for context
        days_mentioned = re.findall(r'(\d+)\s*days?', recent_history)
        if days_mentioned:
            context_data.append(f"Recent time period: {days_mentioned[-1]} days")
        
        # Check for merchant lists or user discussions
        if 'merchant_user_id' in recent_history:
            context_data.append("Previous query involved individual merchant analysis")
        
        if 'success_rate_percent' in recent_history:
            context_data.append("Previous query involved success rate analysis")
        
        if 'subscription' in recent_history.lower():
            context_data.append("Previous query involved subscription data")
            
        if 'payment' in recent_history.lower():
            context_data.append("Previous query involved payment data")
        
        return "\n".join(context_data) if context_data else "No specific numerical context found."

# Core Client Class
class UniversalClient:
    def __init__(self, config: dict):
        self.config = config
        self.nlp = GeminiNLPProcessor()
        self.session = None
        self.formatter = ResultFormatter()
        self.history = []

    async def __aenter__(self):
        # FIXED: Better connection handling to prevent connection resets
        connector = aiohttp.TCPConnector(
            ssl=ssl.create_default_context(cafile=certifi.where()),
            limit=10,           # Total connection limit
            limit_per_host=5,   # Per-host connection limit
            enable_cleanup_closed=True,  # Clean up closed connections
            force_close=True,   # Force close connections after each request
            ttl_dns_cache=300   # DNS cache TTL
        )
        self.session = aiohttp.ClientSession(
            connector=connector,
            timeout=aiohttp.ClientTimeout(total=120),  # Increased timeout
            headers={'Connection': 'close'}  # Force connection close
        )
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()

    def _fix_sql_syntax(self, sql_query: str) -> str:
        """Automatically fix common SQL syntax errors for MySQL"""
        original_sql = sql_query
        
        # Fix SQLite date functions to MySQL equivalents
        sql_fixes = [
            # SQLite DATE('now', '-N days') → MySQL DATE_SUB(CURDATE(), INTERVAL N DAY)
            (r"DATE\('now',\s*'-(\d+)\s+days?'\)", r"DATE_SUB(CURDATE(), INTERVAL \1 DAY)"),
            (r"DATE\('now',\s*'-(\d+)\s+day'\)", r"DATE_SUB(CURDATE(), INTERVAL \1 DAY)"),
            
            # SQLite DATETIME('now') → MySQL NOW()
            (r"DATETIME\('now'\)", "NOW()"),
            (r"DATE\('now'\)", "CURDATE()"),
            
            # SQLite strftime → MySQL DATE_FORMAT
            (r"strftime\('%Y-%m-%d',\s*([^)]+)\)", r"DATE_FORMAT(\1, '%Y-%m-%d')"),
            
            # SQLite julianday differences → MySQL DATEDIFF
            (r"julianday\(([^)]+)\)\s*-\s*julianday\(([^)]+)\)", r"DATEDIFF(\1, \2)"),
            
            # SQLite || for concatenation → MySQL CONCAT()
            (r"(['\"][^'\"]*['\"])\s*\|\|\s*(['\"][^'\"]*['\"])", r"CONCAT(\1, \2)"),
            
            # Fix common quote issues
            (r"'(\d{4}-\d{2}-\d{2})'", r"'\1'"),  # Ensure proper date format
        ]
        
        for pattern, replacement in sql_fixes:
            sql_query = re.sub(pattern, replacement, sql_query, flags=re.IGNORECASE)
        
        # If SQL was changed, log what was fixed
        if sql_query != original_sql:
            logger.info(f"🔧 SQL auto-fix applied:")
            logger.info(f"   Before: {original_sql}")
            logger.info(f"   After:  {sql_query}")
        
        return sql_query

    async def call_tool(self, tool_name: str, parameters: Dict = None, original_query: str = "") -> QueryResult:
        """Call a tool on the API server with enhanced error handling and auto-fix retry logic"""
        headers = {
            "Authorization": f"Bearer {self.config['API_KEY_1']}",
            "Connection": "close"  # Force connection close
        }
        payload = {"tool_name": tool_name, "parameters": parameters or {}}
        server_url = self.config['SUBSCRIPTION_API_URL']
        
        # Log the request for debugging
        if tool_name == 'execute_dynamic_sql':
            logger.info(f"🔍 Executing SQL: {parameters.get('sql_query', 'N/A')}")
        
        # Retry logic for connection issues and SQL syntax errors
        max_retries = 3
        for attempt in range(max_retries):
            try:
                async with self.session.post(f"{server_url}/execute", json=payload, headers=headers) as response:
                    if response.status == 401:
                        return QueryResult(
                            success=False,
                            error="Authentication failed. Please check your API key.",
                            tool_used=tool_name
                        )
                    elif response.status == 404:
                        return QueryResult(
                            success=False,
                            error=f"Tool '{tool_name}' not found on server.",
                            tool_used=tool_name
                        )
                    elif response.status == 400:
                        error_text = await response.text()
                        return QueryResult(
                            success=False,
                            error=f"Bad request: {error_text}",
                            tool_used=tool_name
                        )
                    elif response.status != 200:
                        error_text = await response.text()
                        return QueryResult(
                            success=False,
                            error=f"Server error (HTTP {response.status}): {error_text}",
                            tool_used=tool_name
                        )
                    
                    result_data = await response.json()
                    
                    # Check for SQL syntax errors and auto-fix
                    if (not result_data.get('success', False) and 
                        tool_name == 'execute_dynamic_sql' and 
                        'sql_query' in parameters and
                        attempt < max_retries - 1 and
                        'SQL syntax' in str(result_data.get('error', ''))):
                        
                        logger.warning(f"🔧 SQL syntax error detected, attempting auto-fix...")
                        
                        # Try to fix common SQL syntax issues
                        fixed_sql = self._fix_sql_syntax(parameters['sql_query'])
                        if fixed_sql != parameters['sql_query']:
                            logger.info(f"🔧 Auto-fixed SQL: {fixed_sql}")
                            payload['parameters']['sql_query'] = fixed_sql
                            parameters['sql_query'] = fixed_sql  # Update for next iteration
                            continue  # Retry with fixed SQL
                    
                    return QueryResult(
                        success=result_data.get('success', False),
                        data=result_data.get('data'),
                        error=result_data.get('error'),
                        message=result_data.get('message'),
                        tool_used=tool_name,
                        parameters=parameters,
                        is_dynamic=(tool_name == 'execute_dynamic_sql'),
                        original_query=original_query,
                        generated_sql=parameters.get('sql_query') if tool_name == 'execute_dynamic_sql' else None
                    )
                    
            except aiohttp.ClientError as e:
                if "Connection reset by peer" in str(e) and attempt < max_retries - 1:
                    logger.warning(f"⚠️ Connection reset on attempt {attempt + 1}, retrying...")
                    await asyncio.sleep(1)  # Brief delay before retry
                    continue
                else:
                    logger.error(f"Network error calling tool {tool_name}: {e}")
                    return QueryResult(
                        success=False,
                        error=f"Network error: {str(e)}. Check if the server is running.",
                        tool_used=tool_name
                    )
            except asyncio.TimeoutError:
                if attempt < max_retries - 1:
                    logger.warning(f"⚠️ Timeout on attempt {attempt + 1}, retrying...")
                    await asyncio.sleep(1)
                    continue
                else:
                    logger.error(f"Timeout calling tool {tool_name}")
                    return QueryResult(
                        success=False,
                        error="Request timed out. The server may be overloaded.",
                        tool_used=tool_name
                    )
            except Exception as e:
                logger.error(f"Unexpected error calling tool {tool_name}: {e}", exc_info=True)
                return QueryResult(
                    success=False,
                    error=f"Unexpected error: {str(e)}",
                    tool_used=tool_name
                )
        
        # If we get here, all retries failed
        return QueryResult(
            success=False,
            error="All retry attempts failed. Connection issues with server.",
            tool_used=tool_name
        )

    async def query(self, nl_query: str) -> Union[QueryResult, List[QueryResult]]:
        """Process a natural language query"""
        try:
            parsed_calls = self.nlp.parse_query(nl_query, self.history)
            
            if len(parsed_calls) > 1:
                # Multiple tool calls
                results = await asyncio.gather(*[
                    self.call_tool(call['tool'], call['parameters'], call['original_query'])
                    for call in parsed_calls
                ])
                return results
            else:
                # Single tool call
                call = parsed_calls[0]
                return await self.call_tool(call['tool'], call['parameters'], call['original_query'])
                
        except Exception as e:
            logger.error(f"Error in query processing: {e}", exc_info=True)
            return QueryResult(
                success=False,
                error=f"Query processing failed: {e}"
            )

    def manage_history(self, query: str, response: str):
        """Manage conversation history"""
        self.history.extend([f"User: {query}", f"Assistant: {response}"])
        # Keep only last 6 entries (3 turns)
        self.history = self.history[-6:]
    
    async def submit_feedback(self, result: QueryResult, helpful: bool):
        """Submit feedback for a dynamic query with enhanced negative feedback handling."""
        if result.is_dynamic and result.generated_sql and result.original_query:
            try:
                feedback_result = await self.call_tool(
                    'record_query_feedback',
                    {
                        'original_question': result.original_query,
                        'sql_query': result.generated_sql,
                        'was_helpful': helpful
                    }
                )
                
                if feedback_result.success and feedback_result.message:
                    print(f"✅ {feedback_result.message}")
                    
                    # If negative feedback, offer to show similar queries to help understand the issue
                    if not helpful:
                        print("🔍 The system will now remember this as an example to avoid.")
                        print("💡 This helps improve future query generation for similar questions.")
                        
                        try:
                            # Get suggestions to show what the system learned
                            suggestions_result = await self.call_tool(
                                'get_query_suggestions',
                                {'original_question': result.original_query}
                            )
                            
                            if suggestions_result.success and suggestions_result.data:
                                suggestions = suggestions_result.data
                                if suggestions.get('recommendations'):
                                    print(f"\n📚 Found {suggestions['similar_queries_found']} similar queries in memory:")
                                    for i, rec in enumerate(suggestions['recommendations'][:2], 1):
                                        feedback_type = "✅ Positive" if rec['was_helpful'] else "❌ Negative"
                                        print(f"   {i}. {feedback_type} example (similarity: {rec['similarity_score']})")
                                        print(f"      Question: {rec['previous_question'][:60]}...")
                        except:
                            pass  # Don't break if suggestions fail
                else:
                    print(f"⚠️ Feedback not recorded: {feedback_result.error or 'Server may not support learning'}")
            except Exception as e:
                print(f"⚠️ Could not submit feedback (server may be offline): {str(e)}")
                print("💡 Your feedback is noted locally. The system will still work without this feature.")

# Standalone Interactive Mode
async def interactive_mode():
    """Run the interactive CLI mode"""
    print("✨ Subscription Analytics AI Agent ✨")
    print("=" * 50)
    
    # Get configuration
    config_manager = ConfigManager()
    
    try:
        user_config = config_manager.get_config()
        
        # Configure Gemini AI
        genai.configure(api_key=user_config['GOOGLE_API_KEY'])
        
        print(f"🔗 Connected to server: {user_config['SUBSCRIPTION_API_URL']}")
        
        async with UniversalClient(config=user_config) as client:
            print("\n💬 Enter questions in natural language. Type 'quit' or 'exit' to leave.")
            print("\n📚 Example queries:")
            print("  • How many new subscriptions in the last 7 days?")
            print("  • What's the payment success rate for the last month?")
            print("  • Show me payment history for user abc123")
            print("  • Which users have the highest success rates?")
            print("  • How many users have success rate above 15%?")
            
            while True:
                try:
                    query = input("\n> ").strip()
                    if query.lower() in ['quit', 'exit', 'q']:
                        break
                    if not query:
                        continue
                    
                    print("🤔 Processing your query...")
                    result = await client.query(query)
                    
                    # Format and display result
                    if isinstance(result, list):
                        output = client.formatter.format_multi_result(result, query)
                    else:
                        output = client.formatter.format_single_result(result)
                    
                    print(f"\n{output}")
                    
                    # Update conversation history
                    client.manage_history(query, output)
                    
                    # Handle feedback for dynamic queries
                    if (isinstance(result, QueryResult) and 
                        result.is_dynamic and 
                        result.success and 
                        result.data is not None):
                        
                        print("\n" + "="*50)
                        print("📝 This answer was generated using a custom SQL query.")
                        while True:
                            feedback_input = input("Was this answer helpful and accurate? (y/n/skip): ").lower().strip()
                            if feedback_input in ['y', 'yes']:
                                await client.submit_feedback(result, True)
                                break
                            elif feedback_input in ['n', 'no']:
                                await client.submit_feedback(result, False)
                                print("Thank you for the feedback. This helps improve the system.")
                                break
                            elif feedback_input in ['s', 'skip', '']:
                                break
                            else:
                                print("Please enter 'y' for yes, 'n' for no, or 'skip'.")
                
                except (KeyboardInterrupt, EOFError):
                    break
                except Exception as e:
                    logger.error("Error in interactive loop", exc_info=True)
                    print(f"❌ Error: {e}")
                    
    except Exception as e:  
        logger.error("Client failed to initialize", exc_info=True)
        print(f"❌ Critical Error: {e}")
        print("\n🔧 Troubleshooting:")
        print("1. Check your internet connection")
        print("2. Verify your API keys in client/config.json")
        print("3. Ensure the server is running and accessible")
    
    print("\n👋 Goodbye!")

if __name__ == "__main__":
    try:
        asyncio.run(interactive_mode())
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
    except Exception as e:
        print(f"❌ Failed to start client: {e}")
        print("\n💡 The client only needs Google API key and server API key, not database credentials!")