#!/usr/bin/env python3
"""
GUARANTEED WORKING API Server for Render.com
- Health endpoint is PUBLIC (no auth)
- Other endpoints require authentication
- Minimal dependencies to avoid crashes
"""

import datetime
import os
import secrets
import json
import decimal
import sys
from typing import Dict, List, Optional, Any
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Depends, Header, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator
import mysql.connector
from mysql.connector import Error
import uvicorn
from contextlib import asynccontextmanager
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
load_dotenv()

# Database Configuration
DB_CONFIG = {
    'host': os.getenv('DB_HOST'),
    'port': int(os.getenv('DB_PORT', 3306)),
    'database': os.getenv('DB_NAME'),
    'user': os.getenv('DB_USER'),
    'password': os.getenv('DB_PASSWORD'),
    'autocommit': True,
    'connect_timeout': 10,
    'charset': 'utf8mb4',
    'raise_on_warnings': False,
    'use_pure': True
}

# Pydantic Models
class ToolRequest(BaseModel):
    tool_name: str = Field(..., description="Name of the tool to execute")
    parameters: Dict = Field(default_factory=dict, description="Parameters for the tool")
    
    @field_validator('tool_name')
    @classmethod
    def validate_tool_name(cls, v):
        if not v or not isinstance(v, str):
            raise ValueError('tool_name must be a non-empty string')
        return v.strip()

class ToolResponse(BaseModel):
    success: bool
    data: Optional[Any] = None
    message: Optional[str] = None
    error: Optional[str] = None

class ToolInfo(BaseModel):
    name: str
    description: str
    parameters: Dict

# Database Functions
def get_db_connection():
    """Get database connection with error handling."""
    try:
        connection = mysql.connector.connect(**DB_CONFIG)
        if connection.is_connected():
            return connection
        else:
            logger.error("Database connection failed")
            return None
    except Error as e:
        logger.error(f"MySQL Connection Error: {e}")
        return None
    except Exception as e:
        logger.error(f"Unexpected database error: {e}")
        return None

def sanitize_for_json(obj):
    """Convert objects to JSON-serializable format."""
    if isinstance(obj, list):
        return [sanitize_for_json(item) for item in obj]
    elif isinstance(obj, dict):
        return {key: sanitize_for_json(value) for key, value in obj.items()}
    elif isinstance(obj, decimal.Decimal):
        return float(obj)
    elif isinstance(obj, (datetime.datetime, datetime.date)):
        return obj.isoformat()
    else:
        return obj

def _execute_query(query: str, params: tuple = ()):
    """Execute database query with error handling."""
    connection = None
    cursor = None
    
    try:
        logger.info(f"Executing query: {query[:100]}...")
        
        connection = get_db_connection()
        if not connection:
            return None, "Database connection failed"
        
        cursor = connection.cursor(dictionary=True)
        cursor.execute(query, params)
        results = cursor.fetchall()
        
        logger.info(f"Query executed successfully, returned {len(results)} rows")
        return results, None
        
    except Error as e:
        error_msg = f"Database error: {str(e)}"
        logger.error(f"Database query failed: {e}")
        return None, error_msg
        
    except Exception as e:
        error_msg = f"Unexpected error: {str(e)}"
        logger.error(f"Unexpected error in query execution: {e}")
        return None, error_msg
        
    finally:
        try:
            if cursor:
                cursor.close()
            if connection and connection.is_connected():
                connection.close()
        except Exception as e:
            logger.warning(f"Error closing database connection: {e}")

# Tool Functions
def get_subscriptions_in_last_days(days: int) -> Dict:
    """Get subscription statistics for the last N days."""
    try:
        days = int(days)
        if days <= 0 or days > 365:
            return {"error": "Days must be between 1 and 365"}
    except (ValueError, TypeError):
        return {"error": "Days must be a valid integer"}
    
    query = """
        SELECT 
            COUNT(*) as new_subscriptions,
            COALESCE(SUM(CASE WHEN status = 'ACTIVE' THEN 1 ELSE 0 END), 0) as active_count,
            COALESCE(SUM(CASE WHEN status = 'INACTIVE' THEN 1 ELSE 0 END), 0) as inactive_count
        FROM subscription_contract_v2 
        WHERE subcription_start_date BETWEEN %s AND %s
    """
    
    end_date = datetime.date.today()
    start_date = end_date - datetime.timedelta(days=days)
    
    results, error = _execute_query(query, (start_date, end_date))
    
    if error:
        return {"error": error}
    
    return {"data": sanitize_for_json(results[0]) if results else {}}

def get_payment_success_rate_in_last_days(days: int) -> Dict:
    """Get payment success rate and revenue for the last N days."""
    try:
        days = int(days)
        if days <= 0 or days > 365:
            return {"error": "Days must be between 1 and 365"}
    except (ValueError, TypeError):
        return {"error": "Days must be a valid integer"}
    
    query = """
        SELECT 
            COUNT(*) as total_payments,
            SUM(CASE WHEN status = 'ACTIVE' THEN 1 ELSE 0 END) as successful_payments,
            SUM(CASE WHEN status != 'ACTIVE' OR status IS NULL THEN 1 ELSE 0 END) as failed_payments,
            SUM(CASE WHEN status = 'ACTIVE' THEN trans_amount_decimal ELSE 0 END) as total_revenue,
            ROUND(
                (SUM(CASE WHEN status = 'ACTIVE' THEN 1 ELSE 0 END) * 100.0 / 
                 NULLIF(COUNT(*), 0)), 
                2
            ) as success_rate_percent
        FROM subscription_payment_details 
        WHERE created_date BETWEEN %s AND %s
    """
    
    end_date = datetime.date.today()
    start_date = end_date - datetime.timedelta(days=days)
    
    results, error = _execute_query(query, (start_date, end_date))
    
    if error:
        return {"error": error}
    
    return {"data": sanitize_for_json(results[0]) if results else {}}

def get_user_payment_history(merchant_user_id: str, days: int = 90) -> Dict:
    """Get payment history for a specific user."""
    if not merchant_user_id or not isinstance(merchant_user_id, str):
        return {"error": "merchant_user_id must be a non-empty string"}
    
    try:
        days = int(days)
        if days <= 0 or days > 365:
            days = 90
    except (ValueError, TypeError):
        days = 90
    
    query = """
        SELECT 
            spd.created_date as payment_date,
            spd.trans_amount_decimal as payment_amount,
            spd.status as payment_status,
            scv.status as subscription_status
        FROM subscription_payment_details as spd
        JOIN subscription_contract_v2 as scv ON spd.subscription_id = scv.subscription_id
        WHERE scv.merchant_user_id = %s 
        AND spd.created_date >= %s
        ORDER BY spd.created_date DESC
        LIMIT 100
    """
    
    cutoff_date = datetime.date.today() - datetime.timedelta(days=days)
    results, error = _execute_query(query, (merchant_user_id, cutoff_date))
    
    if error:
        return {"error": error}
    
    if not results:
        return {"message": f"No payment history found for user {merchant_user_id} in the last {days} days"}
    
    return {"data": sanitize_for_json(results)}

def get_database_status() -> Dict:
    """Check database connection and get basic statistics."""
    query = """
        SELECT 
            (SELECT COUNT(*) FROM subscription_contract_v2) as total_subscriptions,
            (SELECT COUNT(*) FROM subscription_payment_details) as total_payments,
            (SELECT COUNT(DISTINCT merchant_user_id) FROM subscription_contract_v2) as unique_users
    """
    
    results, error = _execute_query(query)
    
    if error:
        return {"error": error}
    
    status_data = {
        "status": "connected",
        "timestamp": datetime.datetime.now().isoformat(),
        "platform": "render.com"
    }
    
    if results:
        status_data.update(sanitize_for_json(results[0]))
    
    return {"data": status_data}

def execute_dynamic_sql(sql_query: str) -> Dict:
    """Execute dynamic SQL query with safety checks."""
    logger.info(f"Dynamic SQL request received: {sql_query[:100]}...")
    
    try:
        # Input validation
        if not sql_query or not isinstance(sql_query, str):
            return {"error": "SQL query is required and must be a string"}
        
        # Only allow SELECT statements
        cleaned_query = sql_query.strip()
        if not cleaned_query.upper().startswith('SELECT'):
            return {"error": "Only SELECT statements are allowed"}
        
        logger.info(f"Executing: {sql_query}")
        
        # Execute the query
        results, error = _execute_query(sql_query)
        
        if error:
            logger.error(f"SQL execution failed: {error}")
            return {"error": f"SQL execution failed: {error}"}
        
        if not results:
            logger.info("Query executed successfully but returned no results")
            return {"message": "Query executed successfully, but no matching records were found."}
        
        # Limit result size for performance
        if len(results) > 1000:
            results = results[:1000]
            logger.info(f"Results limited to first 1000 rows")
            return {
                "data": sanitize_for_json(results),
                "message": f"Results limited to first 1000 rows (query returned more)"
            }
        
        logger.info(f"Dynamic SQL completed successfully with {len(results)} rows")
        return {"data": sanitize_for_json(results)}
        
    except Exception as e:
        logger.error(f"Unexpected error in dynamic SQL execution: {e}")
        return {"error": f"Unexpected error during SQL execution: {str(e)}"}

# Tool Registry
TOOL_REGISTRY = {
    "get_subscriptions_in_last_days": {
        "function": get_subscriptions_in_last_days,
        "description": "Get subscription statistics for the last N days",
        "parameters": {
            "type": "object",
            "properties": {
                "days": {"type": "integer", "description": "Number of days to look back (1-365)"}
            },
            "required": ["days"]
        }
    },
    "get_payment_success_rate_in_last_days": {
        "function": get_payment_success_rate_in_last_days,
        "description": "Get payment success rate and revenue statistics for the last N days",
        "parameters": {
            "type": "object",
            "properties": {
                "days": {"type": "integer", "description": "Number of days to look back (1-365)"}
            },
            "required": ["days"]
        }
    },
    "get_user_payment_history": {
        "function": get_user_payment_history,
        "description": "Get payment history for a specific user by merchant_user_id",
        "parameters": {
            "type": "object",
            "properties": {
                "merchant_user_id": {"type": "string", "description": "The merchant user ID"},
                "days": {"type": "integer", "description": "Number of days to look back (default: 90)"}
            },
            "required": ["merchant_user_id"]
        }
    },
    "get_database_status": {
        "function": get_database_status,
        "description": "Check database connection and get basic statistics",
        "parameters": {"type": "object", "properties": {}}
    },
    "execute_dynamic_sql": {
        "function": execute_dynamic_sql,
        "description": "Execute a custom SELECT SQL query for complex analytics",
        "parameters": {
            "type": "object",
            "properties": {
                "sql_query": {"type": "string", "description": "SELECT SQL query to execute"}
            },
            "required": ["sql_query"]
        }
    }
}

# API Configuration
API_KEY = os.getenv("API_KEY_1")
if not API_KEY:
    logger.error("FATAL: API_KEY_1 environment variable is not set")
    raise ValueError("API_KEY_1 must be set")

def verify_api_key(authorization: str = Header(None)):
    """Verify API key from Authorization header."""
    if not authorization:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authorization header is required"
        )
    
    if not authorization.startswith("Bearer "):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authorization header must start with 'Bearer '"
        )
    
    token = authorization.split(" ")[1]
    if not secrets.compare_digest(token, API_KEY):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key"
        )

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    logger.info("🚀 Starting RENDER.COM Subscription Analytics API Server")
    logger.info(f"Available tools: {len(TOOL_REGISTRY)}")
    yield
    logger.info("🛑 Shutting down API Server")

# CRITICAL: Create FastAPI app WITHOUT global authentication
app = FastAPI(
    title="Subscription Analytics API",
    description="Render.com deployment with public health endpoint",
    version="6.0.0-render-guaranteed-fix",
    lifespan=lifespan
    # IMPORTANT: NO dependencies=[Depends(verify_api_key)] here!
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# HEALTH ENDPOINT - COMPLETELY PUBLIC (NO AUTH)
@app.get("/health")
def health_check():
    """Health check endpoint - COMPLETELY PUBLIC for Render.com health checks."""
    return {
        "status":