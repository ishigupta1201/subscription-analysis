#!/usr/bin/env python3
"""
Corrected API Server for Subscription Analytics
Fixed version with enhanced error handling and proper date range support
"""

# 1. IMPORTS
import datetime
import os
import secrets
from typing import Dict, List, Optional
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Depends, Header, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field
import mysql.connector
from mysql.connector import Error
import uvicorn
from contextlib import asynccontextmanager
import logging
import traceback

# Configure enhanced logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# DEBUG: Print all environment variables related to database
print("🔍 === DATABASE DEBUG INFO ===")
print(f"DB_HOST: '{os.getenv('DB_HOST')}'")
print(f"DB_PORT: '{os.getenv('DB_PORT')}'")
print(f"DB_USER: '{os.getenv('DB_USER')}'")
print(f"DB_NAME: '{os.getenv('DB_NAME')}'")
print(f"DB_PASSWORD exists: {bool(os.getenv('DB_PASSWORD'))}")
print("🔍 === END DEBUG INFO ===")

# Enhanced Database Configuration
DB_CONFIG = {
    'host': os.getenv('DB_HOST', 'localhost'),
    'port': int(os.getenv('DB_PORT', '3306')),
    'database': os.getenv('DB_NAME', 'SUBS_STAGING'),
    'user': os.getenv('DB_USER', 'root'),
    'password': os.getenv('DB_PASSWORD', '12345678'),
    'autocommit': True,
    'connect_timeout': 30,
    'raise_on_warnings': False
}

print(f"🔍 Final DB_CONFIG: host={DB_CONFIG['host']}, port={DB_CONFIG['port']}, database={DB_CONFIG['database']}")

# --- API Models ---
class ToolRequest(BaseModel):
    tool_name: str = Field(..., description="Name of the tool to execute")
    parameters: Dict = Field(default_factory=dict, description="Parameters for the tool")

class ToolResponse(BaseModel):
    success: bool
    data: Optional[Dict] = None
    error: Optional[str] = None

class ToolInfo(BaseModel):
    name: str
    description: str
    parameters: Dict

class APIStatus(BaseModel):
    status: str
    version: str
    available_tools: int
    database_status: str

# 2. ENHANCED DATABASE FUNCTIONS

def get_db_connection():
    """Create and return a database connection with enhanced error handling"""
    try:
        logger.debug(f"Attempting database connection...")
        connection = mysql.connector.connect(**DB_CONFIG)
        if connection.is_connected():
            logger.debug("✅ Database connection successful")
            return connection
        else:
            logger.error("❌ Database connection failed - not connected")
            return None
    except Error as e:
        logger.error(f"❌ MySQL Error: {e}")
        logger.error(f"Error Code: {e.errno if hasattr(e, 'errno') else 'No code'}")
        return None
    except Exception as e:
        logger.error(f"❌ Unexpected database error: {e}")
        return None

def validate_date_format(date_string: str, date_name: str) -> datetime.date:
    """Validate and parse date string"""
    try:
        return datetime.datetime.strptime(date_string, '%Y-%m-%d').date()
    except ValueError as e:
        raise ValueError(f"Invalid {date_name} format. Use YYYY-MM-DD. Error: {str(e)}")

def get_subscriptions_in_last_days(days: int) -> Dict:
    """Get subscription data for the last x days - ENHANCED VERSION"""
    # Input validation
    if not isinstance(days, int) or days <= 0 or days > 365:
        return {"error": "Days must be an integer between 1 and 365"}
    
    connection = get_db_connection()
    if not connection:
        return {"error": "Database connection failed"}
    
    try:
        cursor = connection.cursor(dictionary=True)
        today = datetime.date.today()
        start_date = today - datetime.timedelta(days=days)
        
        query = """
            SELECT 
                COUNT(*) as new_subscriptions,
                SUM(CASE WHEN status = 'ACTIVE' THEN 1 ELSE 0 END) as active_count,
                SUM(CASE WHEN status IN ('CLOSED', 'REJECT') THEN 1 ELSE 0 END) as cancelled_count
            FROM subscription_contract_v2 
            WHERE subcription_start_date BETWEEN %s AND %s
        """
        
        logger.info(f"Executing subscription query for last {days} days: {start_date} to {today}")
        cursor.execute(query, (start_date, today))
        result = cursor.fetchone()
        
        if not result:
            logger.warning("No results returned from subscription query")
            return {"error": "No data returned from database"}
        
        response_data = {
            "new_subscriptions": result['new_subscriptions'] or 0,
            "active_subscriptions": result['active_count'] or 0,
            "cancelled_subscriptions": result['cancelled_count'] or 0,
            "period_days": days,
            "date_range": {
                "start": str(start_date),
                "end": str(today)
            }
        }
        
        logger.info(f"Subscription query successful: {response_data}")
        return response_data
        
    except Error as e:
        error_msg = f"Database query failed: {str(e)}"
        logger.error(error_msg)
        return {"error": error_msg}
    except Exception as e:
        error_msg = f"Unexpected error: {str(e)}"
        logger.error(error_msg)
        logger.error(traceback.format_exc())
        return {"error": error_msg}
    finally:
        if connection and connection.is_connected():
            cursor.close()
            connection.close()

def get_payment_success_rate_in_last_days(days: int) -> Dict:
    """Get payment success rate for the last x days - ENHANCED VERSION"""
    # Input validation
    if not isinstance(days, int) or days <= 0 or days > 365:
        return {"error": "Days must be an integer between 1 and 365"}
    
    connection = get_db_connection()
    if not connection:
        return {"error": "Database connection failed"}
    
    try:
        cursor = connection.cursor(dictionary=True)
        today = datetime.date.today()
        start_date = today - datetime.timedelta(days=days)
        
        query = """
            SELECT 
                COUNT(*) as total_payments,
                SUM(CASE WHEN status = 'ACTIVE' THEN 1 ELSE 0 END) as successful_payments,
                SUM(CASE WHEN status = 'FAIL' THEN 1 ELSE 0 END) as failed_payments,
                SUM(CASE WHEN status = 'ACTIVE' THEN trans_amount_decimal ELSE 0 END) as total_revenue,
                SUM(CASE WHEN status = 'FAIL' THEN trans_amount_decimal ELSE 0 END) as lost_revenue
            FROM subscription_payment_details 
            WHERE created_date BETWEEN %s AND %s
        """
        
        logger.info(f"Executing payment query for last {days} days: {start_date} to {today}")
        cursor.execute(query, (start_date, today))
        result = cursor.fetchone()
        
        if not result:
            logger.warning("No results returned from payment query")
            return {"error": "No data returned from database"}
        
        total_payments = result['total_payments'] or 0
        successful_payments = result['successful_payments'] or 0
        failed_payments = result['failed_payments'] or 0
        total_revenue = float(result['total_revenue'] or 0)
        lost_revenue = float(result['lost_revenue'] or 0)
        
        if total_payments == 0:
            response_data = {
                "success_rate": "0.00%",
                "failure_rate": "0.00%",
                "total_payments": 0,
                "successful_payments": 0,
                "failed_payments": 0,
                "total_revenue": "$0.00",
                "lost_revenue": "$0.00",
                "period_days": days,
                "date_range": {"start": str(start_date), "end": str(today)}
            }
        else:
            success_rate = (successful_payments / total_payments) * 100
            failure_rate = (failed_payments / total_payments) * 100
            
            response_data = {
                "success_rate": f"{success_rate:.2f}%",
                "failure_rate": f"{failure_rate:.2f}%",
                "total_payments": total_payments,
                "successful_payments": successful_payments,
                "failed_payments": failed_payments,
                "total_revenue": f"${total_revenue:.2f}",
                "lost_revenue": f"${lost_revenue:.2f}",
                "period_days": days,
                "date_range": {"start": str(start_date), "end": str(today)}
            }
        
        logger.info(f"Payment query successful: {response_data}")
        return response_data
        
    except Error as e:
        error_msg = f"Database query failed: {str(e)}"
        logger.error(error_msg)
        return {"error": error_msg}
    except Exception as e:
        error_msg = f"Unexpected error: {str(e)}"
        logger.error(error_msg)
        logger.error(traceback.format_exc())
        return {"error": error_msg}
    finally:
        if connection and connection.is_connected():
            cursor.close()
            connection.close()

def get_subscription_summary(days: int = 30) -> Dict:
    """Get comprehensive subscription and payment summary - ENHANCED VERSION"""
    if not isinstance(days, int) or days <= 0 or days > 365:
        return {"error": "Days must be an integer between 1 and 365"}
    
    logger.info(f"Getting subscription summary for {days} days")
    
    subscription_data = get_subscriptions_in_last_days(days)
    payment_data = get_payment_success_rate_in_last_days(days)
    
    if "error" in subscription_data or "error" in payment_data:
        return {
            "error": "Failed to fetch data from database",
            "subscription_error": subscription_data.get("error"),
            "payment_error": payment_data.get("error")
        }
    
    return {
        "period_days": days,
        "subscriptions": subscription_data,
        "payments": payment_data,
        "summary": f"In the last {days} days: {subscription_data['new_subscriptions']} new subscriptions, "
                  f"{payment_data['successful_payments']} successful payments ({payment_data['success_rate']}), "
                  f"total revenue: {payment_data['total_revenue']}"
    }

def get_database_status() -> Dict:
    """Check database connection and get basic statistics - ENHANCED VERSION"""
    connection = get_db_connection()
    if not connection:
        return {"status": "disconnected", "error": "Cannot connect to database"}
    
    try:
        cursor = connection.cursor(dictionary=True)
        
        # Test basic connectivity
        cursor.execute("SELECT 1 as test")
        test_result = cursor.fetchone()
        if not test_result or test_result['test'] != 1:
            return {"status": "error", "error": "Database test query failed"}
        
        # Get table counts
        cursor.execute("SELECT COUNT(*) as count FROM subscription_contract_v2")
        total_subscriptions = cursor.fetchone()['count']
        
        cursor.execute("SELECT COUNT(*) as count FROM subscription_payment_details")
        total_payments = cursor.fetchone()['count']

        cursor.execute("SELECT COUNT(DISTINCT merchant_user_id) as count FROM subscription_contract_v2")
        unique_users = cursor.fetchone()['count']
        
        # Get latest dates
        cursor.execute("SELECT MAX(subcription_start_date) as latest_sub FROM subscription_contract_v2")
        latest_sub = cursor.fetchone()['latest_sub']
        
        cursor.execute("SELECT MAX(created_date) as latest_payment FROM subscription_payment_details")
        latest_payment = cursor.fetchone()['latest_payment']
        
        # Calculate overall success rate
        cursor.execute("""
            SELECT 
                COUNT(*) as total,
                SUM(CASE WHEN status = 'ACTIVE' THEN 1 ELSE 0 END) as successful
            FROM subscription_payment_details
        """)
        stats = cursor.fetchone()
        overall_success_rate = (stats['successful'] / stats['total'] * 100) if stats['total'] > 0 else 0
        
        return {
            "status": "connected",
            "database": DB_CONFIG['database'],
            "host": DB_CONFIG['host'],
            "tables": ["subscription_contract_v2", "subscription_payment_details"],
            "total_subscriptions": total_subscriptions,
            "total_payments": total_payments,
            "unique_users": unique_users,
            "latest_subscription": str(latest_sub) if latest_sub else None,
            "latest_payment": str(latest_payment) if latest_payment else None,
            "overall_success_rate": f"{overall_success_rate:.2f}%"
        }
        
    except Error as e:
        error_msg = f"Database status check failed: {str(e)}"
        logger.error(error_msg)
        return {"status": "error", "error": error_msg}
    except Exception as e:
        error_msg = f"Unexpected error in database status: {str(e)}"
        logger.error(error_msg)
        logger.error(traceback.format_exc())
        return {"status": "error", "error": error_msg}
    finally:
        if connection and connection.is_connected():
            cursor.close()
            connection.close()

def get_user_payment_history(merchant_user_id: str, days: int = 90) -> Dict:
    """Get payment history for a specific user - ENHANCED VERSION"""
    # Input validation
    if not isinstance(days, int) or days <= 0 or days > 365:
        return {"error": "Days must be an integer between 1 and 365"}
    
    if not merchant_user_id or not merchant_user_id.strip():
        return {"error": "merchant_user_id cannot be empty"}
    
    connection = get_db_connection()
    if not connection:
        return {"error": "Database connection failed"}
    
    try:
        cursor = connection.cursor(dictionary=True)
        today = datetime.date.today()
        start_date = today - datetime.timedelta(days=days)
        
        query = """
            SELECT 
                spd.created_date as payment_date,
                spd.trans_amount_decimal as payment_amount,
                spd.status as payment_status,
                scv.status as subscription_status,
                scv.subcription_start_date
            FROM 
                subscription_payment_details as spd
            JOIN 
                subscription_contract_v2 as scv ON spd.subscription_id = scv.subscription_id
            WHERE 
                scv.merchant_user_id = %s AND spd.created_date BETWEEN %s AND %s
            ORDER BY 
                spd.created_date DESC
        """
        
        logger.info(f"Getting payment history for user {merchant_user_id} for last {days} days")
        cursor.execute(query, (merchant_user_id, start_date, today))
        payments = cursor.fetchall()
        
        if not payments:
            return {
                "merchant_user_id": merchant_user_id,
                "period_days": days,
                "message": f"No payments found for user {merchant_user_id} in the last {days} days"
            }
            
        total_payments = len(payments)
        successful = sum(1 for p in payments if p['payment_status'] == 'ACTIVE')
        total_spent = sum(p['payment_amount'] for p in payments if p['payment_status'] == 'ACTIVE')
        
        return {
            "merchant_user_id": merchant_user_id,
            "period_days": days,
            "total_payments": total_payments,
            "successful_payments": successful,
            "success_rate": f"{(successful/total_payments*100):.2f}%" if total_payments > 0 else "0.00%",
            "total_spent": f"${total_spent:.2f}",
            "subscription_start": str(payments[0]['subcription_start_date']) if payments else None,
            "current_status": payments[0]['subscription_status'] if payments else None,
            "payments": [
                {
                    "date": str(p['payment_date']),
                    "amount": f"${p['payment_amount']:.2f}",
                    "status": "success" if p['payment_status'] == 'ACTIVE' else "failure"
                }
                for p in payments
            ]
        }
        
    except Error as e:
        error_msg = f"User payment history query failed: {str(e)}"
        logger.error(error_msg)
        return {"error": error_msg}
    except Exception as e:
        error_msg = f"Unexpected error in user payment history: {str(e)}"
        logger.error(error_msg)
        logger.error(traceback.format_exc())
        return {"error": error_msg}
    finally:
        if connection and connection.is_connected():
            cursor.close()
            connection.close()

# FIXED DATE RANGE FUNCTIONS

def get_subscriptions_by_date_range(start_date: str, end_date: str) -> Dict:
    """Get subscription data for a specific date range - FIXED VERSION"""
    connection = get_db_connection()
    if not connection:
        return {"error": "Database connection failed"}
    
    try:
        # Validate and parse dates
        start_dt = validate_date_format(start_date, "start_date")
        end_dt = validate_date_format(end_date, "end_date")
        
        # Validate date range
        if start_dt > end_dt:
            return {"error": "Start date cannot be after end date"}
        
        # Check if date range is too far in the future
        today = datetime.date.today()
        if start_dt > today:
            return {"error": "Start date cannot be in the future"}
        
        cursor = connection.cursor(dictionary=True)
        
        query = """
            SELECT 
                COUNT(*) as new_subscriptions,
                SUM(CASE WHEN status = 'ACTIVE' THEN 1 ELSE 0 END) as active_count,
                SUM(CASE WHEN status IN ('CLOSED', 'REJECT') THEN 1 ELSE 0 END) as cancelled_count
            FROM subscription_contract_v2 
            WHERE subcription_start_date BETWEEN %s AND %s
        """
        
        logger.info(f"Executing subscription date range query from {start_dt} to {end_dt}")
        cursor.execute(query, (start_dt, end_dt))
        result = cursor.fetchone()
        
        if not result:
            logger.warning("No results returned from subscription date range query")
            return {"error": "No data returned from database"}
        
        # Calculate period days
        period_days = (end_dt - start_dt).days + 1
        
        response_data = {
            "new_subscriptions": result['new_subscriptions'] or 0,
            "active_subscriptions": result['active_count'] or 0,
            "cancelled_subscriptions": result['cancelled_count'] or 0,
            "period_days": period_days,
            "date_range": {
                "start": str(start_dt),
                "end": str(end_dt)
            }
        }
        
        logger.info(f"Subscription date range query successful: {response_data}")
        return response_data
        
    except ValueError as e:
        error_msg = str(e)
        logger.error(f"Date validation error: {error_msg}")
        return {"error": error_msg}
    except Error as e:
        error_msg = f"Database query failed: {str(e)}"
        logger.error(error_msg)
        logger.error(f"Query was: {query}")
        logger.error(f"Parameters: start_dt={start_dt}, end_dt={end_dt}")
        return {"error": error_msg}
    except Exception as e:
        error_msg = f"Unexpected error in get_subscriptions_by_date_range: {str(e)}"
        logger.error(error_msg)
        logger.error(traceback.format_exc())
        return {"error": error_msg}
    finally:
        if connection and connection.is_connected():
            cursor.close()
            connection.close()

def get_payments_by_date_range(start_date: str, end_date: str) -> Dict:
    """Get payment data for a specific date range - FIXED VERSION"""
    connection = get_db_connection()
    if not connection:
        return {"error": "Database connection failed"}
    
    try:
        # Validate and parse dates
        start_dt = validate_date_format(start_date, "start_date")
        end_dt = validate_date_format(end_date, "end_date")
        
        # Validate date range
        if start_dt > end_dt:
            return {"error": "Start date cannot be after end date"}
        
        # Check if date range is too far in the future
        today = datetime.date.today()
        if start_dt > today:
            return {"error": "Start date cannot be in the future"}
        
        cursor = connection.cursor(dictionary=True)
        
        query = """
            SELECT 
                COUNT(*) as total_payments,
                SUM(CASE WHEN status = 'ACTIVE' THEN 1 ELSE 0 END) as successful_payments,
                SUM(CASE WHEN status = 'FAIL' THEN 1 ELSE 0 END) as failed_payments,
                SUM(CASE WHEN status = 'ACTIVE' THEN trans_amount_decimal ELSE 0 END) as total_revenue,
                SUM(CASE WHEN status = 'FAIL' THEN trans_amount_decimal ELSE 0 END) as lost_revenue
            FROM subscription_payment_details 
            WHERE created_date BETWEEN %s AND %s
        """
        
        logger.info(f"Executing payment date range query from {start_dt} to {end_dt}")
        cursor.execute(query, (start_dt, end_dt))
        result = cursor.fetchone()
        
        if not result:
            logger.warning("No results returned from payment date range query")
            return {"error": "No data returned from database"}
        
        total_payments = result['total_payments'] or 0
        successful_payments = result['successful_payments'] or 0
        failed_payments = result['failed_payments'] or 0
        total_revenue = float(result['total_revenue'] or 0)
        lost_revenue = float(result['lost_revenue'] or 0)
        
        # Calculate period days
        period_days = (end_dt - start_dt).days + 1
        
        if total_payments == 0:
            response_data = {
                "success_rate": "0.00%",
                "failure_rate": "0.00%",
                "total_payments": 0,
                "successful_payments": 0,
                "failed_payments": 0,
                "total_revenue": "$0.00",
                "lost_revenue": "$0.00",
                "period_days": period_days,
                "date_range": {"start": str(start_dt), "end": str(end_dt)}
            }
        else:
            success_rate = (successful_payments / total_payments) * 100
            failure_rate = (failed_payments / total_payments) * 100
            
            response_data = {
                "success_rate": f"{success_rate:.2f}%",
                "failure_rate": f"{failure_rate:.2f}%",
                "total_payments": total_payments,
                "successful_payments": successful_payments,
                "failed_payments": failed_payments,
                "total_revenue": f"${total_revenue:.2f}",
                "lost_revenue": f"${lost_revenue:.2f}",
                "period_days": period_days,
                "date_range": {"start": str(start_dt), "end": str(end_dt)}
            }
        
        logger.info(f"Payment date range query successful: {response_data}")
        return response_data
        
    except ValueError as e:
        error_msg = str(e)
        logger.error(f"Date validation error: {error_msg}")
        return {"error": error_msg}
    except Error as e:
        error_msg = f"Database query failed: {str(e)}"
        logger.error(error_msg)
        logger.error(f"Query was: {query}")
        logger.error(f"Parameters: start_dt={start_dt}, end_dt={end_dt}")
        return {"error": error_msg}
    except Exception as e:
        error_msg = f"Unexpected error in get_payments_by_date_range: {str(e)}"
        logger.error(error_msg)
        logger.error(traceback.format_exc())
        return {"error": error_msg}
    finally:
        if connection and connection.is_connected():
            cursor.close()
            connection.close()

def get_analytics_by_date_range(start_date: str, end_date: str) -> Dict:
    """Get comprehensive analytics for a specific date range - FIXED VERSION"""
    logger.info(f"Getting analytics for date range: {start_date} to {end_date}")
    
    try:
        # Validate dates first before making any database calls
        start_dt = validate_date_format(start_date, "start_date")
        end_dt = validate_date_format(end_date, "end_date")
        
        if start_dt > end_dt:
            return {"error": "Start date cannot be after end date"}
        
        # Check if date range is too far in the future
        today = datetime.date.today()
        if start_dt > today:
            return {"error": "Start date cannot be in the future"}
        
        # Get subscription and payment data
        subscription_data = get_subscriptions_by_date_range(start_date, end_date)
        payment_data = get_payments_by_date_range(start_date, end_date)
        
        # Check for errors in either dataset
        if "error" in subscription_data:
            logger.error(f"Subscription data error: {subscription_data['error']}")
            return {
                "error": "Failed to fetch subscription data from database",
                "subscription_error": subscription_data["error"]
            }
        
        if "error" in payment_data:
            logger.error(f"Payment data error: {payment_data['error']}")
            return {
                "error": "Failed to fetch payment data from database", 
                "payment_error": payment_data["error"]
            }
        
        period_days = subscription_data.get('period_days', 0)
        
        analytics_response = {
            "start_date": start_date,
            "end_date": end_date,
            "period_days": period_days,
            "subscriptions": subscription_data,
            "payments": payment_data,
            "summary": f"From {start_date} to {end_date} ({period_days} days): "
                      f"{subscription_data['new_subscriptions']} new subscriptions, "
                      f"{payment_data['successful_payments']} successful payments ({payment_data['success_rate']}), "
                      f"total revenue: {payment_data['total_revenue']}"
        }
        
        logger.info(f"Analytics query successful for {start_date} to {end_date}")
        return analytics_response
        
    except ValueError as e:
        error_msg = str(e)
        logger.error(f"Date validation error in analytics: {error_msg}")
        return {"error": error_msg}
    except Exception as e:
        error_msg = f"Unexpected error in get_analytics_by_date_range: {str(e)}"
        logger.error(error_msg)
        logger.error(traceback.format_exc())
        return {"error": error_msg}

# 3. API KEY MANAGER CLASS
class APIKeyManager:
    def __init__(self):
        # Get API keys from environment variables
        api_key_1 = os.getenv('API_KEY_1')
        api_key_2 = os.getenv('API_KEY_2')
        
        # If no API keys found, handle based on environment
        if not api_key_1 or not api_key_2:
            if os.getenv('PORT') or os.getenv('RAILWAY_ENVIRONMENT'):
                # Running on Railway - API keys should be set as environment variables
                logger.error("API keys not found in environment variables!")
                logger.error("Please set API_KEY_1 and API_KEY_2 in Railway dashboard")
                raise ValueError("Missing API keys in production environment")
            else:
                # Running locally - generate new keys and save to .env
                logger.warning("API keys not found in .env file, generating new ones...")
                api_key_1 = 'sub_analytics_' + secrets.token_urlsafe(32)
                api_key_2 = 'sub_analytics_' + secrets.token_urlsafe(32)
                
                # Update .env file with new keys (only for local development)
                try:
                    with open('.env', 'a') as f:
                        f.write(f"\nAPI_KEY_1={api_key_1}")
                        f.write(f"\nAPI_KEY_2={api_key_2}")
                    logger.info(f"Generated new API keys and saved to .env file")
                except Exception as e:
                    logger.warning(f"Could not write to .env file: {e}")
        
        self.api_keys = {api_key_1, api_key_2}
        logger.info(f"Initialized with {len(self.api_keys)} API keys")
    
    def is_valid_key(self, api_key: str) -> bool:
        return api_key in self.api_keys
    
    def generate_new_key(self) -> str:
        new_key = 'sub_analytics_' + secrets.token_urlsafe(32)
        self.api_keys.add(new_key)
        return new_key

# 4. INITIALIZE API_KEY_MANAGER
try:
    api_key_manager = APIKeyManager()
    logger.info("✅ API Key Manager initialized successfully")
except Exception as e:
    logger.error(f"❌ Failed to initialize API Key Manager: {e}")
    raise

# 5. VERIFY_API_KEY FUNCTION
def verify_api_key(authorization: str = Header(None)) -> str:
    """Verify API key from Authorization header"""
    if not authorization:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing Authorization header"
        )
    
    if not authorization.startswith("Bearer "):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid Authorization header format. Use 'Bearer <api_key>'"
        )
    
    api_key = authorization.split(" ")[1]
    if not api_key_manager.is_valid_key(api_key):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key"
        )
    
    return api_key

# --- ENHANCED TOOL REGISTRY ---
TOOL_REGISTRY = {
    "get_subscriptions_in_last_days": {
        "function": get_subscriptions_in_last_days,
        "description": "Get subscription statistics for the last N days",
        "parameters": {
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
    },
    "get_payment_success_rate_in_last_days": {
        "function": get_payment_success_rate_in_last_days,
        "description": "Get payment success rate statistics for the last N days",
        "parameters": {
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
    },
    "get_subscription_summary": {
        "function": get_subscription_summary,
        "description": "Get comprehensive subscription and payment summary",
        "parameters": {
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
    },
    "get_database_status": {
        "function": get_database_status,
        "description": "Check database connection and get basic statistics",
        "parameters": {
            "type": "object",
            "properties": {},
            "required": []
        }
    },
    "get_user_payment_history": {
        "function": get_user_payment_history,
        "description": "Get payment history for a specific user",
        "parameters": {
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
    },
    "get_subscriptions_by_date_range": {
        "function": get_subscriptions_by_date_range,
        "description": "Get subscription statistics for a specific date range",
        "parameters": {
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
    },
    "get_payments_by_date_range": {
        "function": get_payments_by_date_range,
        "description": "Get payment statistics for a specific date range",
        "parameters": {
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
    },
    "get_analytics_by_date_range": {
        "function": get_analytics_by_date_range,
        "description": "Get comprehensive analytics for a specific date range",
        "parameters": {
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
    }
}

# 6. FASTAPI APP CREATION
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("🚀 Starting Subscription Analytics API...")
    logger.info(f"Environment: {'Production' if os.getenv('PORT') else 'Development'}")
    
    # Test database connection on startup
    db_status = get_database_status()
    if db_status.get("status") == "connected":
        logger.info("✅ Database connection verified on startup")
        logger.info(f"Database: {db_status.get('database')} on {db_status.get('host')}")
        logger.info(f"Total subscriptions: {db_status.get('total_subscriptions')}")
        logger.info(f"Total payments: {db_status.get('total_payments')}")
    else:
        logger.warning("⚠️ Database connection failed on startup")
        logger.warning(f"Error: {db_status.get('error', 'Unknown error')}")
    
    yield
    
    # Shutdown
    logger.info("🛑 Shutting down Subscription Analytics API...")

app = FastAPI(
    title="Subscription Analytics API",
    description="Enhanced HTTP API for subscription payment analytics with date range support",
    version="1.1.0",
    lifespan=lifespan,
    docs_url="/docs" if not os.getenv('PORT') else None,  # Disable docs in production
    redoc_url="/redoc" if not os.getenv('PORT') else None
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure this properly for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 7. API ROUTES

@app.get("/", response_model=APIStatus)
async def root():
    """API status and basic information"""
    db_status = get_database_status()
    return APIStatus(
        status="online",
        version="1.1.0",
        available_tools=len(TOOL_REGISTRY),
        database_status=db_status.get("status", "unknown")
    )

@app.get("/health")
async def health_check():
    """Enhanced health check endpoint"""
    db_status = get_database_status()
    
    health_status = {
        "status": "healthy" if db_status.get("status") == "connected" else "unhealthy",
        "timestamp": datetime.datetime.now().isoformat(),
        "version": "1.1.0",
        "database": {
            "status": db_status.get("status", "unknown"),
            "host": db_status.get("host", "unknown"),
            "database": db_status.get("database", "unknown")
        },
        "environment": "production" if os.getenv('PORT') else "development"
    }
    
    if "error" in db_status:
        health_status["database"]["error"] = db_status["error"]
    
    return health_status

@app.get("/tools", response_model=List[ToolInfo])
async def list_tools(api_key: str = Depends(verify_api_key)):
    """List all available tools"""
    tools = []
    for tool_name, tool_config in TOOL_REGISTRY.items():
        tools.append(ToolInfo(
            name=tool_name,
            description=tool_config["description"],
            parameters=tool_config["parameters"]
        ))
    return tools

@app.post("/execute", response_model=ToolResponse)
async def execute_tool(
    request: ToolRequest, 
    api_key: str = Depends(verify_api_key)
):
    """Execute a specific tool with parameters - ENHANCED VERSION"""
    tool_name = request.tool_name
    parameters = request.parameters
    
    logger.info(f"🔧 Executing tool: {tool_name} with parameters: {parameters}")
    
    if tool_name not in TOOL_REGISTRY:
        logger.error(f"❌ Tool not found: {tool_name}")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Tool '{tool_name}' not found. Available tools: {list(TOOL_REGISTRY.keys())}"
        )
    
    try:
        tool_function = TOOL_REGISTRY[tool_name]["function"]
        
        # Execute the tool function
        result = tool_function(**parameters)
        
        # Check if the result contains an error
        if isinstance(result, dict) and "error" in result:
            logger.error(f"❌ Tool execution error: {result['error']}")
            return ToolResponse(success=False, error=result["error"])
        
        logger.info(f"✅ Tool {tool_name} executed successfully")
        return ToolResponse(success=True, data=result)
        
    except TypeError as e:
        error_msg = f"Invalid parameters for tool '{tool_name}': {str(e)}"
        logger.error(f"❌ Parameter error: {error_msg}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=error_msg
        )
    except Exception as e:
        error_msg = f"Tool execution failed: {str(e)}"
        logger.error(f"❌ Execution error: {error_msg}")
        logger.error(traceback.format_exc())
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=error_msg
        )

# DEBUG ROUTES (only available in development)
if not os.getenv('PORT'):
    @app.get("/debug/db-test")
    async def test_database(api_key: str = Depends(verify_api_key)):
        """Test database connection and table structure"""
        connection = get_db_connection()
        if not connection:
            return {"error": "Cannot connect to database"}
        
        try:
            cursor = connection.cursor()
            
            # Test table existence
            cursor.execute("SHOW TABLES")
            tables = [table[0] for table in cursor.fetchall()]
            
            debug_info = {
                "connection_status": "connected",
                "tables": tables,
                "database_config": {
                    "host": DB_CONFIG['host'],
                    "port": DB_CONFIG['port'],
                    "database": DB_CONFIG['database'],
                    "user": DB_CONFIG['user']
                }
            }
            
            # Test subscription table structure if it exists
            if 'subscription_contract_v2' in tables:
                cursor.execute("DESCRIBE subscription_contract_v2")
                sub_columns = cursor.fetchall()
                debug_info["subscription_columns"] = sub_columns
                
                cursor.execute("SELECT COUNT(*) FROM subscription_contract_v2")
                sub_count = cursor.fetchone()[0]
                debug_info["subscription_count"] = sub_count
                
                # Check for data in date range
                cursor.execute("""
                    SELECT 
                        MIN(subcription_start_date) as earliest_sub,
                        MAX(subcription_start_date) as latest_sub
                    FROM subscription_contract_v2
                """)
                date_range = cursor.fetchone()
                debug_info["subscription_date_range"] = {
                    "earliest": str(date_range[0]) if date_range[0] else None,
                    "latest": str(date_range[1]) if date_range[1] else None
                }
            
            # Test payment table structure if it exists
            if 'subscription_payment_details' in tables:
                cursor.execute("DESCRIBE subscription_payment_details")
                pay_columns = cursor.fetchall()
                debug_info["payment_columns"] = pay_columns
                
                cursor.execute("SELECT COUNT(*) FROM subscription_payment_details")
                pay_count = cursor.fetchone()[0]
                debug_info["payment_count"] = pay_count
                
                # Check for data in date range
                cursor.execute("""
                    SELECT 
                        MIN(created_date) as earliest_payment,
                        MAX(created_date) as latest_payment
                    FROM subscription_payment_details
                """)
                date_range = cursor.fetchone()
                debug_info["payment_date_range"] = {
                    "earliest": str(date_range[0]) if date_range[0] else None,
                    "latest": str(date_range[1]) if date_range[1] else None
                }
            
            return debug_info
            
        except Exception as e:
            return {"error": str(e), "error_type": type(e).__name__}
        finally:
            if connection.is_connected():
                cursor.close()
                connection.close()
    
    @app.get("/debug/test-date-ranges")
    async def debug_date_ranges(api_key: str = Depends(verify_api_key)):
        """Debug endpoint to test date range functions"""
        test_start = "2024-12-01"
        test_end = "2024-12-11"
        
        logger.info(f"🧪 Testing date range functions from {test_start} to {test_end}")
        
        # Test each function individually
        sub_result = get_subscriptions_by_date_range(test_start, test_end)
        pay_result = get_payments_by_date_range(test_start, test_end)
        analytics_result = get_analytics_by_date_range(test_start, test_end)
        
        # Also test a broader range to see if there's any data
        broad_start = "2024-01-01"
        broad_end = "2024-12-31"
        broad_sub_result = get_subscriptions_by_date_range(broad_start, broad_end)
        broad_pay_result = get_payments_by_date_range(broad_start, broad_end)
        
        return {
            "test_period": f"{test_start} to {test_end}",
            "results": {
                "subscriptions": sub_result,
                "payments": pay_result,
                "analytics": analytics_result
            },
            "broad_test_period": f"{broad_start} to {broad_end}",
            "broad_results": {
                "subscriptions": broad_sub_result,
                "payments": broad_pay_result
            }
        }
    
    @app.get("/debug/recent-data")
    async def debug_recent_data(api_key: str = Depends(verify_api_key)):
        """Check for recent data in the database"""
        connection = get_db_connection()
        if not connection:
            return {"error": "Cannot connect to database"}
        
        try:
            cursor = connection.cursor(dictionary=True)
            
            # Check recent subscriptions
            cursor.execute("""
                SELECT 
                    subcription_start_date,
                    COUNT(*) as count
                FROM subscription_contract_v2 
                WHERE subcription_start_date >= DATE_SUB(CURDATE(), INTERVAL 30 DAY)
                GROUP BY subcription_start_date
                ORDER BY subcription_start_date DESC
                LIMIT 10
            """)
            recent_subs = cursor.fetchall()
            
            # Check recent payments
            cursor.execute("""
                SELECT 
                    created_date,
                    COUNT(*) as count,
                    SUM(CASE WHEN status = 'ACTIVE' THEN 1 ELSE 0 END) as successful
                FROM subscription_payment_details 
                WHERE created_date >= DATE_SUB(CURDATE(), INTERVAL 30 DAY)
                GROUP BY created_date
                ORDER BY created_date DESC
                LIMIT 10
            """)
            recent_payments = cursor.fetchall()
            
            # Test last 7 days specifically
            cursor.execute("""
                SELECT COUNT(*) as count 
                FROM subscription_contract_v2 
                WHERE subcription_start_date >= DATE_SUB(CURDATE(), INTERVAL 7 DAY)
            """)
            last_7_days_subs = cursor.fetchone()['count']
            
            cursor.execute("""
                SELECT COUNT(*) as count 
                FROM subscription_payment_details 
                WHERE created_date >= DATE_SUB(CURDATE(), INTERVAL 7 DAY)
            """)
            last_7_days_payments = cursor.fetchone()['count']
            
            return {
                "recent_subscriptions": recent_subs,
                "recent_payments": recent_payments,
                "last_7_days": {
                    "subscriptions": last_7_days_subs,
                    "payments": last_7_days_payments
                }
            }
            
        except Exception as e:
            return {"error": str(e), "error_type": type(e).__name__}
        finally:
            if connection.is_connected():
                cursor.close()
                connection.close()

# 8. DEVELOPMENT SERVER
if __name__ == "__main__":
    # Print API keys for development
    if not os.getenv('PORT'):
        print("\n" + "="*60)
        print("🔑 API KEYS (Save these securely!):")
        for i, key in enumerate(api_key_manager.api_keys, 1):
            if key:  # Only print non-None keys
                print(f"   Key {i}: {key}")
        print("="*60)
        print("🌐 Starting development server...")
        print("📚 API Documentation: http://localhost:8000/docs")
        print("🔍 Health Check: http://localhost:8000/health")
        print("🧪 Debug DB Test: http://localhost:8000/debug/db-test")
        print("📊 Debug Date Ranges: http://localhost:8000/debug/test-date-ranges")
        print("="*60 + "\n")
    
    # Get port from environment or default to 8000
    port = int(os.getenv('PORT', 8000))
    
    uvicorn.run(
        "api_server:app",
        host="0.0.0.0",
        port=port,
        reload=not os.getenv('PORT'),  # Only reload in development
        log_level="info"
    )