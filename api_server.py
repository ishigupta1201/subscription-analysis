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

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# DEBUG: Print all environment variables related to database
print("🔍 === DATABASE DEBUG INFO ===")
print(f"DB_HOST: '{os.getenv('DB_HOST')}'")
print(f"DB_PORT: '{os.getenv('DB_PORT')}'")
print(f"DB_USER: '{os.getenv('DB_USER')}'")
print(f"DB_NAME: '{os.getenv('DB_NAME')}'")
print(f"DB_PASSWORD exists: {bool(os.getenv('DB_PASSWORD'))}")
print("🔍 === END DEBUG INFO ===")

# Your existing DB_CONFIG
DB_CONFIG = {
    'host': os.getenv('DB_HOST', 'localhost'),
    'port': int(os.getenv('DB_PORT', '3306')),
    'database': os.getenv('DB_NAME', 'SUBS_STAGING'),
    'user': os.getenv('DB_USER', 'root'),
    'password': os.getenv('DB_PASSWORD', '12345678'),
    'autocommit': True
}

print(f"🔍 Final DB_CONFIG port: {DB_CONFIG['port']}")

# --- Database Configuration -
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

# 2. DATABASE FUNCTIONS
def get_db_connection():
    """Create and return a database connection"""
    try:
        connection = mysql.connector.connect(**DB_CONFIG)
        if connection.is_connected():
            return connection
    except Error as e:
        logger.error(f"Error connecting to MySQL: {e}")
        return None

def get_subscriptions_in_last_days(days: int) -> Dict:
    """Get subscription data for the last x days"""
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
        
        cursor.execute(query, (start_date, today))
        result = cursor.fetchone()
        
        return {
            "new_subscriptions": result['new_subscriptions'] or 0,
            "active_subscriptions": result['active_count'] or 0,
            "cancelled_subscriptions": result['cancelled_count'] or 0,
            "period_days": days,
            "date_range": {
                "start": str(start_date),
                "end": str(today)
            }
        }
        
    except Error as e:
        logger.error(f"Database query failed: {str(e)}")
        return {"error": f"Database query failed: {str(e)}"}
    finally:
        if connection and connection.is_connected():
            cursor.close()
            connection.close()

def get_payment_success_rate_in_last_days(days: int) -> Dict:
    """Get payment success rate for the last x days"""
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
        
        cursor.execute(query, (start_date, today))
        result = cursor.fetchone()
        
        total_payments = result['total_payments'] or 0
        successful_payments = result['successful_payments'] or 0
        failed_payments = result['failed_payments'] or 0
        total_revenue = float(result['total_revenue'] or 0)
        lost_revenue = float(result['lost_revenue'] or 0)
        
        if total_payments == 0:
            return {
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
        
        success_rate = (successful_payments / total_payments) * 100
        failure_rate = (failed_payments / total_payments) * 100
        
        return {
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
        
    except Error as e:
        logger.error(f"Database query failed: {str(e)}")
        return {"error": f"Database query failed: {str(e)}"}
    finally:
        if connection and connection.is_connected():
            cursor.close()
            connection.close()

def get_subscription_summary(days: int = 30) -> Dict:
    """Get comprehensive subscription and payment summary"""
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
    """Check database connection and get basic statistics"""
    connection = get_db_connection()
    if not connection:
        return {"status": "disconnected", "error": "Cannot connect to database"}
    
    try:
        cursor = connection.cursor(dictionary=True)
        
        cursor.execute("SELECT COUNT(*) as count FROM subscription_contract_v2")
        total_subscriptions = cursor.fetchone()['count']
        
        cursor.execute("SELECT COUNT(*) as count FROM subscription_payment_details")
        total_payments = cursor.fetchone()['count']

        cursor.execute("SELECT COUNT(DISTINCT merchant_user_id) as count FROM subscription_contract_v2")
        unique_users = cursor.fetchone()['count']
        
        cursor.execute("SELECT MAX(subcription_start_date) as latest_sub FROM subscription_contract_v2")
        latest_sub = cursor.fetchone()['latest_sub']
        
        cursor.execute("SELECT MAX(created_date) as latest_payment FROM subscription_payment_details")
        latest_payment = cursor.fetchone()['latest_payment']
        
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
            "tables": ["subscription_contract_v2", "subscription_payment_details"],
            "total_subscriptions": total_subscriptions,
            "total_payments": total_payments,
            "unique_users": unique_users,
            "latest_subscription": str(latest_sub) if latest_sub else None,
            "latest_payment": str(latest_payment) if latest_payment else None,
            "overall_success_rate": f"{overall_success_rate:.2f}%"
        }
        
    except Error as e:
        logger.error(f"Database status check failed: {str(e)}")
        return {"status": "error", "error": str(e)}
    finally:
        if connection and connection.is_connected():
            cursor.close()
            connection.close()

def get_user_payment_history(merchant_user_id: str, days: int = 90) -> Dict:
    """Get payment history for a specific user"""
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
        
        cursor.execute(query, (merchant_user_id, start_date, today))
        payments = cursor.fetchall()
        
        if not payments:
            return {
                "merchant_user_id": merchant_user_id,
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
        logger.error(f"User payment history query failed: {str(e)}")
        return {"error": f"Database query failed: {str(e)}"}
    finally:
        if connection and connection.is_connected():
            cursor.close()
            connection.close()

# 3. APIKEYMANAGER CLASS
class APIKeyManager:
    def __init__(self):
        # Load environment variables from .env file (for local development)
        load_dotenv()
        
        # Get API keys from environment variables (works for both local .env and Railway)
        api_key_1 = os.getenv('API_KEY_1')
        api_key_2 = os.getenv('API_KEY_2')
        
        # If no API keys found, generate new ones (only for local development)
        if not api_key_1 or not api_key_2:
            # Check if running on Railway (Railway sets PORT automatically)
            if os.getenv('PORT'):
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
                with open('.env', 'a') as f:
                    f.write(f"\nAPI_KEY_1={api_key_1}")
                    f.write(f"\nAPI_KEY_2={api_key_2}")
                
                logger.info(f"Generated new API keys and saved to .env file")
        
        self.api_keys = {api_key_1, api_key_2}
        logger.info(f"Initialized with {len(self.api_keys)} API keys")
    
    def is_valid_key(self, api_key: str) -> bool:
        return api_key in self.api_keys
    
    def generate_new_key(self) -> str:
        new_key = 'sub_analytics_' + secrets.token_urlsafe(32)
        self.api_keys.add(new_key)
        return new_key

# 4. INITIALIZE API_KEY_MANAGER
api_key_manager = APIKeyManager()
logger.info(f"Loaded API keys: {api_key_manager.api_keys}")

# 5. VERIFY_API_KEY FUNCTION (BEFORE ROUTES)
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

# --- Tool Registry ---
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
    }
}

# 6. FASTAPI APP CREATION
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Starting Subscription Analytics API...")
    db_status = get_database_status()
    if db_status.get("status") == "connected":
        logger.info("✅ Database connection verified")
    else:
        logger.warning("⚠️ Database connection failed")
    yield
    # Shutdown
    logger.info("Shutting down Subscription Analytics API...")

app = FastAPI(
    title="Subscription Analytics API",
    description="HTTP API wrapper for subscription payment analytics tools",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure this properly for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 7. ROUTES THAT USE VERIFY_API_KEY
@app.get("/", response_model=APIStatus)
async def root():
    """API status and basic information"""
    db_status = get_database_status()
    return APIStatus(
        status="online",
        version="1.0.0",
        available_tools=len(TOOL_REGISTRY),
        database_status=db_status.get("status", "unknown")
    )

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
    """Execute a specific tool with parameters"""
    tool_name = request.tool_name
    parameters = request.parameters
    
    if tool_name not in TOOL_REGISTRY:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Tool '{tool_name}' not found. Available tools: {list(TOOL_REGISTRY.keys())}"
        )
    
    try:
        tool_function = TOOL_REGISTRY[tool_name]["function"]
        result = tool_function(**parameters)
        
        if "error" in result:
            return ToolResponse(success=False, error=result["error"])
        
        return ToolResponse(success=True, data=result)
        
    except TypeError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid parameters for tool '{tool_name}': {str(e)}"
        )
    except Exception as e:
        logger.error(f"Tool execution failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Tool execution failed: {str(e)}"
        )

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    db_status = get_database_status()
    return {
        "status": "healthy" if db_status.get("status") == "connected" else "unhealthy",
        "database": db_status.get("status", "unknown"),
        "timestamp": datetime.datetime.now().isoformat()
    }

# --- Development Server ---
if __name__ == "__main__":
    # Print API keys for development
    print("\n" + "="*50)
    print("🔑 API KEYS (Save these securely!):")
    for i, key in enumerate(api_key_manager.api_keys, 1):
        print(f"   Key {i}: {key}")
    print("="*50 + "\n")
    
    uvicorn.run(
        "api_server:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )