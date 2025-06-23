#!/usr/bin/env python3
"""
CRASH-RESISTANT API Server for Subscription Analytics
- Fixes connection reset issues
- Improved error handling for dynamic SQL
- Stable semantic learning with LOCAL MODEL
- Better validation and logging
- NO DOWNLOADS - Uses local model only
"""

import datetime
import os
import secrets
import json
import decimal
import re
import sys
import gc
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
import traceback

# Configure logging with more detail
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('api_server.log')
    ]
)
logger = logging.getLogger(__name__)
load_dotenv()

# FORCE OFFLINE MODE BEFORE ANY IMPORTS
os.environ.update({
    'HF_HUB_OFFLINE': '1',
    'TRANSFORMERS_OFFLINE': '1',
    'HF_HUB_DISABLE_PROGRESS_BARS': '1',
    'PYTORCH_ENABLE_MPS_FALLBACK': '1',
    'PYTORCH_MPS_HIGH_WATERMARK_RATIO': '0.0',
    'PYTORCH_DISABLE_MMAP': '1',
    'OMP_NUM_THREADS': '1',
    'MKL_NUM_THREADS': '1',
    'TOKENIZERS_PARALLELISM': 'false',
    'CUDA_VISIBLE_DEVICES': '',
})

# SEMANTIC LEARNING WITH LOCAL MODEL ONLY
SEMANTIC_LEARNING_ENABLED = False
try:
    import numpy as np
    import faiss
    from sentence_transformers import SentenceTransformer, models
    import torch
    LIBRARIES_INSTALLED = True
    logger.info("✅ Semantic learning libraries available")
except ImportError as e:
    LIBRARIES_INSTALLED = False
    logger.info(f"ℹ️ Semantic learning libraries not available: {e}")

LOCAL_MODEL_PATH = "./model"

class SemanticQueryLearner:
    """Ultra-stable semantic learning with LOCAL MODEL ONLY - NO DOWNLOADS."""
    
    def __init__(self, queries_file='query_memory.json', vectors_file='query_vectors.npy'):
        global SEMANTIC_LEARNING_ENABLED
        
        if not LIBRARIES_INSTALLED:
            logger.info("Semantic learning disabled - libraries not installed")
            self.model = None
            return
            
        if not os.path.exists(LOCAL_MODEL_PATH):
            logger.info(f"Semantic learning disabled - model directory '{LOCAL_MODEL_PATH}' not found")
            self.model = None
            return
        
        try:
            logger.info(f"🧠 Loading LOCAL semantic model from {LOCAL_MODEL_PATH} - NO DOWNLOADS")
            
            # Force CPU and disable all GPU/MPS
            import torch
            torch.set_default_dtype(torch.float32)
            if hasattr(torch.backends, 'mps'):
                torch.backends.mps.is_available = lambda: False
            
            # Load model components manually from LOCAL DIRECTORY ONLY
            word_embedding_model = models.Transformer(LOCAL_MODEL_PATH)
            pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())
            
            # Create model and force to CPU
            self.model = SentenceTransformer(modules=[word_embedding_model, pooling_model])
            self.model = self.model.to('cpu')
            self.model.eval()
            
            # Force all parameters to CPU and disable gradients
            for param in self.model.parameters():
                param.data = param.data.cpu()
                param.requires_grad = False
            
            SEMANTIC_LEARNING_ENABLED = True
            
            self.queries_file = queries_file
            self.vectors_file = vectors_file
            self.known_queries = []
            self.known_vectors = None
            self.index = None
            
            self._load_memory()
            logger.info("✅ Semantic Query Learner: LOCAL MODEL LOADED - NO INTERNET USED")
            
        except Exception as e:
            logger.warning(f"⚠️ Semantic learning disabled due to setup issue: {e}")
            SEMANTIC_LEARNING_ENABLED = False
            self.model = None

    def _load_memory(self):
        """Load existing query memory safely."""
        if not self.model:
            return
            
        try:
            if os.path.exists(self.queries_file) and os.path.exists(self.vectors_file):
                with open(self.queries_file, 'r') as f:
                    self.known_queries = json.load(f)
                
                self.known_vectors = np.load(self.vectors_file)
                
                if self.known_vectors is not None and self.known_vectors.shape[0] > 0:
                    dimension = self.known_vectors.shape[1]
                    self.index = faiss.IndexFlatL2(dimension)
                    vectors_cpu = self.known_vectors.astype('float32')
                    self.index.add(vectors_cpu)
                    logger.info(f"🧠 Loaded {len(self.known_queries)} queries from memory")
                    
        except Exception as e:
            logger.warning(f"⚠️ Could not load query memory: {e}")

    def _save_memory(self):
        """Save query memory safely."""
        if not self.model:
            return
            
        try:
            with open(self.queries_file, 'w') as f:
                json.dump(self.known_queries, f, indent=2)
            
            if self.known_vectors is not None:
                np.save(self.vectors_file, self.known_vectors)
                
        except Exception as e:
            logger.warning(f"⚠️ Failed to save query memory: {e}")

    def add_successful_query(self, original_question: str, sql_query: str, was_helpful: bool = True):
        """Add query feedback to memory - both positive and negative."""
        if not self.model:
            logger.info("📝 Feedback noted (semantic learning not available)")
            return
            
        try:
            feedback_type = "positive" if was_helpful else "negative"
            logger.info(f"🧠 Adding {feedback_type} feedback to memory...")
            
            # MAXIMUM SAFETY ENCODING
            import torch
            with torch.no_grad():
                # Ensure model is in eval mode and on CPU
                self.model.eval()
                
                # Multiple fallback encoding strategies
                vector = None
                encoding_strategies = [
                    # Strategy 1: Full parameter specification
                    lambda: self.model.encode([original_question], 
                                            show_progress_bar=False,
                                            convert_to_tensor=False,
                                            device='cpu',
                                            batch_size=1)[0],
                    # Strategy 2: Basic encoding
                    lambda: self.model.encode([original_question], 
                                            show_progress_bar=False,
                                            convert_to_tensor=False)[0],
                    # Strategy 3: Minimal encoding
                    lambda: self.model.encode([original_question])[0]
                ]
                
                for i, strategy in enumerate(encoding_strategies):
                    try:
                        vector = strategy()
                        logger.info(f"✅ Encoding successful with strategy {i+1}")
                        break
                    except Exception as e:
                        logger.warning(f"⚠️ Encoding strategy {i+1} failed: {e}")
                        continue
                
                if vector is None:
                    logger.warning("⚠️ All encoding strategies failed")
                    return
            
            # Ensure vector is numpy array
            if hasattr(vector, 'cpu'):
                vector = vector.cpu().numpy()
            elif hasattr(vector, 'numpy'):
                vector = vector.numpy()
            
            # Add to memory with feedback information
            self.known_queries.append({
                'question': original_question,
                'sql': sql_query,
                'was_helpful': was_helpful,
                'feedback_type': feedback_type,
                'timestamp': datetime.datetime.now().isoformat()
            })
            
            if self.known_vectors is None:
                self.known_vectors = np.array([vector])
            else:
                self.known_vectors = np.vstack([self.known_vectors, vector])
            
            # Rebuild index safely
            try:
                if self.index is not None:
                    del self.index
                    
                dimension = self.known_vectors.shape[1]
                self.index = faiss.IndexFlatL2(dimension)
                vectors_cpu = self.known_vectors.astype('float32')
                self.index.add(vectors_cpu)
            except Exception as faiss_error:
                logger.warning(f"⚠️ FAISS indexing failed: {faiss_error}")
            
            # Cleanup and save
            gc.collect()
            self._save_memory()
            logger.info(f"💾 {feedback_type.title()} feedback stored successfully")
            
        except Exception as e:
            logger.warning(f"⚠️ Feedback processing failed gracefully: {e}")
            # Never raise exceptions from feedback - system must continue

    def find_similar_queries(self, question: str, threshold: float = 0.85):
        """Find similar queries in memory, including negative examples to warn about."""
        if not self.model or not self.index or len(self.known_queries) == 0:
            return []
        
        try:
            # Encode the question
            import torch
            with torch.no_grad():
                self.model.eval()
                question_vector = self.model.encode([question], 
                                                  show_progress_bar=False, 
                                                  convert_to_tensor=False,
                                                  device='cpu')[0]
            
            # Ensure vector is numpy array
            if hasattr(question_vector, 'cpu'):
                question_vector = question_vector.cpu().numpy()
            elif hasattr(question_vector, 'numpy'):
                question_vector = question_vector.numpy()
            
            # Search for similar queries
            k = min(5, len(self.known_queries))  # Don't search for more than we have
            distances, indices = self.index.search(np.array([question_vector]).astype('float32'), k=k)
            
            results = []
            for distance, idx in zip(distances[0], indices[0]):
                if distance < threshold and idx < len(self.known_queries):
                    query_data = self.known_queries[idx]
                    similarity = 1.0 - (distance / 2.0)  # Convert distance to similarity score
                    results.append({
                        'similarity': similarity,
                        'question': query_data['question'],
                        'sql': query_data['sql'],
                        'was_helpful': query_data.get('was_helpful', True),
                        'feedback_type': query_data.get('feedback_type', 'positive'),
                        'timestamp': query_data['timestamp']
                    })
            
            # Sort by similarity
            results.sort(key=lambda x: x['similarity'], reverse=True)
            logger.info(f"🔍 Found {len(results)} similar queries for analysis")
            return results
            
        except Exception as e:
            logger.warning(f"⚠️ Similar query search failed: {e}")
            return []

# Initialize with error handling
try:
    semantic_learner = SemanticQueryLearner()
except Exception as e:
    logger.warning(f"Failed to initialize semantic learner: {e}")
    semantic_learner = None

# Database Configuration with improved error handling
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
    'use_pure': True  # Use pure Python implementation
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

# Enhanced Database Functions
def get_db_connection():
    """Get database connection with improved error handling."""
    try:
        connection = mysql.connector.connect(**DB_CONFIG)
        if connection.is_connected():
            return connection
        else:
            logger.error("Database connection failed - connection not established")
            return None
    except Error as e:
        logger.error(f"❌ MySQL Connection Error: {e}")
        return None
    except Exception as e:
        logger.error(f"❌ Unexpected database error: {e}")
        return None

def sanitize_for_json(obj):
    """Convert objects to JSON-serializable format."""
    if isinstance(obj, list):
        return [sanitize_for_json(item) for item in obj]
    elif isinstance(obj, dict):
        return {key: sanitize_for_json(value) for key, value in obj.items()}
    elif isinstance(obj, decimal.Decimal):
        return float(obj)  # Convert to float for consistency
    elif isinstance(obj, (datetime.datetime, datetime.date)):
        return obj.isoformat()
    else:
        return obj

def _execute_query(query: str, params: tuple = ()):
    """Execute database query with comprehensive error handling."""
    connection = None
    cursor = None
    
    try:
        logger.info(f"🔍 Executing query: {query[:100]}...")
        
        connection = get_db_connection()
        if not connection:
            return None, "Database connection failed"
        
        cursor = connection.cursor(dictionary=True)
        cursor.execute(query, params)
        results = cursor.fetchall()
        
        logger.info(f"✅ Query executed successfully, returned {len(results)} rows")
        return results, None
        
    except Error as e:
        error_msg = f"Database error: {str(e)}"
        logger.error(f"❌ Database query failed: {e}")
        return None, error_msg
        
    except Exception as e:
        error_msg = f"Unexpected error: {str(e)}"
        logger.error(f"❌ Unexpected error in query execution: {e}")
        return None, error_msg
        
    finally:
        try:
            if cursor:
                cursor.close()
            if connection and connection.is_connected():
                connection.close()
        except Exception as e:
            logger.warning(f"Error closing database connection: {e}")

def validate_sql_query(sql_query: str) -> tuple[bool, str]:
    """Enhanced SQL query validation with detailed error messages."""
    if not sql_query or not isinstance(sql_query, str):
        return False, "SQL query is required and must be a string"
    
    # Clean query for analysis
    cleaned_query = re.sub(r'--.*$', '', sql_query, flags=re.MULTILINE)
    cleaned_query = re.sub(r'/\*.*?\*/', '', cleaned_query, flags=re.DOTALL)
    cleaned_query = cleaned_query.strip()
    
    if not cleaned_query:
        return False, "Query is empty after removing comments"
    
    # Check if it starts with SELECT (case insensitive)
    if not cleaned_query.upper().startswith('SELECT'):
        return False, "Only SELECT statements are allowed"
    
    # Check for prohibited keywords
    prohibited_patterns = [
        (r'\bDROP\b', 'DROP statements are not allowed'),
        (r'\bDELETE\b', 'DELETE statements are not allowed'),
        (r'\bINSERT\b', 'INSERT statements are not allowed'),
        (r'\bUPDATE\b', 'UPDATE statements are not allowed'),
        (r'\bCREATE\b', 'CREATE statements are not allowed'),
        (r'\bALTER\b', 'ALTER statements are not allowed'),
        (r'\bTRUNCATE\b', 'TRUNCATE statements are not allowed'),
        (r'\bGRANT\b', 'GRANT statements are not allowed'),
        (r'\bREVOKE\b', 'REVOKE statements are not allowed'),
        (r'\bEXEC\b', 'EXEC statements are not allowed'),
        (r'\bEXECUTE\b', 'EXECUTE statements are not allowed')
    ]
    
    cleaned_upper = cleaned_query.upper()
    for pattern, error_msg in prohibited_patterns:
        if re.search(pattern, cleaned_upper):
            return False, error_msg
    
    # Length check
    if len(sql_query) > 5000:
        return False, "SQL query is too long (maximum 5000 characters)"
    
    return True, "Query validation passed"

# Enhanced Tool Functions
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
            ) as success_rate_percent,
            GROUP_CONCAT(DISTINCT status) as all_status_values
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
        "semantic_learning": "enabled" if SEMANTIC_LEARNING_ENABLED else "disabled"
    }
    
    if results:
        status_data.update(sanitize_for_json(results[0]))
    
    return {"data": status_data}

def execute_dynamic_sql(sql_query: str) -> Dict:
    """Execute dynamic SQL query with comprehensive safety checks."""
    logger.info(f"🔍 Dynamic SQL request received: {sql_query[:100]}...")
    
    try:
        # Input validation
        if not sql_query or not isinstance(sql_query, str):
            return {"error": "SQL query is required and must be a string"}
        
        # Enhanced validation
        is_valid, validation_message = validate_sql_query(sql_query)
        if not is_valid:
            logger.warning(f"❌ SQL validation failed: {validation_message}")
            return {"error": f"SQL validation failed: {validation_message}"}
        
        logger.info(f"✅ SQL validation passed")
        logger.info(f"🔍 Executing: {sql_query}")
        
        # Execute the query
        results, error = _execute_query(sql_query)
        
        if error:
            logger.error(f"❌ SQL execution failed: {error}")
            return {"error": f"SQL execution failed: {error}"}
        
        if not results:
            logger.info("ℹ️ Query executed successfully but returned no results")
            return {"message": "Query executed successfully, but no matching records were found."}
        
        # Limit result size for performance
        if len(results) > 1000:
            results = results[:1000]
            logger.info(f"⚠️ Results limited to first 1000 rows")
            return {
                "data": sanitize_for_json(results),
                "message": f"Results limited to first 1000 rows (query returned more)"
            }
        
        logger.info(f"✅ Dynamic SQL completed successfully with {len(results)} rows")
        return {"data": sanitize_for_json(results)}
        
    except Exception as e:
        logger.error(f"❌ Unexpected error in dynamic SQL execution: {e}")
        logger.error(f"❌ Traceback: {traceback.format_exc()}")
        return {"error": f"Unexpected error during SQL execution: {str(e)}"}

def record_query_feedback(original_question: str, sql_query: str, was_helpful: bool) -> Dict:
    """Record user feedback - both positive and negative for learning."""
    try:
        if not original_question or not sql_query:
            return {"error": "Both original_question and sql_query are required"}
        
        if not isinstance(was_helpful, bool):
            return {"error": "was_helpful must be a boolean value"}
        
        if semantic_learner:
            # Store both positive and negative feedback
            semantic_learner.add_successful_query(original_question, sql_query, was_helpful)
            
            if was_helpful:
                return {"message": "Thank you! Your positive feedback has been recorded and will help improve the system."}
            else:
                return {"message": "Thank you! Your negative feedback has been recorded and will help the system avoid similar mistakes in the future."}
        else:
            return {"message": "Thank you for your feedback."}
            
    except Exception as e:
        logger.warning(f"⚠️ Feedback processing failed: {e}")
        return {"message": "Thank you for your feedback (recorded without semantic learning)."}

def get_query_suggestions(original_question: str) -> Dict:
    """Get suggestions based on similar queries in memory - helps with negative feedback learning."""
    try:
        if not semantic_learner:
            return {"message": "Query suggestions not available (semantic learning disabled)"}
        
        similar_queries = semantic_learner.find_similar_queries(original_question)
        
        if not similar_queries:
            return {"message": "No similar queries found in memory"}
        
        suggestions = {
            "similar_queries_found": len(similar_queries),
            "recommendations": []
        }
        
        for query in similar_queries[:3]:  # Top 3 most similar
            suggestion = {
                "similarity_score": f"{query['similarity']:.2f}",
                "previous_question": query['question'],
                "was_helpful": query['was_helpful'],
                "feedback_type": query['feedback_type']
            }
            
            if query['was_helpful']:
                suggestion["recommendation"] = "✅ This SQL worked well for a similar question"
                suggestion["sql"] = query['sql']
            else:
                suggestion["recommendation"] = "❌ This SQL failed for a similar question - avoid this approach"
                suggestion["failed_sql"] = query['sql']
            
            suggestions["recommendations"].append(suggestion)
        
        return {"data": suggestions}
        
    except Exception as e:
        logger.warning(f"⚠️ Query suggestions failed: {e}")
        return {"error": f"Query suggestions failed: {str(e)}"}

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
    },
    "record_query_feedback": {
        "function": record_query_feedback,
        "description": "Record user feedback on a dynamic query - both positive and negative",
        "parameters": {
            "type": "object",
            "properties": {
                "original_question": {"type": "string"},
                "sql_query": {"type": "string"},
                "was_helpful": {"type": "boolean"}
            },
            "required": ["original_question", "sql_query", "was_helpful"]
        }
    },
    "get_query_suggestions": {
        "function": get_query_suggestions,
        "description": "Get suggestions based on similar queries in memory to help avoid past mistakes",
        "parameters": {
            "type": "object",
            "properties": {
                "original_question": {"type": "string", "description": "The question to find similar queries for"}
            },
            "required": ["original_question"]
        }
    }
}

# API Configuration
API_KEY = os.getenv("API_KEY_1")
if not API_KEY:
    logger.error("❌ FATAL: API_KEY_1 environment variable is not set")
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
    logger.info("🚀 Starting CRASH-RESISTANT Subscription Analytics API Server")
    logger.info(f"Semantic learning: {'enabled' if SEMANTIC_LEARNING_ENABLED else 'disabled'}")
    logger.info(f"Available tools: {len(TOOL_REGISTRY)}")
    yield
    logger.info("🛑 Shutting down API Server")

# Create FastAPI app
app = FastAPI(
    title="Subscription Analytics API",
    description="Crash-resistant AI-powered subscription analytics",
    version="6.0.0-crash-resistant-local-model",
    lifespan=lifespan,
    dependencies=[Depends(verify_api_key)]
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# API Endpoints
@app.get("/health")
def health_check():
    """Health check endpoint."""
    health_data = {
        "status": "ok",
        "semantic_learning": "enabled" if SEMANTIC_LEARNING_ENABLED else "disabled",
        "timestamp": datetime.datetime.now().isoformat(),
        "available_tools": len(TOOL_REGISTRY),
        "stability": "crash-resistant-local-model-negative-feedback"
    }
    
    # Add learning system stats
    if semantic_learner and hasattr(semantic_learner, 'known_queries'):
        total_queries = len(semantic_learner.known_queries)
        positive_count = sum(1 for q in semantic_learner.known_queries if q.get('was_helpful', True))
        negative_count = total_queries - positive_count
        
        health_data.update({
            "learning_stats": {
                "total_learned_queries": total_queries,
                "positive_examples": positive_count,
                "negative_examples": negative_count
            }
        })
    
    return health_data

@app.get("/tools", response_model=List[ToolInfo])
def list_tools():
    """List all available tools."""
    return [
        ToolInfo(name=name, description=info["description"], parameters=info["parameters"])
        for name, info in TOOL_REGISTRY.items()
        if name not in ["record_query_feedback", "get_query_suggestions"]  # Hide internal tools
    ]

@app.post("/execute", response_model=ToolResponse)
def execute_tool(request: ToolRequest):
    """Execute a specific tool with comprehensive error handling."""
    start_time = datetime.datetime.now()
    
    try:
        logger.info(f"🔧 Tool execution request: {request.tool_name}")
        
        if request.tool_name not in TOOL_REGISTRY:
            raise HTTPException(
                status_code=404,
                detail=f"Tool '{request.tool_name}' not found"
            )
        
        tool_info = TOOL_REGISTRY[request.tool_name]
        
        # Execute the tool with comprehensive error handling
        try:
            logger.info(f"⚙️ Executing {request.tool_name} with parameters: {request.parameters}")
            result = tool_info["function"](**request.parameters)
        except TypeError as e:
            logger.error(f"❌ Parameter error for {request.tool_name}: {e}")
            raise HTTPException(
                status_code=400,
                detail=f"Invalid parameters for tool '{request.tool_name}': {str(e)}"
            )
        except Exception as e:
            logger.error(f"❌ Tool execution error for {request.tool_name}: {e}")
            logger.error(f"❌ Traceback: {traceback.format_exc()}")
            return ToolResponse(
                success=False,
                error=f"Tool execution failed: {str(e)}"
            )
        
        # Process the result
        execution_time = (datetime.datetime.now() - start_time).total_seconds()
        
        if "error" in result:
            logger.warning(f"Tool {request.tool_name} returned error: {result['error']}")
            return ToolResponse(success=False, error=result["error"])
        
        logger.info(f"✅ Tool {request.tool_name} completed successfully in {execution_time:.2f}s")
        return ToolResponse(
            success=True,
            data=result.get("data"),
            message=result.get("message")
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Unexpected error in execute_tool: {e}")
        logger.error(f"❌ Traceback: {traceback.format_exc()}")
        return ToolResponse(
            success=False,
            error=f"Server error: {str(e)}"
        )
    finally:
        # Force garbage collection after each request
        gc.collect()

if __name__ == "__main__":
    # Validate environment variables
    required_env_vars = ['DB_HOST', 'DB_NAME', 'DB_USER', 'DB_PASSWORD', 'API_KEY_1']
    missing_vars = [var for var in required_env_vars if not os.getenv(var)]
    
    if missing_vars:
        logger.error(f"❌ FATAL: Missing required environment variables: {missing_vars}")
        sys.exit(1)
    
    # Cloud Run sets PORT environment variable
    port = int(os.getenv("PORT", 8000))
    
    logger.info(f"🚀 Starting server on port {port}")
    logger.info("🛡️ Enhanced error handling and logging enabled")
    logger.info("🧠 Semantic learning enabled for Cloud Run")
    
    # Optimized configuration for Cloud Run
    uvicorn.run(
        "api_server:app",
        host="0.0.0.0",
        port=port,
        reload=False,
        workers=1,
        log_level="info",
        access_log=True,
        loop="asyncio"
    )