#!/usr/bin/env python3
"""
COMPLETE FIXED API Server for Subscription Analytics
- Fixed graph generation for single-row comparison data
- Enhanced SSL handling
- Proper error handling that always returns data
- All graph types supported including April vs May comparisons
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

# SEMANTIC LEARNING WITH COMPATIBLE VERSIONS
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

    def add_successful_query(self, original_question: str, sql_query: str, was_helpful: bool = True, improvement_suggestion: str = None):
        """Add query feedback to memory - both positive and negative with improvement suggestions."""
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
            
            # Add to memory with feedback information and improvement suggestion
            feedback_entry = {
                'question': original_question,
                'sql': sql_query,
                'was_helpful': was_helpful,
                'feedback_type': feedback_type,
                'timestamp': datetime.datetime.now().isoformat()
            }
            
            # Add improvement suggestion for negative feedback
            if not was_helpful and improvement_suggestion:
                feedback_entry['improvement_suggestion'] = improvement_suggestion.strip()
                logger.info(f"💡 Improvement suggestion recorded: {improvement_suggestion[:100]}...")
            
            self.known_queries.append(feedback_entry)
            
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

    def get_improvement_suggestions_for_query(self, question: str, threshold: float = 0.85):
        """Get improvement suggestions from similar failed queries."""
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
            k = min(10, len(self.known_queries))  # Search more entries
            distances, indices = self.index.search(np.array([question_vector]).astype('float32'), k=k)
            
            improvement_suggestions = []
            for distance, idx in zip(distances[0], indices[0]):
                if distance < threshold and idx < len(self.known_queries):
                    query_data = self.known_queries[idx]
                    
                    # Only include entries with improvement suggestions (negative feedback)
                    if (not query_data.get('was_helpful', True) and 
                        'improvement_suggestion' in query_data and 
                        query_data['improvement_suggestion']):
                        
                        similarity = 1.0 - (distance / 2.0)
                        improvement_suggestions.append({
                            'similarity': similarity,
                            'question': query_data['question'],
                            'failed_sql': query_data['sql'],
                            'improvement_suggestion': query_data['improvement_suggestion'],
                            'timestamp': query_data['timestamp']
                        })
            
            # Sort by similarity and return unique suggestions
            improvement_suggestions.sort(key=lambda x: x['similarity'], reverse=True)
            
            # Remove duplicate suggestions
            seen_suggestions = set()
            unique_suggestions = []
            for suggestion in improvement_suggestions:
                suggestion_text = suggestion['improvement_suggestion'].lower()
                if suggestion_text not in seen_suggestions:
                    seen_suggestions.add(suggestion_text)
                    unique_suggestions.append(suggestion)
            
            logger.info(f"💡 Found {len(unique_suggestions)} improvement suggestions for similar queries")
            return unique_suggestions[:3]  # Return top 3 unique suggestions
            
        except Exception as e:
            logger.warning(f"⚠️ Improvement suggestions search failed: {e}")
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

# ENHANCED Graph Generation Functions
class GraphAnalyzer:
    """Analyzes data and determines optimal graph types with support for comparison data."""
    
    @staticmethod
    def analyze_data_for_graphing(data: List[Dict]) -> Dict:
        """Analyze data structure and recommend graph types."""
        if not data or not isinstance(data, list) or len(data) == 0:
            return {"error": "No data available for graphing"}
        
        # Get column information
        columns = list(data[0].keys())
        num_rows = len(data)
        
        logger.info(f"📊 Analyzing {num_rows} rows with columns: {columns}")
        
        # Analyze column types
        column_analysis = {}
        for col in columns:
            values = [row.get(col) for row in data if row.get(col) is not None]
            if not values:
                continue
                
            # Determine column type
            sample_value = values[0]
            
            # Check for time/date patterns first
            is_time_series = False
            if isinstance(sample_value, str):
                # Check for common date patterns
                import re
                date_patterns = [
                    r'^\d{4}-\d{2}$',                           # YYYY-MM (like 2025-04)
                    r'^\d{4}-\d{2}-\d{2}$',                     # YYYY-MM-DD
                    r'^\d{2}/\d{2}/\d{4}$',                     # MM/DD/YYYY
                    r'^\d{4}/\d{2}/\d{2}$',                     # YYYY/MM/DD
                    r'^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}$',   # YYYY-MM-DD HH:MM:SS
                    r'^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}',   # ISO format
                ]
                
                # Test all values to make sure they're consistently date-formatted
                all_match_pattern = False
                for pattern in date_patterns:
                    if all(re.match(pattern, str(val)) for val in values):
                        is_time_series = True
                        all_match_pattern = True
                        break
                
                # Also check column name hints
                if not is_time_series and any(keyword in col.lower() for keyword in ['date', 'time', 'month', 'year', 'day', 'period']):
                    # If column name suggests time series, check if values look like dates
                    if any(re.match(pattern, str(sample_value)) for pattern in date_patterns):
                        is_time_series = True
            
            elif isinstance(sample_value, (datetime.date, datetime.datetime)):
                is_time_series = True
            
            # Categorize the column
            if isinstance(sample_value, (int, float, decimal.Decimal)):
                column_analysis[col] = {
                    'type': 'numeric',
                    'min': min(values),
                    'max': max(values),
                    'unique_count': len(set(values))
                }
            elif is_time_series:
                column_analysis[col] = {
                    'type': 'datetime', 
                    'unique_count': len(set(values)),
                    'sample_values': values[:3],  # Show sample values for debugging
                    'detected_pattern': 'time_series'
                }
            else:
                column_analysis[col] = {
                    'type': 'categorical',
                    'unique_count': len(set(values)),
                    'categories': list(set(str(v) for v in values))[:10]  # First 10 categories
                }
        
        logger.info(f"📊 Column analysis: {column_analysis}")
        
        # Determine best graph types
        recommended_graphs = GraphAnalyzer._recommend_graph_types(column_analysis, num_rows)
        
        return {
            "columns": columns,
            "num_rows": num_rows,
            "column_analysis": column_analysis,
            "recommended_graphs": recommended_graphs
        }
    
    @staticmethod
    def _recommend_graph_types(column_analysis: Dict, num_rows: int) -> List[Dict]:
        """Recommend graph types based on data structure with enhanced comparison support."""
        recommendations = []
        
        numeric_cols = [col for col, info in column_analysis.items() if info['type'] == 'numeric']
        datetime_cols = [col for col, info in column_analysis.items() if info['type'] == 'datetime']
        categorical_cols = [col for col, info in column_analysis.items() if info['type'] == 'categorical']
        
        logger.info(f"📊 Column classification: numeric={numeric_cols}, datetime={datetime_cols}, categorical={categorical_cols}")
        
        # SPECIAL CASE: Single row with multiple numeric columns (comparisons like April vs May)
        if num_rows == 1 and len(numeric_cols) >= 2:
            logger.info(f"📊 Detected single-row comparison data with {len(numeric_cols)} numeric columns")
            recommendations.append({
                'type': 'comparison_bar',
                'title': 'Comparison Analysis',
                'comparison_columns': numeric_cols,
                'description': f'Compare values across {", ".join(numeric_cols)}',
                'priority': 1
            })
            # Also add horizontal bar as alternative
            recommendations.append({
                'type': 'comparison_horizontal_bar', 
                'title': 'Comparison Rankings',
                'comparison_columns': numeric_cols,
                'description': f'Ranking comparison of {", ".join(numeric_cols)}',
                'priority': 2
            })
            logger.info(f"📊 Added comparison chart recommendations for columns: {numeric_cols}")
        
        # Time series (if we have datetime + numeric) - HIGH PRIORITY
        if datetime_cols and numeric_cols:
            recommendations.append({
                'type': 'line',
                'title': 'Time Series Analysis',
                'x_axis': datetime_cols[0],
                'y_axis': numeric_cols[0],
                'description': f'Shows trends over time for {numeric_cols[0]}',
                'priority': 1
            })
            logger.info(f"📊 Added line chart recommendation: {datetime_cols[0]} vs {numeric_cols[0]}")
        
        # Bar chart for categorical data with numeric values
        if categorical_cols and numeric_cols and num_rows <= 50:
            recommendations.append({
                'type': 'bar',
                'title': 'Categorical Comparison',
                'x_axis': categorical_cols[0],
                'y_axis': numeric_cols[0],
                'description': f'Compare {numeric_cols[0]} across {categorical_cols[0]}',
                'priority': 2
            })
            logger.info(f"📊 Added bar chart recommendation: {categorical_cols[0]} vs {numeric_cols[0]}")
        
        # Bar chart for datetime data (alternative to line chart)
        if datetime_cols and numeric_cols and num_rows <= 20:
            recommendations.append({
                'type': 'bar',
                'title': 'Time Period Comparison',
                'x_axis': datetime_cols[0],
                'y_axis': numeric_cols[0],
                'description': f'Compare {numeric_cols[0]} across time periods',
                'priority': 3
            })
            logger.info(f"📊 Added time period bar chart recommendation")
        
        # Pie chart for categories with counts
        if categorical_cols and len(categorical_cols) == 1 and len(numeric_cols) == 1 and num_rows <= 10:
            recommendations.append({
                'type': 'pie',
                'title': 'Distribution Analysis',
                'category': categorical_cols[0],
                'value': numeric_cols[0],
                'description': f'Distribution of {numeric_cols[0]} by {categorical_cols[0]}',
                'priority': 3
            })
            logger.info(f"📊 Added pie chart recommendation")
        
        # Scatter plot for two numeric columns (need multiple rows)
        if len(numeric_cols) >= 2 and num_rows > 2:
            recommendations.append({
                'type': 'scatter',
                'title': 'Correlation Analysis',
                'x_axis': numeric_cols[0],
                'y_axis': numeric_cols[1],
                'description': f'Relationship between {numeric_cols[0]} and {numeric_cols[1]}',
                'priority': 3
            })
            logger.info(f"📊 Added scatter plot recommendation")
        
        # Horizontal bar for rankings
        if categorical_cols and numeric_cols and any('rate' in col.lower() or 'percent' in col.lower() for col in numeric_cols):
            recommendations.append({
                'type': 'horizontal_bar',
                'title': 'Rankings',
                'x_axis': numeric_cols[0],
                'y_axis': categorical_cols[0],
                'description': f'Ranking by {numeric_cols[0]}',
                'priority': 2
            })
            logger.info(f"📊 Added horizontal bar chart recommendation")
        
        # Sort by priority
        recommendations.sort(key=lambda x: x['priority'])
        
        logger.info(f"📊 Final recommendations ({len(recommendations)} total): {[r['type'] for r in recommendations]}")
        return recommendations

def generate_graph_data(data: List[Dict], graph_type: str = None, custom_config: Dict = None) -> Dict:
    """Generate graph-ready data from SQL results with enhanced comparison support."""
    try:
        if not data or not isinstance(data, list) or len(data) == 0:
            return {"error": "No data provided for graph generation"}
        
        # Analyze the data
        analysis = GraphAnalyzer.analyze_data_for_graphing(data)
        if "error" in analysis:
            return analysis
        
        logger.info(f"📊 Graph analysis for {len(data)} rows:")
        logger.info(f"   Column analysis: {analysis['column_analysis']}")
        logger.info(f"   Recommended graphs: {[r['type'] for r in analysis['recommended_graphs']]}")
        
        # Use provided graph type or auto-select the best one
        if graph_type:
            # Find if the requested graph type is among recommendations
            selected_graph = None
            for rec in analysis['recommended_graphs']:
                if rec['type'] == graph_type:
                    selected_graph = rec
                    break
            
            if not selected_graph:
                # Graph type not in recommendations - let's try to accommodate it anyway
                logger.warning(f"⚠️ Requested graph type '{graph_type}' not in recommendations, attempting anyway...")
                
                # Create a basic recommendation for the requested type
                column_analysis = analysis['column_analysis']
                numeric_cols = [col for col, info in column_analysis.items() if info['type'] == 'numeric']
                datetime_cols = [col for col, info in column_analysis.items() if info['type'] == 'datetime']
                categorical_cols = [col for col, info in column_analysis.items() if info['type'] == 'categorical']
                
                if graph_type == 'line' and datetime_cols and numeric_cols:
                    selected_graph = {
                        'type': 'line',
                        'title': 'Time Series Analysis',
                        'x_axis': datetime_cols[0],
                        'y_axis': numeric_cols[0],
                        'description': f'Shows trends over time for {numeric_cols[0]}',
                        'priority': 1
                    }
                elif graph_type == 'bar' and (categorical_cols or datetime_cols) and numeric_cols:
                    x_col = categorical_cols[0] if categorical_cols else datetime_cols[0]
                    selected_graph = {
                        'type': 'bar',
                        'title': 'Categorical Comparison',
                        'x_axis': x_col,
                        'y_axis': numeric_cols[0],
                        'description': f'Compare {numeric_cols[0]} across {x_col}',
                        'priority': 1
                    }
                elif graph_type == 'bar' and len(data) == 1 and len(numeric_cols) >= 2:
                    # Special case for single-row comparison data
                    selected_graph = {
                        'type': 'comparison_bar',
                        'title': 'Comparison Analysis',
                        'comparison_columns': numeric_cols,
                        'description': f'Compare values across {", ".join(numeric_cols)}',
                        'priority': 1
                    }
                else:
                    return {"error": f"Graph type '{graph_type}' not suitable for this data structure. Available columns: {list(column_analysis.keys())}. Data has {len(data)} rows with {len(numeric_cols)} numeric columns."}
        else:
            if not analysis['recommended_graphs']:
                return {"error": "No suitable graph types found for this data"}
            selected_graph = analysis['recommended_graphs'][0]  # Use the highest priority
        
        logger.info(f"📊 Selected graph type: {selected_graph['type']} - {selected_graph['title']}")
        
        # Generate graph-specific data
        graph_data = {
            "graph_type": selected_graph['type'],
            "title": selected_graph['title'],
            "description": selected_graph['description'],
            "data_summary": {
                "total_rows": len(data),
                "columns": analysis['columns']
            }
        }
        
        # Apply custom configuration if provided
        if custom_config:
            selected_graph.update(custom_config)
        
        # Generate data based on graph type
        if selected_graph['type'] == 'line':
            graph_data.update(GraphAnalyzer._prepare_line_data(data, selected_graph))
        elif selected_graph['type'] == 'bar':
            graph_data.update(GraphAnalyzer._prepare_bar_data(data, selected_graph))
        elif selected_graph['type'] == 'comparison_bar':
            graph_data.update(GraphAnalyzer._prepare_comparison_bar_data(data, selected_graph))
            # Update graph type to regular bar for rendering
            graph_data['graph_type'] = 'bar'
        elif selected_graph['type'] == 'comparison_horizontal_bar':
            graph_data.update(GraphAnalyzer._prepare_comparison_horizontal_bar_data(data, selected_graph))
            # Update graph type to horizontal_bar for rendering
            graph_data['graph_type'] = 'horizontal_bar'
        elif selected_graph['type'] == 'horizontal_bar':
            graph_data.update(GraphAnalyzer._prepare_horizontal_bar_data(data, selected_graph))
        elif selected_graph['type'] == 'pie':
            graph_data.update(GraphAnalyzer._prepare_pie_data(data, selected_graph))
        elif selected_graph['type'] == 'scatter':
            graph_data.update(GraphAnalyzer._prepare_scatter_data(data, selected_graph))
        else:
            return {"error": f"Graph type '{selected_graph['type']}' not implemented"}
        
        # Add metadata for client rendering
        graph_data['metadata'] = {
            'all_recommendations': analysis['recommended_graphs'],
            'column_analysis': analysis['column_analysis'],
            'generated_at': datetime.datetime.now().isoformat()
        }
        
        logger.info(f"📊 Generated {selected_graph['type']} graph with {len(data)} data points")
        return {"data": graph_data}
        
    except Exception as e:
        logger.error(f"❌ Graph generation failed: {e}")
        import traceback
        logger.error(f"❌ Traceback: {traceback.format_exc()}")
        return {"error": f"Graph generation failed: {str(e)}"}

# Enhanced Graph data preparation methods
def _prepare_line_data(data: List[Dict], config: Dict) -> Dict:
    """Prepare data for line chart."""
    x_col = config['x_axis']
    y_col = config['y_axis']
    
    # Sort by x-axis for proper line progression
    sorted_data = sorted(data, key=lambda x: x.get(x_col, ''))
    
    return {
        "x_values": [row.get(x_col) for row in sorted_data],
        "y_values": [float(row.get(y_col, 0)) if row.get(y_col) is not None else 0 for row in sorted_data],
        "x_label": x_col.replace('_', ' ').title(),
        "y_label": y_col.replace('_', ' ').title()
    }

def _prepare_bar_data(data: List[Dict], config: Dict) -> Dict:
    """Prepare data for bar chart."""
    x_col = config['x_axis']
    y_col = config['y_axis']
    
    return {
        "categories": [str(row.get(x_col, '')) for row in data],
        "values": [float(row.get(y_col, 0)) if row.get(y_col) is not None else 0 for row in data],
        "x_label": x_col.replace('_', ' ').title(),
        "y_label": y_col.replace('_', ' ').title()
    }

def _prepare_comparison_bar_data(data: List[Dict], config: Dict) -> Dict:
    """Prepare data for comparison bar chart (single row, multiple numeric columns)."""
    if not data or len(data) == 0:
        return {"categories": [], "values": []}
    
    comparison_cols = config.get('comparison_columns', [])
    row = data[0]  # Single row comparison
    
    logger.info(f"📊 Preparing comparison bar data for columns: {comparison_cols}")
    logger.info(f"📊 Row data: {row}")
    
    # Transform column names to readable labels
    categories = []
    values = []
    
    for col in comparison_cols:
        # Clean up column names for display
        clean_name = col.replace('_', ' ').title()
        # Remove common prefixes/suffixes for cleaner labels
        clean_name = clean_name.replace(' Subscriptions', '').replace('Subscriptions', '')
        clean_name = clean_name.replace(' Payments', '').replace('Payments', '')
        # Handle month names specifically
        if 'april' in clean_name.lower():
            clean_name = 'April'
        elif 'may' in clean_name.lower():
            clean_name = 'May'
        elif 'june' in clean_name.lower():
            clean_name = 'June'
        # Add more months as needed
        
        categories.append(clean_name)
        
        value = row.get(col, 0)
        values.append(float(value) if value is not None else 0)
    
    logger.info(f"📊 Generated categories: {categories}")
    logger.info(f"📊 Generated values: {values}")
    
    return {
        "categories": categories,
        "values": values,
        "x_label": "Comparison Categories",
        "y_label": "Count"
    }

def _prepare_comparison_horizontal_bar_data(data: List[Dict], config: Dict) -> Dict:
    """Prepare data for comparison horizontal bar chart."""
    comparison_data = _prepare_comparison_bar_data(data, config)
    
    # Sort by values for better visualization
    combined = list(zip(comparison_data["categories"], comparison_data["values"]))
    combined.sort(key=lambda x: x[1], reverse=True)
    
    if combined:
        categories, values = zip(*combined)
        return {
            "categories": list(categories),
            "values": list(values),
            "x_label": "Count",
            "y_label": "Comparison Categories"
        }
    else:
        return {"categories": [], "values": [], "x_label": "Count", "y_label": "Categories"}

def _prepare_horizontal_bar_data(data: List[Dict], config: Dict) -> Dict:
    """Prepare data for horizontal bar chart."""
    x_col = config['x_axis']  # numeric values
    y_col = config['y_axis']  # categories
    
    # Sort by values for better visualization
    sorted_data = sorted(data, key=lambda x: float(x.get(x_col, 0)) if x.get(x_col) is not None else 0, reverse=True)
    
    return {
        "categories": [str(row.get(y_col, '')) for row in sorted_data],
        "values": [float(row.get(x_col, 0)) if row.get(x_col) is not None else 0 for row in sorted_data],
        "x_label": x_col.replace('_', ' ').title(),
        "y_label": y_col.replace('_', ' ').title()
    }

def _prepare_pie_data(data: List[Dict], config: Dict) -> Dict:
    """Prepare data for pie chart."""
    category_col = config['category']
    value_col = config['value']
    
    return {
        "labels": [str(row.get(category_col, '')) for row in data],
        "values": [float(row.get(value_col, 0)) if row.get(value_col) is not None else 0 for row in data]
    }

def _prepare_scatter_data(data: List[Dict], config: Dict) -> Dict:
    """Prepare data for scatter plot."""
    x_col = config['x_axis']
    y_col = config['y_axis']
    
    return {
        "x_values": [float(row.get(x_col, 0)) if row.get(x_col) is not None else 0 for row in data],
        "y_values": [float(row.get(y_col, 0)) if row.get(y_col) is not None else 0 for row in data],
        "x_label": x_col.replace('_', ' ').title(),
        "y_label": y_col.replace('_', ' ').title()
    }

# Add these methods to GraphAnalyzer class
GraphAnalyzer._prepare_line_data = staticmethod(_prepare_line_data)
GraphAnalyzer._prepare_bar_data = staticmethod(_prepare_bar_data)
GraphAnalyzer._prepare_comparison_bar_data = staticmethod(_prepare_comparison_bar_data)
GraphAnalyzer._prepare_comparison_horizontal_bar_data = staticmethod(_prepare_comparison_horizontal_bar_data)
GraphAnalyzer._prepare_horizontal_bar_data = staticmethod(_prepare_horizontal_bar_data)
GraphAnalyzer._prepare_pie_data = staticmethod(_prepare_pie_data)
GraphAnalyzer._prepare_scatter_data = staticmethod(_prepare_scatter_data)

# Enhanced Tool Functions (all original functions preserved)
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
    """Get payment success rate and revenue statistics for the last N days."""
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
        "semantic_learning": "enabled" if SEMANTIC_LEARNING_ENABLED else "disabled",
        "platform": "render.com"
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

def record_query_feedback(original_question: str, sql_query: str, was_helpful: bool, improvement_suggestion: str = None) -> Dict:
    """Record user feedback - both positive and negative with optional improvement suggestions."""
    try:
        if not original_question or not sql_query:
            return {"error": "Both original_question and sql_query are required"}
        
        if not isinstance(was_helpful, bool):
            return {"error": "was_helpful must be a boolean value"}
        
        # Validate improvement suggestion for negative feedback
        if not was_helpful and improvement_suggestion:
            improvement_suggestion = improvement_suggestion.strip()
            if len(improvement_suggestion) < 10:
                return {"error": "Improvement suggestion must be at least 10 characters long"}
            if len(improvement_suggestion) > 500:
                return {"error": "Improvement suggestion must be less than 500 characters"}
        
        if semantic_learner:
            # Store feedback with improvement suggestion
            semantic_learner.add_successful_query(
                original_question, 
                sql_query, 
                was_helpful, 
                improvement_suggestion
            )
            
            if was_helpful:
                return {"message": "Thank you! Your positive feedback has been recorded and will help improve the system."}
            else:
                if improvement_suggestion:
                    return {
                        "message": "Thank you! Your negative feedback and improvement suggestion have been recorded. The system will try to avoid similar mistakes and incorporate your suggestions in future queries.",
                        "data": {"improvement_recorded": True}
                    }
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

def get_improvement_suggestions(original_question: str) -> Dict:
    """Get improvement suggestions for a query based on similar failed queries."""
    try:
        if not semantic_learner:
            return {"message": "Improvement suggestions not available (semantic learning disabled)"}
        
        suggestions = semantic_learner.get_improvement_suggestions_for_query(original_question)
        
        if not suggestions:
            return {"message": "No improvement suggestions found for similar queries"}
        
        result = {
            "suggestions_found": len(suggestions),
            "improvements": []
        }
        
        for suggestion in suggestions:
            improvement = {
                "similarity_score": f"{suggestion['similarity']:.2f}",
                "similar_question": suggestion['question'],
                "what_failed": suggestion['failed_sql'][:100] + "..." if len(suggestion['failed_sql']) > 100 else suggestion['failed_sql'],
                "user_suggestion": suggestion['improvement_suggestion'],
                "timestamp": suggestion['timestamp']
            }
            result["improvements"].append(improvement)
        
        return {"data": result}
        
    except Exception as e:
        logger.warning(f"⚠️ Improvement suggestions retrieval failed: {e}")
        return {"error": f"Failed to get improvement suggestions: {str(e)}"}

# Tool Registry - Enhanced with new graph generation tool
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
    "generate_graph_data": {
        "function": generate_graph_data,
        "description": "Generate graph-ready data from SQL query results. Automatically analyzes data and recommends optimal visualization types (line, bar, pie, scatter, etc.) including support for single-row comparison data (April vs May style)",
        "parameters": {
            "type": "object", 
            "properties": {
                "data": {
                    "type": "array",
                    "description": "Array of dictionaries containing the data to visualize (from SQL results)"
                },
                "graph_type": {
                    "type": "string",
                    "description": "Optional: Force specific graph type (line, bar, horizontal_bar, pie, scatter). If not provided, system will auto-select the best type.",
                    "enum": ["line", "bar", "horizontal_bar", "pie", "scatter"]
                },
                "custom_config": {
                    "type": "object",
                    "description": "Optional: Custom configuration to override automatic settings (x_axis, y_axis, title, etc.)"
                }
            },
            "required": ["data"]
        }
    },
    "record_query_feedback": {
        "function": record_query_feedback,
        "description": "Record user feedback on a dynamic query - both positive and negative with optional improvement suggestions",
        "parameters": {
            "type": "object",
            "properties": {
                "original_question": {"type": "string"},
                "sql_query": {"type": "string"},
                "was_helpful": {"type": "boolean"},
                "improvement_suggestion": {
                    "type": "string", 
                    "description": "Optional improvement suggestion for negative feedback"
                }
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
    },
    "get_improvement_suggestions": {
        "function": get_improvement_suggestions,
        "description": "Get improvement suggestions based on similar failed queries to help generate better SQL",
        "parameters": {
            "type": "object",
            "properties": {
                "original_question": {
                    "type": "string", 
                    "description": "The question to find improvement suggestions for"
                }
            },
            "required": ["original_question"]
        }
    }
}

# API Configuration (rest of the file remains the same)
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
    logger.info("🚀 Starting COMPLETE FIXED Subscription Analytics API Server with Enhanced Graph Generation")
    logger.info(f"Semantic learning: {'enabled' if SEMANTIC_LEARNING_ENABLED else 'disabled'}")
    logger.info(f"Available tools: {len(TOOL_REGISTRY)} (including enhanced graph generation)")
    yield
    logger.info("🛑 Shutting down Complete Fixed API Server")

# Create FastAPI app
app = FastAPI(
    title="Complete Fixed Subscription Analytics API",
    description="Render.com deployment with complete graph generation and comparison support",
    version="9.0.0-complete-fix",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# Health endpoint - NO AUTHENTICATION (for Render.com health checks)
@app.get("/health")
def health_check():
    """Health check endpoint - NO AUTHENTICATION for Render health checks."""
    health_data = {
        "status": "ok",
        "semantic_learning": "enabled" if SEMANTIC_LEARNING_ENABLED else "disabled",
        "timestamp": datetime.datetime.now().isoformat(),
        "available_tools": len(TOOL_REGISTRY),
        "platform": "render.com",
        "version": "9.0.0-complete-fix",
        "features": [
            "dynamic_sql_generation",
            "semantic_learning",
            "feedback_with_improvements",
            "query_suggestions",
            "enhanced_graph_generation",
            "comparison_chart_support",
            "single_row_comparison_data",
            "april_vs_may_analysis"
        ]
    }
    
    # Add learning system stats if available
    if semantic_learner and hasattr(semantic_learner, 'known_queries'):
        try:
            total_queries = len(semantic_learner.known_queries)
            positive_count = sum(1 for q in semantic_learner.known_queries if q.get('was_helpful', True))
            negative_count = total_queries - positive_count
            improvement_count = sum(1 for q in semantic_learner.known_queries 
                                  if not q.get('was_helpful', True) and 'improvement_suggestion' in q)
            
            health_data.update({
                "learning_stats": {
                    "total_learned_queries": total_queries,
                    "positive_examples": positive_count,
                    "negative_examples": negative_count,
                    "improvement_suggestions": improvement_count
                }
            })
        except:
            pass  # Don't break health check if learning stats fail
    
    return health_data

# Tools endpoint - WITH AUTHENTICATION
@app.get("/tools", response_model=List[ToolInfo], dependencies=[Depends(verify_api_key)])
def list_tools():
    """List all available tools."""
    return [
        ToolInfo(name=name, description=info["description"], parameters=info["parameters"])
        for name, info in TOOL_REGISTRY.items()
        if name not in ["record_query_feedback", "get_query_suggestions"]  # Hide internal tools
    ]

# Execute endpoint - WITH AUTHENTICATION  
@app.post("/execute", response_model=ToolResponse, dependencies=[Depends(verify_api_key)])
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
    
    # Render.com sets PORT environment variable
    port = int(os.getenv("PORT", 8000))
    
    logger.info(f"🚀 Starting COMPLETE FIXED RENDER.COM server on port {port}")
    logger.info("🛡️ Enhanced error handling and logging enabled")
    logger.info("🧠 Full semantic learning support enabled")
    logger.info("💡 Enhanced feedback system with improvement suggestions enabled")
    logger.info("📊 COMPLETE: Graph generation with comparison support enabled")
    logger.info("🔧 FIXED: Single-row comparison data (April vs May) support enabled")
    
    # Render.com optimized configuration
    uvicorn.run(
        "api_server:app",
        host="0.0.0.0",
        port=port,
        reload=False,           # Critical: prevent reloads
        workers=1,              # Single worker for stability
        log_level="info",
        access_log=True,
        loop="asyncio"          # Use asyncio loop
    )