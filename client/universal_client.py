# client/universal_client.py - COMPLETE FIXED VERSION WITH ALL FUNCTIONALITY AND MULTITOOL SUPPORT

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
from datetime import datetime, timedelta
import argparse

# Graph visualization imports
try:
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    from matplotlib.figure import Figure
    import numpy as np
    MATPLOTLIB_AVAILABLE = True
    logger = logging.getLogger(__name__)
    logger.info("✅ Matplotlib available for graph generation")
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning("⚠️ Matplotlib not available - graphs will be disabled")

from pathlib import Path
current_dir = Path(__file__).parent.absolute()
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

# Try to import config manager with multiple fallback paths
try:
    from config_manager import ConfigManager
except ImportError:
    try:
        # Try importing from parent directory
        sys.path.insert(0, str(current_dir.parent))
        from config_manager import ConfigManager
    except ImportError:
        # Create a minimal config manager if not found
        class ConfigManager:
            def __init__(self):
                self.config_path = Path.cwd() / 'config.json'
            
            def get_config(self):
                if self.config_path.exists():
                    with open(self.config_path, 'r') as f:
                        return json.load(f)
                return {}

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

RESET = "\033[0m"
BOLD = "\033[1m"
CYAN = "\033[36m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
RED = "\033[31m"
MAGENTA = "\033[35m"
BLUE = "\033[34m"
GREY = "\033[90m"

def color_text(text, color):
    return f"{color}{text}{RESET}"

def print_header(text):
    print(f"{BOLD}{CYAN}\n{'='*len(text)}\n{text}\n{'='*len(text)}{RESET}")

def print_section(text):
    print(f"{BOLD}{BLUE}\n{text}{RESET}")

def print_separator():
    print(f"{GREY}{'-'*60}{RESET}")

def print_success(text):
    print(f"{GREEN}{text}{RESET}")

def print_warning(text):
    print(f"{YELLOW}{text}{RESET}")

def print_error(text):
    print(f"{RED}{text}{RESET}")

def print_feedback_prompt(text):
    print(f"{MAGENTA}{text}{RESET}")

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
    graph_data: Optional[Dict] = None
    graph_generated: bool = False

class CompleteGraphGenerator:
    """COMPLETE graph generator with full smart data handling and production-ready features."""
    
    def __init__(self):
        self.graphs_dir = self._setup_graphs_directory()
        self.supported_types = ['line', 'bar', 'horizontal_bar', 'pie', 'scatter']
        
    def _setup_graphs_directory(self) -> Path:
        """Setup graphs directory with smart fallbacks."""
        possible_dirs = [
            Path.cwd() / "generated_graphs",
            Path(__file__).parent / "generated_graphs",
            Path.home() / "subscription_graphs",
            Path.cwd()  # Final fallback
        ]
        
        for directory in possible_dirs:
            try:
                directory.mkdir(parents=True, exist_ok=True)
                # Test write permissions
                test_file = directory / ".write_test"
                test_file.write_text("test")
                test_file.unlink()
                logger.info(f"📊 Graph directory set to: {directory}")
                return directory
            except Exception as e:
                logger.debug(f"Cannot use directory {directory}: {e}")
                continue
        
        raise RuntimeError("No writable directory found for graph generation")
    
    def can_generate_graphs(self) -> bool:
        return MATPLOTLIB_AVAILABLE
    
    def generate_graph(self, graph_data: Dict, query: str) -> Optional[str]:
        """Generate graph with complete enhanced error handling and smart type detection."""
        if not self.can_generate_graphs():
            logger.warning("Cannot generate graph - matplotlib not available")
            return None
        
        try:
            # Validate and prepare data
            if not self._validate_graph_data(graph_data):
                logger.error("Invalid graph data structure")
                return None
            
            # Smart graph type selection
            graph_type = graph_data.get('graph_type', '').lower()
            query_lower = query.lower()
            
            # If user requested bar or line chart, always use that type
            if 'bar chart' in query_lower or graph_type == 'bar':
                graph_type = 'bar'
            elif 'line chart' in query_lower or graph_type == 'line':
                graph_type = 'line'
            # If data has two columns and one is a date, use date as x and value as y for bar/line
            columns = list(graph_data.get('columns', []))
            data_rows = graph_data.get('rows', [])
            if (graph_type in ['bar', 'line']) and columns and data_rows and len(columns) == 2:
                date_col = None
                value_col = None
                for col in columns:
                    if 'date' in col.lower() or 'time' in col.lower():
                        date_col = col
                    else:
                        value_col = col
                if date_col and value_col:
                    x_values = [row[columns.index(date_col)] for row in data_rows]
                    y_values = [row[columns.index(value_col)] for row in data_rows]
                    graph_data['x_values'] = x_values
                    graph_data['y_values'] = y_values
            # Never use pie chart for time series data
            if graph_type == 'pie' and columns and any('date' in col.lower() or 'time' in col.lower() for col in columns):
                logger.warning("Pie chart requested for time series data; switching to bar chart.")
                graph_type = 'bar'
                # Map data as above
                date_col = None
                value_col = None
                for col in columns:
                    if 'date' in col.lower() or 'time' in col.lower():
                        date_col = col
                    else:
                        value_col = col
                if date_col and value_col:
                    x_values = [row[columns.index(date_col)] for row in data_rows]
                    y_values = [row[columns.index(value_col)] for row in data_rows]
                    graph_data['x_values'] = x_values
                    graph_data['y_values'] = y_values
            # Set up matplotlib with error handling
            plt.style.use('default')
            fig, ax = plt.subplots(figsize=(12, 8))
            
            # Generate graph based on type
            success = self._create_graph_by_type(ax, graph_data, graph_type)
            
            if not success:
                logger.error(f"Failed to create {graph_type} chart")
                plt.close(fig)
                print(f"❌ Could not generate a {graph_type} chart for this data. Try a different chart type or aggregation.")
                print(f"SQL: {graph_data.get('sql_query', 'N/A')}")
                print(f"Data: {graph_data.get('rows', [])}")
                return None
            
            # Enhance graph appearance
            self._enhance_graph_appearance(fig, ax, graph_data, graph_type)
            
            # Save with smart naming
            filepath = self._save_graph_safely(fig, graph_type)
            plt.close(fig)
            
            if filepath and filepath.exists():
                self._auto_open_graph(str(filepath))
                logger.info(f"✅ Graph generated successfully: {graph_type}")
                return str(filepath)
            
            return None
            
        except Exception as e:
            logger.error(f"Graph generation failed: {e}")
            plt.close('all')
            return None
    
    def _determine_optimal_graph_type(self, graph_data: Dict, query: str) -> str:
        """Smart graph type determination based on data and query."""
        requested_type = graph_data.get('graph_type', '').lower()
        query_lower = query.lower()
        
        # Check for explicit requests in query
        if any(word in query_lower for word in ['pie chart', 'pie', 'distribution', 'breakdown']):
            return 'pie'
        elif any(word in query_lower for word in ['line chart', 'line', 'trend', 'over time', 'timeline']):
            return 'line'
        elif any(word in query_lower for word in ['scatter', 'correlation', 'relationship']):
            return 'scatter'
        elif any(word in query_lower for word in ['horizontal', 'h-bar']):
            return 'horizontal_bar'
        
        # Use requested type if valid
        if requested_type in self.supported_types:
            return requested_type
        
        # Smart defaults based on data characteristics
        if 'labels' in graph_data and 'values' in graph_data:
            data_count = len(graph_data.get('values', []))
            if data_count <= 10:
                return 'pie'
            elif data_count <= 20:
                return 'bar'
            else:
                return 'horizontal_bar'
        elif 'x_values' in graph_data and 'y_values' in graph_data:
            return 'line'
        
        return 'bar'  # Default fallback
    
    def _validate_graph_data(self, graph_data: Dict) -> bool:
        """Complete enhanced data validation."""
        try:
            # Check for required data structures
            has_xy_data = 'x_values' in graph_data and 'y_values' in graph_data
            has_categorical_data = 'labels' in graph_data and 'values' in graph_data
            has_category_value_data = 'categories' in graph_data and 'values' in graph_data
            
            if not (has_xy_data or has_categorical_data or has_category_value_data):
                return False
            
            # Validate data consistency
            if has_xy_data:
                x_vals = graph_data['x_values']
                y_vals = graph_data['y_values']
                return len(x_vals) > 0 and len(y_vals) > 0 and len(x_vals) == len(y_vals)
            
            if has_categorical_data:
                labels = graph_data['labels']
                values = graph_data['values']
                return len(labels) > 0 and len(values) > 0 and len(labels) == len(values)
            
            if has_category_value_data:
                categories = graph_data['categories']
                values = graph_data['values']
                return len(categories) > 0 and len(values) > 0 and len(categories) == len(values)
            
            return False
        except Exception:
            return False
    
    def _create_graph_by_type(self, ax, graph_data: Dict, graph_type: str) -> bool:
        """Create graph based on type with complete enhanced error handling."""
        try:
            if graph_type == 'pie':
                return self._create_complete_pie_chart(ax, graph_data)
            elif graph_type == 'bar':
                return self._create_complete_bar_chart(ax, graph_data)
            elif graph_type == 'horizontal_bar':
                return self._create_complete_horizontal_bar_chart(ax, graph_data)
            elif graph_type == 'line':
                return self._create_complete_line_chart(ax, graph_data)
            elif graph_type == 'scatter':
                return self._create_complete_scatter_plot(ax, graph_data)
            else:
                logger.warning(f"Unknown graph type: {graph_type}")
                return False
        except Exception as e:
            logger.error(f"Error creating {graph_type} chart: {e}")
            return False
    
    def _create_complete_pie_chart(self, ax, graph_data: Dict) -> bool:
        """Create complete enhanced pie chart with smart data handling."""
        try:
            labels = graph_data.get('labels', graph_data.get('categories', []))
            values = graph_data.get('values', [])
            
            if not labels or not values or len(labels) != len(values):
                return False
            
            # Filter positive values and prepare data
            filtered_data = [(label, float(value)) for label, value in zip(labels, values) 
                           if isinstance(value, (int, float)) and float(value) > 0]
            
            if not filtered_data:
                ax.text(0.5, 0.5, 'No positive data to display', 
                       ha='center', va='center', transform=ax.transAxes, fontsize=14)
                return True
            
            # Sort by value for better visualization
            filtered_data.sort(key=lambda x: x[1], reverse=True)
            
            # Limit categories if too many
            if len(filtered_data) > 8:
                top_data = filtered_data[:7]
                others_sum = sum(item[1] for item in filtered_data[7:])
                if others_sum > 0:
                    top_data.append(('Others', others_sum))
                filtered_data = top_data
            
            labels, values = zip(*filtered_data)
            
            # Create pie chart with complete enhanced styling
            colors = plt.cm.Set3(np.linspace(0, 1, len(labels)))
            wedges, texts, autotexts = ax.pie(values, labels=labels, autopct='%1.1f%%', 
                                            startangle=90, colors=colors)
            
            # Complete enhance text appearance
            for autotext in autotexts:
                autotext.set_color('white')
                autotext.set_fontweight('bold')
                autotext.set_fontsize(10)
            
            for text in texts:
                text.set_fontsize(9)
            
            ax.set_aspect('equal')
            return True
            
        except Exception as e:
            logger.error(f"Complete pie chart creation failed: {e}")
            return False
    
    def _create_complete_bar_chart(self, ax, graph_data: Dict) -> bool:
        """Create complete enhanced bar chart with smart formatting and fallback."""
        try:
            # Prefer x_values/y_values for bar chart if present
            if 'x_values' in graph_data and 'y_values' in graph_data:
                x_values = graph_data['x_values']
                y_values = graph_data['y_values']
                if not x_values or not y_values or len(x_values) != len(y_values):
                    return False
                bars = ax.bar(x_values, y_values, color='steelblue', alpha=0.8, edgecolor='darkblue')
                ax.set_xlabel(graph_data.get('x_label', 'Categories'), fontsize=12)
                ax.set_ylabel(graph_data.get('y_label', 'Values'), fontsize=12)
                if len(x_values) > 10 or any(len(str(x)) > 8 for x in x_values):
                    ax.set_xticklabels([str(x)[:15] for x in x_values], rotation=45, ha='right')
                else:
                    ax.set_xticklabels([str(x)[:15] for x in x_values])
                if len(x_values) <= 15:
                    max_val = max(y_values) if y_values else 1
                    for i, (bar, value) in enumerate(zip(bars, y_values)):
                        height = bar.get_height()
                        ax.text(bar.get_x() + bar.get_width()/2., height + max_val * 0.01,
                               f'{value:.1f}' if isinstance(value, float) else str(value),
                               ha='center', va='bottom', fontsize=8)
                ax.grid(True, alpha=0.3, axis='y')
                return True
            # Fallback to categories/values
            categories = graph_data.get('categories', graph_data.get('labels', []))
            values = graph_data.get('values', [])
            if (not categories or not values or len(categories) != len(values)) and 'x_values' in graph_data and 'y_values' in graph_data:
                categories = graph_data['x_values']
                values = graph_data['y_values']
            if not categories or not values or len(categories) != len(values):
                return False
            if len(categories) > 30:
                categories = categories[:30]
                values = values[:30]
                logger.info("Limited bar chart to 30 categories for readability")
            bars = ax.bar(range(len(categories)), values, color='steelblue', alpha=0.8, edgecolor='darkblue')
            ax.set_xlabel(graph_data.get('x_label', 'Categories'), fontsize=12)
            ax.set_ylabel(graph_data.get('y_label', 'Values'), fontsize=12)
            ax.set_xticks(range(len(categories)))
            ax.set_xticklabels([str(cat)[:15] for cat in categories], rotation=45, ha='right')
            if len(categories) <= 15:
                max_val = max(values) if values else 1
                for i, (bar, value) in enumerate(zip(bars, values)):
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height + max_val * 0.01,
                           f'{value:.1f}' if isinstance(value, float) else str(value),
                           ha='center', va='bottom', fontsize=8)
            ax.grid(True, alpha=0.3, axis='y')
            return True
        except Exception as e:
            logger.error(f"Complete bar chart creation failed: {e}")
            return False
    
    def _create_complete_horizontal_bar_chart(self, ax, graph_data: Dict) -> bool:
        """Create complete enhanced horizontal bar chart."""
        try:
            categories = graph_data.get('categories', graph_data.get('labels', []))
            values = graph_data.get('values', [])
            
            if not categories or not values or len(categories) != len(values):
                return False
            
            # Limit and sort data
            if len(categories) > 20:
                # Sort by value and take top 20
                data_pairs = list(zip(categories, values))
                data_pairs.sort(key=lambda x: float(x[1]) if isinstance(x[1], (int, float)) else 0, reverse=True)
                categories, values = zip(*data_pairs[:20])
                logger.info("Limited horizontal bar chart to top 20 categories")
            
            # Create horizontal bar chart
            bars = ax.barh(range(len(categories)), values, color='lightcoral', alpha=0.8, edgecolor='darkred')
            
            # Set labels and formatting
            ax.set_xlabel(graph_data.get('x_label', 'Values'), fontsize=12)
            ax.set_ylabel(graph_data.get('y_label', 'Categories'), fontsize=12)
            ax.set_yticks(range(len(categories)))
            ax.set_yticklabels([str(cat)[:20] for cat in categories])
            
            # Add value labels
            if len(categories) <= 15:
                max_val = max(values) if values else 1
                for i, (bar, value) in enumerate(zip(bars, values)):
                    width = bar.get_width()
                    ax.text(width + max_val * 0.01, bar.get_y() + bar.get_height()/2,
                           f'{value:.1f}' if isinstance(value, float) else str(value),
                           ha='left', va='center', fontsize=8)
            
            ax.grid(True, alpha=0.3, axis='x')
            return True
            
        except Exception as e:
            logger.error(f"Complete horizontal bar chart creation failed: {e}")
            return False
    
    def _create_complete_line_chart(self, ax, graph_data: Dict) -> bool:
        """Create complete enhanced line chart with smart date handling."""
        try:
            x_values = graph_data.get('x_values', [])
            y_values = graph_data.get('y_values', [])
            
            if not x_values or not y_values or len(x_values) != len(y_values):
                return False
            
            # Handle large datasets by sampling
            if len(x_values) > 100:
                step = max(1, len(x_values) // 50)
                x_values = x_values[::step]
                y_values = y_values[::step]
                logger.info(f"Sampled line chart data to {len(x_values)} points")
            
            # Create line chart with complete enhanced styling
            ax.plot(x_values, y_values, marker='o', linewidth=2, markersize=4, 
                   color='darkgreen', markerfacecolor='lightgreen')
            
            # Set labels and formatting
            ax.set_xlabel(graph_data.get('x_label', 'X Axis'), fontsize=12)
            ax.set_ylabel(graph_data.get('y_label', 'Y Axis'), fontsize=12)
            
            # Smart x-axis formatting for dates
            if len(x_values) > 10:
                ax.tick_params(axis='x', rotation=45)
            
            ax.grid(True, alpha=0.3)
            return True
            
        except Exception as e:
            logger.error(f"Complete line chart creation failed: {e}")
            return False
    
    def _create_complete_scatter_plot(self, ax, graph_data: Dict) -> bool:
        """Create complete enhanced scatter plot."""
        try:
            x_values = graph_data.get('x_values', [])
            y_values = graph_data.get('y_values', [])
            
            if not x_values or not y_values or len(x_values) != len(y_values):
                return False
            
            # Convert to numeric if possible
            try:
                x_numeric = [float(x) for x in x_values]
                y_numeric = [float(y) for y in y_values]
            except (ValueError, TypeError):
                return False
            
            # Handle large datasets
            if len(x_numeric) > 1000:
                step = max(1, len(x_numeric) // 500)
                x_numeric = x_numeric[::step]
                y_numeric = y_numeric[::step]
                logger.info(f"Sampled scatter plot to {len(x_numeric)} points")
            
            # Create scatter plot
            ax.scatter(x_numeric, y_numeric, alpha=0.6, s=50, color='darkblue', edgecolors='lightblue')
            
            # Set labels and formatting
            ax.set_xlabel(graph_data.get('x_label', 'X Axis'), fontsize=12)
            ax.set_ylabel(graph_data.get('y_label', 'Y Axis'), fontsize=12)
            ax.grid(True, alpha=0.3)
            
            return True
            
        except Exception as e:
            logger.error(f"Complete scatter plot creation failed: {e}")
            return False
    
    def _enhance_graph_appearance(self, fig, ax, graph_data: Dict, graph_type: str):
        """Complete enhance overall graph appearance."""
        try:
            title = graph_data.get('title', 'Data Visualization')
            description = graph_data.get('description', '')
            
            if description:
                full_title = f"{title}\n{description}"
                fig.suptitle(full_title, fontsize=14, fontweight='bold', y=0.95)
            else:
                ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
            
            # Adjust layout based on graph type
            if graph_type == 'pie':
                plt.tight_layout()
            else:
                plt.tight_layout()
                plt.subplots_adjust(bottom=0.15)
            
        except Exception as e:
            logger.warning(f"Complete graph enhancement failed: {e}")
    
    def _save_graph_safely(self, fig, graph_type: str) -> Optional[Path]:
        """Save graph with complete smart error handling."""
        try:
            timestamp = int(datetime.now().timestamp())
            filename = f"graph_{graph_type}_{timestamp}.png"
            filepath = self.graphs_dir / filename
            
            # Try primary save location
            try:
                fig.savefig(filepath, dpi=300, bbox_inches='tight', 
                          facecolor='white', edgecolor='none')
                return filepath
            except Exception as save_error:
                # Try fallback location
                fallback_path = Path.cwd() / filename
                fig.savefig(fallback_path, dpi=300, bbox_inches='tight', 
                          facecolor='white', edgecolor='none')
                logger.info(f"Saved to fallback location: {fallback_path}")
                return fallback_path
                
        except Exception as e:
            logger.error(f"Complete graph saving failed: {e}")
            return None
    
    def _auto_open_graph(self, filepath: str) -> bool:
        """Auto-open graph with cross-platform support."""
        try:
            import subprocess
            import os
            
            if os.name == 'nt':  # Windows
                os.startfile(filepath)
            elif sys.platform == 'darwin':  # macOS
                subprocess.run(['open', filepath], check=True, timeout=5)
            else:  # Linux
                subprocess.run(['xdg-open', filepath], check=True, timeout=5)
            
            logger.info(f"📊 Graph opened: {filepath}")
            return True
        except Exception:
            return False

class CompleteSmartNLPProcessor:
    """COMPLETE NLP processor with enhanced threshold detection and better prompting. FIXED MULTITOOL SUPPORT."""
    
    def __init__(self):
        self.model = genai.GenerativeModel('gemini-1.5-flash-latest')
        self.db_schema = self._get_complete_database_schema()
        self.chart_keywords = self._get_chart_keywords()
        self.tools = self._get_tools_config()

    def _get_complete_database_schema(self) -> str:
        """Get COMPLETE CORRECTED database schema with full column information."""
        return """
COMPLETE CORRECTED Database Schema:

Tables:
1. subscription_contract_v2:
   - subscription_id (VARCHAR, PRIMARY KEY)
   - merchant_user_id (VARCHAR) 
   - status (ENUM: 'ACTIVE', 'INACTIVE', 'CLOSED', 'REJECT', 'INIT')
   - subcription_start_date (DATE) -- Note: column name has typo "subcription"

2. subscription_payment_details:
   - subscription_id (VARCHAR, FOREIGN KEY to subscription_contract_v2.subscription_id)
   - status (ENUM: 'ACTIVE', 'FAILED', 'FAIL', 'INIT') 
   - trans_amount_decimal (DECIMAL)
   - created_date (DATE)
   - NOTE: This table does NOT have merchant_user_id directly

CRITICAL COMPLETE SCHEMA RULES:
- To get merchant_user_id info from payments, you MUST JOIN:
  FROM subscription_payment_details p 
  JOIN subscription_contract_v2 c ON p.subscription_id = c.subscription_id
  
- NEVER use merchant_user_id in GROUP BY with subscription_payment_details alone
- Always use proper JOINs when accessing both tables
- merchant_user_id is ONLY in subscription_contract_v2 table

COMPLETE THRESHOLD HANDLING:
- Pay VERY close attention to numbers in queries: "more than 1", "more than 2", "more than 10", etc.
- ALWAYS use the EXACT threshold number specified by the user
- For "more than X": use > X in the WHERE clause
- For "X or more": use >= X in the WHERE clause
- For "less than X": use < X in the WHERE clause

DATE HANDLING:
- Always use single quotes for date strings: '2025-04-23'
- Use DATE() function for date comparisons: DATE(created_date) = '2025-04-23'
- Never use double quotes around dates
- Remember column typo: subcription_start_date (not subscription_start_date)
"""

    def _get_chart_keywords(self) -> Dict[str, List[str]]:
        """Chart type keywords for better detection."""
        return {
            'pie': ['pie', 'pie chart', 'distribution', 'breakdown', 'percentage', 'proportion', 'rate', 'visually'],
            'line': ['line', 'trend', 'over time', 'timeline', 'time series', 'progression'],
            'bar': ['bar', 'comparison', 'compare', 'versus', 'vs'],
            'scatter': ['scatter', 'correlation', 'relationship', 'plot']
        }

    def _get_tools_config(self):
        """Get complete tools configuration."""
        return [
            genai.protos.Tool(
                function_declarations=[
                    genai.protos.FunctionDeclaration(
                        name="get_subscriptions_in_last_days",
                        description="Get subscription statistics for the last N days",
                        parameters=genai.protos.Schema(
                            type=genai.protos.Type.OBJECT,
                            properties={
                                "days": genai.protos.Schema(type=genai.protos.Type.INTEGER, description="Number of days (1-365)")
                            },
                            required=["days"]
                        )
                    ),
                    genai.protos.FunctionDeclaration(
                        name="get_payment_success_rate_in_last_days",
                        description="Get payment success rate and revenue statistics for the last N days",
                        parameters=genai.protos.Schema(
                            type=genai.protos.Type.OBJECT,
                            properties={
                                "days": genai.protos.Schema(type=genai.protos.Type.INTEGER, description="Number of days (1-365)")
                            },
                            required=["days"]
                        )
                    ),
                    genai.protos.FunctionDeclaration(
                        name="get_user_payment_history",
                        description="Get payment history for a specific user",
                        parameters=genai.protos.Schema(
                            type=genai.protos.Type.OBJECT,
                            properties={
                                "merchant_user_id": genai.protos.Schema(type=genai.protos.Type.STRING, description="The merchant user ID"),
                                "days": genai.protos.Schema(type=genai.protos.Type.INTEGER, description="Days to look back (default: 90)")
                            },
                            required=["merchant_user_id"]
                        )
                    ),
                    genai.protos.FunctionDeclaration(
                        name="get_database_status",
                        description="Check database connection and get basic statistics",
                        parameters=genai.protos.Schema(type=genai.protos.Type.OBJECT, properties={})
                    ),
                    genai.protos.FunctionDeclaration(
                        name="execute_dynamic_sql",
                        description="Execute a custom SQL SELECT query for analytics",
                        parameters=genai.protos.Schema(
                            type=genai.protos.Type.OBJECT,
                            properties={
                                "sql_query": genai.protos.Schema(type=genai.protos.Type.STRING, description="SELECT SQL query to execute")
                            },
                            required=["sql_query"]
                        )
                    ),
                    genai.protos.FunctionDeclaration(
                        name="execute_dynamic_sql_with_graph",
                        description="Execute a SQL query AND generate a graph visualization",
                        parameters=genai.protos.Schema(
                            type=genai.protos.Type.OBJECT,
                            properties={
                                "sql_query": genai.protos.Schema(type=genai.protos.Type.STRING, description="SELECT SQL query to execute"),
                                "graph_type": genai.protos.Schema(type=genai.protos.Type.STRING, description="Graph type: line, bar, horizontal_bar, pie, scatter")
                            },
                            required=["sql_query"]
                        )
                    )
                ]
            )
        ]

    async def parse_query(self, user_query: str, history: List[str], client=None) -> List[Dict]:
        """FIXED MULTITOOL SUPPORT: Parse query and return list of tool calls with proper handling for multiple queries"""
        # Detect comparison queries and handle as a single query
        query_lower = user_query.lower().strip()
        threshold_info = self._extract_threshold_info(user_query)
        is_comparison = False
        if (threshold_info['has_threshold'] and threshold_info['numbers'] and
            (('compare' in query_lower or 'vs' in query_lower or 'versus' in query_lower or 'and' in query_lower))):
            numbers = threshold_info['numbers']
            if len(numbers) >= 2:
                is_comparison = True
        # Only split if not a comparison query
        if not is_comparison:
            query_separators = [' and ', ';', '\n']
            individual_queries = [user_query.strip()]
            for separator in query_separators:
                new_queries = []
                for query in individual_queries:
                    parts = [q.strip() for q in query.split(separator) if q.strip()]
                    if len(parts) > 1:
                        new_queries.extend(parts)
                    else:
                        new_queries.append(query)
                individual_queries = new_queries
            # Remove duplicates while preserving order
            seen = set()
            unique_queries = []
            for query in individual_queries:
                if query.lower() not in seen:
                    seen.add(query.lower())
                    unique_queries.append(query)
        else:
            unique_queries = [user_query.strip()]
        logger.info(f"🔧 MULTITOOL: Processing {len(unique_queries)} individual queries")
        all_tool_calls = []
        for i, query in enumerate(unique_queries, 1):
            logger.info(f"🔧 MULTITOOL: Processing query {i}/{len(unique_queries)}: {query[:50]}...")
            try:
                # --- FEEDBACK-AWARE LOGIC ---
                auto_applied = False
                auto_union = False
                auto_no_graph = False
                auto_chart_type = None
                suggestions_result = None
                if client:
                    try:
                        suggestions_result = await client.call_tool('get_improvement_suggestions', {
                            'original_question': query
                        })
                        if getattr(suggestions_result, 'success', False) and getattr(suggestions_result, 'data', None) and suggestions_result.data.get('improvements'):
                            for suggestion in suggestions_result.data['improvements']:
                                sug = suggestion['user_suggestion'].lower()
                                if 'do not generate a graph' in sug or 'no graph' in sug or 'no chart' in sug:
                                    auto_no_graph = True
                                    auto_applied = True
                                    print_section(f'💡 Auto-applying past improvement: "{suggestion["user_suggestion"]}"')
                                if 'use union' in sug or 'same format' in sug or 'consistent format' in sug:
                                    auto_union = True
                                    auto_applied = True
                                    print_section(f'💡 Auto-applying past improvement: "{suggestion["user_suggestion"]}"')
                                if 'bar chart' in sug:
                                    auto_chart_type = 'bar'
                                    auto_applied = True
                                    print_section(f'💡 Auto-applying past improvement: "{suggestion["user_suggestion"]}"')
                                if 'pie chart' in sug:
                                    auto_chart_type = 'pie'
                                    auto_applied = True
                                    print_section(f'💡 Auto-applying past improvement: "{suggestion["user_suggestion"]}"')
                                if 'line chart' in sug:
                                    auto_chart_type = 'line'
                                    auto_applied = True
                                    print_section(f'💡 Auto-applying past improvement: "{suggestion["user_suggestion"]}"')
                    except Exception as e:
                        logger.warning(f"Could not fetch improvement suggestions: {e}")
                # --- END FEEDBACK-AWARE LOGIC ---
                query_tool_calls = await self._process_single_query(query, history, client, auto_union=auto_union, auto_no_graph=auto_no_graph, auto_chart_type=auto_chart_type, force_comparison=is_comparison)
                for call in query_tool_calls:
                    call['query_index'] = i
                    call['total_queries'] = len(unique_queries)
                    call['is_multitool'] = len(unique_queries) > 1
                all_tool_calls.extend(query_tool_calls)
                logger.info(f"🔧 MULTITOOL: Query {i} generated {len(query_tool_calls)} tool calls")
            except Exception as e:
                logger.error(f"❌ MULTITOOL: Error processing query {i}: {e}")
                error_call = {
                    'tool': 'get_database_status',  # Safe fallback
                    'parameters': {},
                    'original_query': query,
                    'wants_graph': False,
                    'chart_analysis': {'chart_type': 'none'},
                    'query_index': i,
                    'total_queries': len(unique_queries),
                    'is_multitool': len(unique_queries) > 1,
                    'error': str(e)
                }
                all_tool_calls.append(error_call)
        logger.info(f"✅ MULTITOOL: Generated total of {len(all_tool_calls)} tool calls for {len(unique_queries)} queries")
        if not all_tool_calls:
            return [{
                'tool': 'get_database_status',
                'parameters': {},
                'original_query': user_query,
                'wants_graph': False,
                'chart_analysis': {'chart_type': 'none'},
                'query_index': 1,
                'total_queries': 1,
                'is_multitool': False
            }]
        return all_tool_calls

    async def _process_single_query(self, query: str, history: List[str], client=None, auto_union=False, auto_no_graph=False, auto_chart_type=None, force_comparison=False) -> List[Dict]:
        """Process a single query and return tool calls, with feedback-aware logic. If force_comparison is True, always generate a single UNION SQL."""
        query_lower = query.lower().strip()
        # 1. Handle specific date queries directly
        date_info = self._extract_date_info(query)
        if date_info['has_date'] and date_info['dates']:
            date_str = date_info['dates'][0]
            try:
                if re.match(r'\d{1,2}\s+(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\w*\s+\d{4}', date_str, re.IGNORECASE):
                    date_obj = datetime.strptime(date_str, '%d %B %Y')
                    date_str = date_obj.strftime('%Y-%m-%d')
                elif re.match(r'\d{2}/\d{2}/\d{4}', date_str):
                    date_obj = datetime.strptime(date_str, '%d/%m/%Y')
                    date_str = date_obj.strftime('%Y-%m-%d')
            except Exception:
                pass
            sql = f"SELECT COUNT(*) as num_subscriptions FROM subscription_contract_v2 WHERE DATE(subcription_start_date) = '{date_str}'"
            sql = self._fix_sql_quotes(sql)
            sql = self._validate_and_autofix_sql(sql)
            sql = self._fix_sql_date_math(sql)
            return [{
                'tool': 'execute_dynamic_sql',
                'parameters': {'sql_query': sql},
                'original_query': query,
                'wants_graph': False,
                'chart_analysis': {'chart_type': 'none'}
            }]
        # 2. Handle comparison queries with UNION ALL
        threshold_info = self._extract_threshold_info(query)
        if force_comparison or (threshold_info['has_threshold'] and threshold_info['numbers'] and (('compare' in query_lower or 'vs' in query_lower or 'versus' in query_lower or 'and' in query_lower))):
            numbers = threshold_info['numbers']
            if len(numbers) >= 2:
                sql = f"""
SELECT 'More than {numbers[0]} Subscriptions' as category, COUNT(*) as value 
FROM (SELECT merchant_user_id FROM subscription_contract_v2 GROUP BY merchant_user_id HAVING COUNT(*) > {numbers[0]}) t1
UNION ALL
SELECT 'More than {numbers[1]} Subscriptions' as category, COUNT(*) as value  
FROM (SELECT merchant_user_id FROM subscription_contract_v2 GROUP BY merchant_user_id HAVING COUNT(*) > {numbers[1]}) t2
"""
                sql = self._fix_sql_quotes(sql)
                sql = self._validate_and_autofix_sql(sql)
                sql = self._fix_sql_date_math(sql)
                return [{
                    'tool': 'execute_dynamic_sql',
                    'parameters': {'sql_query': sql},
                    'original_query': query,
                    'wants_graph': False,
                    'chart_analysis': {'chart_type': 'none'}
                }]
        # 2b. Single threshold
        if threshold_info['has_threshold'] and threshold_info['numbers']:
            threshold = threshold_info['numbers'][0]
            if 'subscription' in query_lower and ('more than' in query_lower or 'greater than' in query_lower):
                sql = f"""
SELECT COUNT(*) as num_subscribers 
FROM (
    SELECT merchant_user_id 
    FROM subscription_contract_v2 
    GROUP BY merchant_user_id 
    HAVING COUNT(*) > {threshold}
) as t
"""
                sql = self._fix_sql_quotes(sql)
                sql = self._validate_and_autofix_sql(sql)
                sql = self._fix_sql_date_math(sql)
                return [{
                    'tool': 'execute_dynamic_sql',
                    'parameters': {'sql_query': sql},
                    'original_query': query,
                    'wants_graph': False,
                    'chart_analysis': {'chart_type': 'none'}
                }]
        # 3. Handle visualization requests with smart chart selection
        wants_graph = any(word in query_lower for word in ['graph', 'visualize', 'bar chart', 'pie chart', 'line chart', 'show as chart', 'chart'])
        if auto_no_graph:
            wants_graph = False
        chart_analysis = self._analyze_complete_chart_requirements(query, history)
        if auto_chart_type:
            chart_analysis['chart_type'] = auto_chart_type
        if wants_graph:
            if ('bar chart' in query_lower or 'visualize' in query_lower) and 'payment' in query_lower:
                if not any(word in query_lower for word in ['over time', 'by date', 'trend', 'daily', 'per day', 'each day', 'timeline', 'monthly', 'week']):
                    sql = """
SELECT c.merchant_user_id, COUNT(*) AS total_payments
FROM subscription_payment_details p
JOIN subscription_contract_v2 c ON p.subscription_id = c.subscription_id
GROUP BY c.merchant_user_id
ORDER BY total_payments DESC
LIMIT 20
"""
                    sql = self._fix_sql_quotes(sql)
                    sql = self._validate_and_autofix_sql(sql)
                    sql = self._fix_sql_date_math(sql)
                    return [{
                        'tool': 'execute_dynamic_sql_with_graph',
                        'parameters': {'sql_query': sql, 'graph_type': 'bar'},
                        'original_query': query,
                        'wants_graph': True,
                        'chart_analysis': chart_analysis
                    }]
            if 'pie chart' in query_lower and ('success' in query_lower or 'failure' in query_lower or 'rate' in query_lower):
                sql = """
SELECT 
    CASE WHEN status = 'ACTIVE' THEN 'Successful' ELSE 'Failed' END as category,
    COUNT(*) as value
FROM subscription_payment_details
GROUP BY CASE WHEN status = 'ACTIVE' THEN 'Successful' ELSE 'Failed' END
"""
                sql = self._fix_sql_quotes(sql)
                sql = self._validate_and_autofix_sql(sql)
                sql = self._fix_sql_date_math(sql)
                return [{
                    'tool': 'execute_dynamic_sql_with_graph',
                    'parameters': {'sql_query': sql, 'graph_type': 'pie'},
                    'original_query': query,
                    'wants_graph': True,
                    'chart_analysis': chart_analysis
                }]
        # 4. Fall back to AI processing for complex queries
        try:
            comparison_info = self._extract_comparison_info(query)
            history_context = self._build_complete_history_context(history)
            improvement_context = await self._get_complete_improvement_context(query, history, client)
            similar_context = await self._get_similar_queries_context(query, client)
            if auto_chart_type:
                chart_analysis['chart_type'] = auto_chart_type
            prompt = self._create_enhanced_threshold_prompt(
                query, history_context, improvement_context, similar_context, 
                chart_analysis, threshold_info, date_info, comparison_info
            )
            tool_calls = await self._generate_with_complete_retries(prompt, query, chart_analysis)
            if auto_no_graph:
                for call in tool_calls:
                    if call['tool'] == 'execute_dynamic_sql_with_graph':
                        call['tool'] = 'execute_dynamic_sql'
                        call['wants_graph'] = False
            if auto_chart_type:
                for call in tool_calls:
                    if call['tool'] == 'execute_dynamic_sql_with_graph':
                        call['parameters']['graph_type'] = auto_chart_type
            for call in tool_calls:
                if 'sql_query' in call['parameters']:
                    call['parameters']['sql_query'] = self._fix_sql_quotes(call['parameters']['sql_query'])
                    call['parameters']['sql_query'] = self._validate_and_autofix_sql(call['parameters']['sql_query'])
                    call['parameters']['sql_query'] = self._fix_sql_date_math(call['parameters']['sql_query'])
            enhanced_calls = self._enhance_and_validate_complete_tool_calls(
                tool_calls, query, chart_analysis, threshold_info
            )
            logger.info(f"🧠 AI selected tool(s): {[tc['tool'] for tc in enhanced_calls]}")
            return enhanced_calls
        except Exception as e:
            logger.error(f"Error in AI query processing: {e}", exc_info=True)
            return self._get_complete_smart_fallback_tool_call(query, history)

    def _extract_threshold_info(self, query: str) -> Dict:
        """Extract threshold information from query with enhanced accuracy."""
        threshold_info = {
            'has_threshold': False,
            'numbers': [],
            'operators': [],
            'context': 'unknown'
        }
        
        query_lower = query.lower()
        
        # Extract numbers
        numbers = re.findall(r'\d+', query)
        threshold_info['numbers'] = [int(n) for n in numbers]
        
        # Detect threshold operators
        threshold_phrases = [
            'more than', 'less than', 'at least', 'or more', 'at most', 'or fewer', 'exactly', 'equal to'
        ]
        if any(phrase in query_lower for phrase in threshold_phrases):
            if 'more than' in query_lower:
                threshold_info['operators'].append('>')
            if 'less than' in query_lower:
                threshold_info['operators'].append('<')
            if 'at least' in query_lower or 'or more' in query_lower:
                threshold_info['operators'].append('>=')
            if 'at most' in query_lower or 'or fewer' in query_lower:
                threshold_info['operators'].append('<=')
            if 'exactly' in query_lower or 'equal to' in query_lower:
                threshold_info['operators'].append('=')
            threshold_info['has_threshold'] = True
        
        # Detect context
        if 'subscription' in query_lower:
            threshold_info['context'] = 'subscriptions'
        elif 'transaction' in query_lower or 'payment' in query_lower:
            threshold_info['context'] = 'transactions'
        elif 'merchant' in query_lower or 'user' in query_lower:
            threshold_info['context'] = 'users'
        
        return threshold_info

    def _extract_date_info(self, query: str) -> Dict:
        """Extract date information from query."""
        date_info = {
            'has_date': False,
            'dates': [],
            'date_context': 'unknown'
        }
        
        # Extract dates in various formats
        date_patterns = [
            r'\d{4}-\d{2}-\d{2}',  # YYYY-MM-DD
            r'\d{2}/\d{2}/\d{4}',  # MM/DD/YYYY
            r'\d{1,2}\s+(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\w*\s+\d{4}',  # 23 april 2025
        ]
        
        for pattern in date_patterns:
            matches = re.findall(pattern, query, re.IGNORECASE)
            if matches:
                date_info['dates'].extend(matches)
                date_info['has_date'] = True
        
        # Detect date context
        query_lower = query.lower()
        if 'new' in query_lower and ('subscriber' in query_lower or 'subscription' in query_lower):
            date_info['date_context'] = 'new_subscriptions'
        elif 'payment' in query_lower or 'transaction' in query_lower:
            date_info['date_context'] = 'payments'
        elif 'last date' in query_lower or 'available' in query_lower:
            date_info['date_context'] = 'data_availability'
        
        return date_info

    def _extract_comparison_info(self, query: str) -> Dict:
        """Extract comparison information from query."""
        comparison_info = {
            'is_comparison': False,
            'elements': [],
            'comparison_type': 'unknown'
        }
        
        query_lower = query.lower()
        
        # Detect comparison keywords
        comparison_keywords = ['compare', 'vs', 'versus', 'and', 'both', 'between']
        if any(keyword in query_lower for keyword in comparison_keywords):
            comparison_info['is_comparison'] = True
        
        # Extract comparison elements for threshold comparisons
        # Look for patterns like "more than 1" and "more than 2"
        threshold_matches = re.findall(r'more than (\d+)', query_lower)
        if len(threshold_matches) >= 2:
            comparison_info['is_comparison'] = True
            comparison_info['elements'] = threshold_matches
            comparison_info['comparison_type'] = 'threshold_comparison'
        
        # Look for active vs inactive patterns
        if 'active' in query_lower and ('inactive' in query_lower or 'other' in query_lower):
            comparison_info['is_comparison'] = True
            comparison_info['comparison_type'] = 'status_comparison'
        
        return comparison_info

    def _create_enhanced_threshold_prompt(self, user_query: str, history_context: str, 
                                        improvement_context: str, similar_context: str, 
                                        chart_analysis: Dict, threshold_info: Dict, 
                                        date_info: Dict, comparison_info: Dict) -> str:
        """Create enhanced prompt with threshold awareness and better AI guidance."""
        # Smart metric/ARPU detection
        metric_keywords = ['arpu', 'average revenue per user', 'average revenue', 'mean revenue', 'arppu', 'arpau', 'total revenue', 'sum', 'average', 'mean']
        user_query_lower = user_query.lower()
        is_metric_query = any(k in user_query_lower for k in metric_keywords)
        metric_guidance = ""
        if is_metric_query:
            metric_guidance = (
                "\n\n🔥 CRITICAL METRIC QUERY RULES (READ CAREFULLY):\n"
                "- If the user asks for ARPU (average revenue per user) or any average/total metric, ALWAYS return a single value (not a breakdown or category table).\n"
                "- Ignore unrelated numbers or thresholds unless the user explicitly requests a comparison.\n"
                "- The output should be a single row: e.g., ARPU: 1234.56\n"
                "- If the user asks for a comparison (e.g., ARPU for two groups), then use UNION ALL.\n"
                "- Otherwise, for metric queries, return only the metric value.\n"
                "- If you are unsure, err on the side of returning a single value.\n"
                "\nEXAMPLE (ARPU, single value):\n"
                "User: Calculate the average revenue per paying user (ARPU) over the last quarter\n"
                "SQL: SELECT SUM(p.trans_amount_decimal) / COUNT(DISTINCT c.merchant_user_id) AS ARPU\n"
                "     FROM subscription_payment_details p\n"
                "     JOIN subscription_contract_v2 c ON p.subscription_id = c.subscription_id\n"
                "     WHERE DATE(p.created_date) >= '2025-04-01' AND p.status = 'ACTIVE'\n"
                "Output: ARPU: 1234.56\n"
                "\n-- For MySQL, NEVER use DATE('now', '-N days'). Instead, use DATE_SUB(CURDATE(), INTERVAL N DAY).\n"
                "-- Example: WHERE DATE(p.created_date) >= DATE_SUB(CURDATE(), INTERVAL 90 DAY)\n"
                "-- For 'last quarter', use the correct date range or interval.\n"
            )
        threshold_guidance = self._get_threshold_guidance(threshold_info, comparison_info)
        date_guidance = self._get_date_guidance(date_info)
        chart_guidance = self._get_complete_chart_guidance(chart_analysis)
        return f"""
You are an expert subscription analytics assistant with ENHANCED database knowledge and PERFECT threshold handling.

CONVERSATION HISTORY:
{history_context}

{improvement_context}

{similar_context}

CURRENT USER QUERY: "{user_query}"

{metric_guidance}
{threshold_guidance}
{date_guidance}
{chart_guidance}
{self.db_schema}

🔥 CRITICAL THRESHOLD AND COMPARISON RULES:
1. ⚠️ EXACT NUMBERS: If user says "more than 2", use EXACTLY > 2 (NOT > 1 or any other number!)
2. ⚠️ If user says "more than 10", use EXACTLY > 10 (NOT > 1 or any other number!)
3. ⚠️ If user says "more than 100", use EXACTLY > 100 (NOT > 1 or any other number!)
4. ⚠️ For comparisons between thresholds, create UNION queries showing both categories
5. ⚠️ DATES: Always use single quotes and DATE(): DATE(created_date) = '2025-04-23'
6. ⚠️ JOINS: For merchant info with payments, ALWAYS JOIN tables
7. ⚠️ TYPO: Column is "subcription_start_date" not "subscription_start_date"

ENHANCED EXAMPLES WITH CORRECT THRESHOLDS:

1. "subscribers with more than 10 transactions":
   SELECT CASE WHEN total_transactions > 10 THEN 'More than 10 Transactions' ELSE '10 or Fewer Transactions' END as category, COUNT(*) as value
   FROM (SELECT c.merchant_user_id, COUNT(*) as total_transactions
         FROM subscription_payment_details p JOIN subscription_contract_v2 c ON p.subscription_id = c.subscription_id
         GROUP BY c.merchant_user_id) stats
   GROUP BY CASE WHEN total_transactions > 10 THEN 'More than 10 Transactions' ELSE '10 or Fewer Transactions' END

2. "new subscribers on 2025-04-23":
   SELECT COUNT(*) as new_subscribers FROM subscription_contract_v2 
   WHERE DATE(subcription_start_date) = '2025-04-23'

3. "number of subscriptions on 2025-04-24":
   SELECT COUNT(*) as num_subscriptions FROM subscription_contract_v2 WHERE DATE(subcription_start_date) = '2025-04-24'

4. "compare more than 1 vs more than 2 subscriptions":
   SELECT 'More than 1 Subscription' as category, COUNT(*) as value 
   FROM (SELECT merchant_user_id FROM subscription_contract_v2 GROUP BY merchant_user_id HAVING COUNT(*) > 1) t1
   UNION ALL
   SELECT 'More than 2 Subscriptions' as category, COUNT(*) as value  
   FROM (SELECT merchant_user_id FROM subscription_contract_v2 GROUP BY merchant_user_id HAVING COUNT(*) > 2) t2

5. "last date for which data is available":
   SELECT MAX(DATE(created_date)) as last_payment_date FROM subscription_payment_details
   UNION ALL
   SELECT MAX(DATE(subcription_start_date)) as last_subscription_date FROM subscription_contract_v2

6. "subscribers with more than 100 transactions":
   SELECT CASE WHEN total_transactions > 100 THEN 'More than 100 Transactions' ELSE '100 or Fewer Transactions' END as category, COUNT(*) as value
   FROM (SELECT c.merchant_user_id, COUNT(*) as total_transactions
         FROM subscription_payment_details p JOIN subscription_contract_v2 c ON p.subscription_id = c.subscription_id
         GROUP BY c.merchant_user_id) stats
   GROUP BY CASE WHEN total_transactions > 100 THEN 'More than 100 Transactions' ELSE '100 or Fewer Transactions' END

7. "ARPU over the last quarter":
   SELECT SUM(p.trans_amount_decimal) / COUNT(DISTINCT c.merchant_user_id) AS ARPU
   FROM subscription_payment_details p
   JOIN subscription_contract_v2 c ON p.subscription_id = c.subscription_id
   WHERE DATE(p.created_date) >= '2025-04-01' AND p.status = 'ACTIVE'
   -- Output should be a single row: ARPU: <value>

CRITICAL SQL GENERATION RULES:
- Always use single quotes for strings and dates: '2025-04-23'
- Never use double quotes in SQL
- Use EXACT thresholds specified by user - if they say "more than 2", use > 2, NOT > 1!
- For date queries, always use DATE() function for date columns
- For MySQL, NEVER use DATE('now', '-N days'). Instead, use DATE_SUB(CURDATE(), INTERVAL N DAY).
- For 'last quarter', use the correct date range or interval.
- Pay attention to the typo in column name: subcription_start_date
- For pie charts, use execute_dynamic_sql_with_graph with graph_type="pie"
- For comparisons, use UNION ALL to show multiple categories

🎯 REMEMBER: The user is asking for SPECIFIC thresholds. Use the EXACT numbers they specify!
"""

    def _get_threshold_guidance(self, threshold_info: Dict, comparison_info: Dict) -> str:
        """Generate specific guidance for threshold handling."""
        if not threshold_info['has_threshold']:
            return ""
        
        guidance = ["⚠️ THRESHOLD DETECTION ALERT:"]
        
        if threshold_info['numbers']:
            guidance.append(f"- Numbers detected: {threshold_info['numbers']}")
            guidance.append(f"- ⚠️ CRITICAL: Use EXACTLY these numbers in SQL: {threshold_info['numbers']}")
        
        if threshold_info['operators']:
            guidance.append(f"- Operators detected: {threshold_info['operators']}")
        
        if comparison_info['is_comparison'] and comparison_info['comparison_type'] == 'threshold_comparison':
            guidance.append("- ⚠️ COMPARISON QUERY: User wants to compare DIFFERENT thresholds")
            guidance.append("- Use UNION ALL to show both categories separately")
            guidance.append(f"- Compare thresholds: {comparison_info['elements']}")
        
        guidance.append(f"- Context: {threshold_info['context']}")
        guidance.append("- ⚠️ CRITICAL: Use the EXACT numbers specified by the user!")
        
        return "\n".join(guidance) + "\n"

    def _get_date_guidance(self, date_info: Dict) -> str:
        """Generate specific guidance for date handling."""
        if not date_info['has_date']:
            return ""
        
        guidance = ["📅 DATE QUERY DETECTED:"]
        
        if date_info['dates']:
            guidance.append(f"- Dates found: {date_info['dates']}")
        
        guidance.append(f"- Context: {date_info['date_context']}")
        guidance.append("- ⚠️ Use single quotes for dates: '2025-04-23'")
        guidance.append("- ⚠️ Use DATE() function: DATE(created_date) = '2025-04-23'")
        guidance.append("- ⚠️ Remember column typo: subcription_start_date (not subscription_start_date)")
        
        if date_info['date_context'] == 'data_availability':
            guidance.append("- ⚠️ For 'last date available', use MAX(DATE()) queries")
        
        return "\n".join(guidance) + "\n"

    def _build_complete_history_context(self, history: List[str], user_query: str = None) -> str:
        """Build complete contextual history with smart filtering. For metric/ARPU queries, filter out unrelated feedback/history."""
        if not history:
            return "No previous context."
        # Smart filtering for metric queries
        metric_keywords = ['arpu', 'average revenue per user', 'average revenue', 'mean revenue', 'arppu', 'arpau', 'total revenue', 'sum', 'average', 'mean']
        is_metric_query = False
        if user_query:
            user_query_lower = user_query.lower()
            is_metric_query = any(k in user_query_lower for k in metric_keywords)
        recent_history = history[-6:] if len(history) > 6 else history
        context_lines = []
        for line in recent_history:
            if is_metric_query:
                # Only include lines relevant to metrics
                if any(k in line.lower() for k in metric_keywords):
                    context_lines.append(line)
            else:
                if any(keyword in line.lower() for keyword in ['feedback', 'improve', 'try again', 'pie chart', 'bar chart', 'error']):
                    context_lines.append(f"IMPORTANT: {line}")
                else:
                    context_lines.append(line)
        if not context_lines:
            return "No previous context."
        return "\n".join(context_lines)

    async def _get_complete_improvement_context(self, user_query: str, history: List[str], client) -> str:
        """Get complete improvement context with full history analysis and auto-apply best improvement suggestion if available."""
        try:
            improvement_lines = ["COMPLETE LEARNED IMPROVEMENTS AND CONTEXT:"]
            best_suggestion = None
            best_chart_type = None
            # Analyze recent history for improvement cues
            if history:
                recent_feedback = self._extract_complete_recent_feedback(history)
                if recent_feedback:
                    improvement_lines.append(f"RECENT USER FEEDBACK: {recent_feedback}")
            # Get semantic improvements if available
            if client:
                try:
                    suggestions_result = await client.call_tool('get_improvement_suggestions', {
                        'original_question': user_query
                    })
                    if (suggestions_result.success and 
                        suggestions_result.data and 
                        suggestions_result.data.get('improvements')):
                        improvements = suggestions_result.data['improvements'][:2]  # Limit to top 2
                        improvement_lines.append("PAST USER IMPROVEMENTS:")
                        # Auto-apply the best suggestion (highest similarity)
                        if improvements:
                            best_suggestion = improvements[0]['user_suggestion']
                            # Extract chart type from improvement
                            imp = improvements[0]
                            if 'chart_type' in imp and imp['chart_type']:
                                best_chart_type = imp['chart_type']
                            else:
                                # Fallback: parse from suggestion text
                                suggestion_text = best_suggestion.lower()
                                if 'bar chart' in suggestion_text:
                                    best_chart_type = 'bar'
                                elif 'pie chart' in suggestion_text:
                                    best_chart_type = 'pie'
                                elif 'line chart' in suggestion_text:
                                    best_chart_type = 'line'
                                elif 'scatter' in suggestion_text:
                                    best_chart_type = 'scatter'
                        for improvement in improvements:
                            improvement_lines.append(f"- Issue: {improvement['user_suggestion']}")
                            improvement_lines.append(f"  Context: {improvement['similar_question']}")
                            improvement_lines.append(f"  Category: {improvement['improvement_category']}")
                except Exception as e:
                    logger.debug(f"Could not get improvement suggestions: {e}")
            # If there is a best suggestion, inject it as a direct instruction
            if best_suggestion:
                improvement_lines.insert(1, f"AUTO-APPLIED IMPROVEMENT: {best_suggestion}")
            # Return both the improvement context and the best chart type for enforcement
            self._last_best_chart_type = best_chart_type  # Store for parse_query to use
            return "\n".join(improvement_lines) if len(improvement_lines) > 1 else ""
        except Exception as e:
            logger.warning(f"Could not get complete improvement context: {e}")
            self._last_best_chart_type = None
            return ""

    async def _get_similar_queries_context(self, user_query: str, client) -> str:
        """Get similar successful queries for better context."""
        try:
            if not client:
                return ""
            
            similar_result = await client.call_tool('get_similar_queries', {
                'original_question': user_query
            })
            
            if (similar_result.success and 
                similar_result.data and 
                similar_result.data.get('queries')):
                
                similar_queries = similar_result.data['queries'][:2]  # Top 2
                
                context_lines = ["SIMILAR SUCCESSFUL QUERIES FOR REFERENCE:"]
                for query in similar_queries:
                    context_lines.append(f"- Similar Q: {query['question']}")
                    context_lines.append(f"  Successful SQL: {query['successful_sql']}")
                    context_lines.append(f"  Category: {query['query_category']}")
                
                return "\n".join(context_lines)
            
            return ""
            
        except Exception as e:
            logger.debug(f"Could not get similar queries context: {e}")
            return ""

    def _extract_complete_recent_feedback(self, history: List[str]) -> str:
        """Extract complete recent feedback from conversation history."""
        try:
            # Look for feedback in last few turns
            for line in reversed(history[-4:]):
                if 'pie chart' in line.lower():
                    return "User specifically requested PIE CHART visualization"
                elif 'bar chart' in line.lower():
                    return "User specifically requested BAR CHART visualization"
                elif 'improve' in line.lower() and ('rate' in line.lower() or 'success' in line.lower()):
                    return "User wants success/failure rate analysis"
                elif 'try again' in line.lower():
                    return "User wants to retry with previous feedback incorporated"
                elif 'error' in line.lower() or 'wrong' in line.lower():
                    return "Previous query had errors - user wants corrected version"
                elif 'merchant' in line.lower() and 'transaction' in line.lower():
                    return "User asking about merchant transaction analysis"
                elif 'threshold' in line.lower() or 'number' in line.lower():
                    return "User wants specific threshold/number analysis"
            
            return ""
        except Exception:
            return ""

    def _analyze_complete_chart_requirements(self, user_query: str, history: List[str]) -> Dict:
        """Analyze complete chart/visualization requirements."""
        query_lower = user_query.lower()
        analysis = {
            'wants_visualization': False,
            'chart_type': None,
            'data_aggregation': None,
            'specific_request': None,
            'is_merchant_analysis': False,
            'needs_success_failure_breakdown': False
        }
        
        # Check for visualization keywords
        viz_keywords = ['chart', 'graph', 'plot', 'visualize', 'show', 'display', 'visually']
        if any(keyword in query_lower for keyword in viz_keywords):
            analysis['wants_visualization'] = True
        
        # Detect specific chart types
        for chart_type, keywords in self.chart_keywords.items():
            if any(keyword in query_lower for keyword in keywords):
                analysis['chart_type'] = chart_type
                break
        
        # Check for merchant analysis
        if 'merchant' in query_lower and ('transaction' in query_lower or 'payment' in query_lower):
            analysis['is_merchant_analysis'] = True
        
        # Check for success/failure analysis
        if any(word in query_lower for word in ['success', 'failure', 'rate']):
            analysis['needs_success_failure_breakdown'] = True
        
        # Check history for chart requests
        if not analysis['chart_type'] and history:
            for line in reversed(history[-3:]):
                if 'pie chart' in line.lower():
                    analysis['chart_type'] = 'pie'
                    analysis['specific_request'] = "User previously requested pie chart"
                    break
        
        # Determine data aggregation needs
        if analysis['chart_type'] == 'pie':
            if analysis['is_merchant_analysis']:
                analysis['data_aggregation'] = 'merchant_aggregation'
            elif analysis['needs_success_failure_breakdown']:
                analysis['data_aggregation'] = 'success_failure_breakdown'
            else:
                analysis['data_aggregation'] = 'total_summary'
        elif 'trend' in query_lower or 'over time' in query_lower:
            analysis['data_aggregation'] = 'time_series'
        elif 'rate' in query_lower or 'percentage' in query_lower:
            analysis['data_aggregation'] = 'rate_calculation'
        
        return analysis

    def _get_complete_chart_guidance(self, chart_analysis: Dict) -> str:
        """Get complete specific guidance based on chart analysis."""
        if not chart_analysis['wants_visualization']:
            return ""
        
        guidance = ["COMPLETE CHART REQUIREMENTS DETECTED:"]
        
        if chart_analysis['chart_type']:
            guidance.append(f"- Requested chart type: {chart_analysis['chart_type'].upper()}")
        
        if chart_analysis['data_aggregation']:
            guidance.append(f"- Data aggregation needed: {chart_analysis['data_aggregation']}")
        
        if chart_analysis['specific_request']:
            guidance.append(f"- IMPORTANT: {chart_analysis['specific_request']}")
        
        if chart_analysis['is_merchant_analysis']:
            guidance.append("- MERCHANT ANALYSIS: Use proper JOINs and categorization")
        
        if chart_analysis['needs_success_failure_breakdown']:
            guidance.append("- SUCCESS/FAILURE: Use aggregated success vs failure analysis")
        
        if chart_analysis['chart_type'] == 'pie':
            guidance.append("- PIE CHART REQUIRES: Aggregated summary data with categories and totals")
            guidance.append("- DO NOT use time series data for pie charts")
            guidance.append("- USE: UNION or proper GROUP BY to create category/value pairs")
            guidance.append("- CRITICAL: Follow complete schema rules for merchant_user_id access")
            guidance.append("- FOR MERCHANTS: Use subqueries to categorize merchants by activity/success")
        
        return "\n".join(guidance) + "\n"

    async def _generate_with_complete_retries(self, prompt: str, user_query: str, chart_analysis: Dict, max_retries: int = 3) -> List[Dict]:
        """Generate AI response with complete enhanced retries and chart-specific handling."""
        for attempt in range(max_retries):
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
                            params = dict(fc.args)
                            
                            # Smart chart type injection
                            if fc.name == 'execute_dynamic_sql_with_graph' and chart_analysis.get('chart_type'):
                                if 'graph_type' not in params or not params['graph_type']:
                                    params['graph_type'] = chart_analysis['chart_type']
                            
                            tool_calls.append({
                                'tool': fc.name,
                                'parameters': params,
                                'original_query': user_query,
                                'wants_graph': fc.name == 'execute_dynamic_sql_with_graph',
                                'chart_analysis': chart_analysis
                            })
                
                if tool_calls:
                    return tool_calls
                    
            except Exception as e:
                logger.warning(f"Complete AI generation attempt {attempt + 1} failed: {e}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(1)
                    continue
        
        return []

    def _enhance_and_validate_complete_tool_calls(self, tool_calls: List[Dict], user_query: str, 
                                                chart_analysis: Dict, threshold_info: Dict = None) -> List[Dict]:
        """Enhanced tool call validation with threshold verification and robust parameter validation."""
        enhanced_calls = []
        
        for call in tool_calls:
            try:
                # Validate tool name
                tool = call.get('tool', '')
                if not tool or not isinstance(tool, str) or tool.strip() == '':
                    logger.warning(f"Skipping tool call with empty or invalid tool name: {call}")
                    continue
                
                # Parameter validation
                params = call.get('parameters', {})
                valid = True
                
                # Validate 'days' parameter
                if 'days' in params:
                    try:
                        days = int(params['days'])
                        if not (1 <= days <= 365):
                            logger.warning(f"Skipping tool call {tool}: days out of range ({days})")
                            valid = False
                    except Exception:
                        logger.warning(f"Skipping tool call {tool}: invalid days param {params['days']}")
                        valid = False
                
                # Validate user ID for user-specific tools
                if tool == 'get_user_payment_history' and not params.get('merchant_user_id'):
                    logger.warning(f"Skipping tool call {tool}: missing merchant_user_id")
                    valid = False
                
                # Validate SQL for dynamic SQL tools
                if tool.startswith('execute_dynamic_sql') and not params.get('sql_query'):
                    logger.warning(f"Skipping tool call {tool}: missing sql_query")
                    valid = False
                
                if not valid:
                    continue
                
                # Enhance SQL for complete schema compliance and threshold accuracy
                if 'sql_query' in call['parameters']:
                    original_sql = call['parameters']['sql_query']
                    enhanced_sql = self._fix_complete_sql_schema_issues(
                        original_sql, chart_analysis, user_query, threshold_info
                    )
                    enhanced_sql = self._fix_sql_quotes(enhanced_sql)
                    enhanced_sql = self._validate_and_autofix_sql(enhanced_sql)
                    call['parameters']['sql_query'] = enhanced_sql
                    
                    if original_sql != enhanced_sql:
                        logger.info("🧠 SQL fixed for complete schema compliance, MySQL compatibility, and syntax.")
                
                # Ensure graph type is set correctly
                if call['tool'] == 'execute_dynamic_sql_with_graph':
                    if chart_analysis.get('chart_type') and 'graph_type' not in call['parameters']:
                        call['parameters']['graph_type'] = chart_analysis['chart_type']
                
                enhanced_calls.append(call)
                
            except Exception as e:
                logger.error(f"Error enhancing complete tool call: {e}")
                continue  # Skip invalid calls instead of keeping them
        
        if not enhanced_calls:
            logger.warning("All tool calls filtered out, using fallback.")
            return self._get_complete_smart_fallback_tool_call(user_query, [])
        
        return enhanced_calls

    def _fix_complete_sql_schema_issues(self, sql_query: str, chart_analysis: Dict, 
                                      user_query: str, threshold_info: Dict = None) -> str:
        """Fix SQL with enhanced threshold handling and schema compliance."""
        try:
            # Clean SQL first
            sql_query = sql_query.strip().strip('"\'')

            # If the SQL is a simple count, do not modify it
            if re.match(r"SELECT\s+COUNT\(\*\)", sql_query, re.IGNORECASE):
                return sql_query
            
            # ENHANCED: Verify threshold accuracy
            if threshold_info and threshold_info['has_threshold'] and threshold_info['numbers']:
                sql_query = self._verify_and_fix_thresholds(sql_query, threshold_info, user_query)
            
            # Handle date queries specifically
            if 'last date' in user_query.lower() or 'available' in user_query.lower():
                if 'MAX' not in sql_query.upper():
                    # Convert to MAX date query
                    sql_query = """
SELECT MAX(DATE(created_date)) as last_payment_date FROM subscription_payment_details
UNION ALL
SELECT MAX(DATE(subcription_start_date)) as last_subscription_date FROM subscription_contract_v2
"""
            
            # Original schema fixes...
            if 'GROUP BY merchant_user_id' in sql_query and 'subscription_payment_details' in sql_query:
                logger.warning("🔧 Fixing merchant_user_id GROUP BY issue with complete logic")
                
                if chart_analysis.get('chart_type') == 'pie' and 'merchant' in user_query.lower():
                    # Use threshold from query if available
                    threshold = 1
                    if threshold_info and threshold_info['numbers']:
                        threshold = threshold_info['numbers'][0]
                    
                    sql_query = f"""
SELECT 
  CASE WHEN total_transactions > {threshold} THEN 'More than {threshold} Transactions' ELSE '{threshold} or Fewer Transactions' END as category,
  COUNT(*) as value
FROM (
  SELECT c.merchant_user_id, COUNT(*) as total_transactions
  FROM subscription_payment_details p 
  JOIN subscription_contract_v2 c ON p.subscription_id = c.subscription_id
  GROUP BY c.merchant_user_id
) merchant_stats
GROUP BY CASE WHEN total_transactions > {threshold} THEN 'More than {threshold} Transactions' ELSE '{threshold} or Fewer Transactions' END
"""
            
            # Apply other fixes...
            sql_query = self._apply_complete_general_sql_optimizations(sql_query)
            
            return sql_query
            
        except Exception as e:
            logger.error(f"Complete SQL fixing failed: {e}")
            return sql_query

    def _verify_and_fix_thresholds(self, sql_query: str, threshold_info: Dict, user_query: str) -> str:
        """Verify and fix threshold numbers in SQL with enhanced accuracy."""
        try:
            # Extract threshold from user query
            user_numbers = threshold_info['numbers']
            if not user_numbers:
                return sql_query
            
            # Find threshold numbers in SQL
            sql_numbers = re.findall(r'>\s*(\d+)|<\s*(\d+)|=\s*(\d+)', sql_query)
            sql_threshold_numbers = []
            for match in sql_numbers:
                for group in match:
                    if group:
                        sql_threshold_numbers.append(int(group))
            
            # Check if SQL uses wrong threshold
            if sql_threshold_numbers and user_numbers:
                expected_threshold = user_numbers[0]  # Use first number found
                actual_threshold = sql_threshold_numbers[0]  # Use first threshold found
                
                if actual_threshold != expected_threshold:
                    logger.warning(f"🔧 Fixing threshold: SQL uses {actual_threshold}, user asked for {expected_threshold}")
                    # Replace the wrong threshold with correct one
                    sql_query = re.sub(rf'>\s*{actual_threshold}\b', f'> {expected_threshold}', sql_query)
                    sql_query = re.sub(rf'<\s*{actual_threshold}\b', f'< {expected_threshold}', sql_query)
                    sql_query = re.sub(rf'=\s*{actual_threshold}\b', f'= {expected_threshold}', sql_query)
                    
                    # Also fix in text labels
                    sql_query = re.sub(rf'More than {actual_threshold}', f'More than {expected_threshold}', sql_query)
                    sql_query = re.sub(rf'{actual_threshold} or Fewer', f'{expected_threshold} or Fewer', sql_query)
                    sql_query = re.sub(rf'{actual_threshold} or Less', f'{expected_threshold} or Less', sql_query)
            
            return sql_query
            
        except Exception as e:
            logger.warning(f"Threshold verification failed: {e}")
            return sql_query

    def _apply_complete_general_sql_optimizations(self, sql_query: str) -> str:
        """Apply complete general SQL optimizations."""
        # Clean quotes safely
        sql_query = sql_query.replace("\\'", "'")  # Remove escaped quotes first
        sql_query = sql_query.replace('\\"', '"')  # Remove escaped double quotes
        
        # Fix quotes more carefully
        sql_query = re.sub(r'"([^"\']*)"', r"'\1'", sql_query)
        
        # Fix status values carefully
        status_values = ['ACTIVE', 'INACTIVE', 'FAILED', 'FAIL', 'INIT', 'CLOSED', 'REJECT']
        for status in status_values:
            # Only fix if not already quoted
            pattern = rf'\bstatus\s*=\s*{status}\b'
            replacement = f"status = '{status}'"
            sql_query = re.sub(pattern, replacement, sql_query, flags=re.IGNORECASE)
        
        # Clean whitespace
        sql_query = re.sub(r'\s+', ' ', sql_query).strip()
        
        return sql_query

    def _get_complete_smart_fallback_tool_call(self, user_query: str, history: List[str]) -> List[Dict]:
        """Get complete smart fallback based on query analysis. For ARPU/metric queries, fallback to a simple ARPU SQL."""
        query_lower = user_query.lower()
        # If a specific date is present, do not fallback to last N days
        date_patterns = [
            r'\d{4}-\d{2}-\d{2}',
            r'\d{2}/\d{2}/\d{4}',
            r'\d{1,2}\s+(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\w*\s+\d{4}',
        ]
        for pattern in date_patterns:
            if re.search(pattern, user_query, re.IGNORECASE):
                return [{
                    'tool': 'execute_dynamic_sql',
                    'parameters': {'sql_query': 'SELECT COUNT(*) as num_subscriptions FROM subscription_contract_v2 WHERE DATE(subcription_start_date) = ...'},
                    'original_query': user_query,
                    'wants_graph': False
                }]
        # ARPU/metric fallback
        metric_keywords = ['arpu', 'average revenue per user', 'average revenue', 'mean revenue', 'arppu', 'arpau', 'total revenue', 'sum', 'average', 'mean']
        if any(k in query_lower for k in metric_keywords):
            # Fallback to simple ARPU SQL for last 90 days
            sql = ("SELECT SUM(p.trans_amount_decimal) / COUNT(DISTINCT c.merchant_user_id) AS ARPU "
                   "FROM subscription_payment_details p "
                   "JOIN subscription_contract_v2 c ON p.subscription_id = c.subscription_id "
                   "WHERE DATE(p.created_date) >= DATE_SUB(CURDATE(), INTERVAL 90 DAY) AND p.status = 'ACTIVE'")
            return [{
                'tool': 'execute_dynamic_sql',
                'parameters': {'sql_query': sql},
                'original_query': user_query,
                'wants_graph': False
            }]
        if 'subscription' in query_lower and any(word in query_lower for word in ['last', 'recent', 'days']):
            return [{
                'tool': 'get_subscriptions_in_last_days',
                'parameters': {'days': 30},
                'original_query': user_query,
                'wants_graph': False
            }]
        elif 'payment' in query_lower and 'rate' in query_lower:
            return [{
                'tool': 'get_payment_success_rate_in_last_days',
                'parameters': {'days': 30},
                'original_query': user_query,
                'wants_graph': False
            }]
        else:
            return [{
                'tool': 'get_database_status',
                'parameters': {},
                'original_query': user_query,
                'wants_graph': False
            }]

    def _fix_sql_quotes(self, sql_query: str) -> str:
        """Replace all double-quoted string literals with single quotes for MySQL compatibility."""
        import re
        # Replace = "SOMETHING" with = 'SOMETHING'
        sql_query = re.sub(r'=\s*"([^"]*)"', r"= '\1'", sql_query)
        # Replace "SOMETHING" in WHERE/IN clauses
        sql_query = re.sub(r'IN\s*\(([^)]*)\)', lambda m: 'IN (' + ', '.join([f"'{x.strip().strip('"')}'" if x.strip().startswith('"') and x.strip().endswith('"') else x for x in m.group(1).split(',')]) + ')', sql_query)
        return sql_query

    def _validate_and_autofix_sql(self, sql_query: str) -> str:
        """Validate and auto-fix common SQL syntax issues: parentheses, IN clauses, dangling literals, and unclosed quotes."""
        import re
        fixed = False
        # 1. Balance parentheses
        open_parens = sql_query.count('(')
        close_parens = sql_query.count(')')
        if open_parens > close_parens:
            sql_query += ')' * (open_parens - close_parens)
            fixed = True
        elif close_parens > open_parens:
            sql_query = '(' * (close_parens - open_parens) + sql_query
            fixed = True
        # 2. Ensure IN (...) clauses are closed
        in_pattern = re.compile(r'IN\s*\(([^)]*)$', re.IGNORECASE)
        match = in_pattern.search(sql_query)
        if match:
            sql_query += ')'
            fixed = True
        # 3. Warn if SQL ends with a dangling string literal
        if re.search(r"'\w+'\s*$", sql_query) and not sql_query.strip().lower().endswith("'as value"):
            logger.warning("SQL ends with a string literal; possible missing clause or context.")
            # Only log, do not print to user
        # 4. Ensure all quotes are closed
        sql_query = self._ensure_closed_quotes(sql_query)
        if fixed:
            logger.warning(f"SQL auto-fixed for syntax: {sql_query}")
            # Only log, do not print to user
        return sql_query

    def _fix_sql_date_math(self, sql_query: str) -> str:
        """Convert SQLite-style date math to MySQL-compatible syntax. Handles both single and double quotes and all common intervals."""
        import re
        # Replace DATE('now', '-N day') or DATE("now", '-N day') with DATE_SUB(CURDATE(), INTERVAL N DAY)
        sql_query = re.sub(r"DATE\(['\"]now['\"],\s*'-?(\d+) day'\)", r"DATE_SUB(CURDATE(), INTERVAL \1 DAY)", sql_query)
        # Replace DATE('now', '-N month') or DATE("now", '-N month') with DATE_SUB(CURDATE(), INTERVAL N MONTH)
        sql_query = re.sub(r"DATE\(['\"]now['\"],\s*'-?(\d+) month'\)", r"DATE_SUB(CURDATE(), INTERVAL \1 MONTH)", sql_query)
        # Replace DATE('now', '-N year') or DATE("now", '-N year') with DATE_SUB(CURDATE(), INTERVAL N YEAR)
        sql_query = re.sub(r"DATE\(['\"]now['\"],\s*'-?(\d+) year'\)", r"DATE_SUB(CURDATE(), INTERVAL \1 YEAR)", sql_query)
        # Replace DATE('now') or DATE("now") with CURDATE()
        sql_query = re.sub(r"DATE\(['\"]now['\"]\)", "CURDATE()", sql_query)
        return sql_query

    def _ensure_closed_quotes(self, sql_query: str) -> str:
        """Ensure all single and double quotes in the SQL query are properly closed."""
        # If odd number of single quotes, append one
        if sql_query.count("'") % 2 != 0:
            sql_query += "'"
        # If odd number of double quotes, append one
        if sql_query.count('"') % 2 != 0:
            sql_query += '"'
        return sql_query

class CompleteEnhancedResultFormatter:
    """COMPLETE enhanced result formatter with smart insights and better presentation."""
    
    def __init__(self):
        self.graph_generator = CompleteGraphGenerator()
    
    def format_single_result(self, result, show_details=False, show_graph=True):
        try:
            output = []
            # Main result
            if result.data and isinstance(result.data, list) and len(result.data) == 1:
                row = result.data[0]
                for k, v in row.items():
                    output.append(f"{k.replace('_', ' ').capitalize()}: {v}")
            elif result.data and isinstance(result.data, list):
                # Table output for breakdowns
                headers = result.data[0].keys()
                output.append(" | ".join(headers))
                output.append("-" * (3 * len(headers)))
                for row in result.data:
                    output.append(" | ".join(str(row[h]) for h in headers))
            else:
                output.append(str(result.data))
            # Only show graph info if show_graph is True
            if show_graph and getattr(result, 'graph_generated', False):
                output.append(f"\n[Graph generated: {getattr(result, 'graph_filepath', '[see file]')}]\n")
            # Show technical details if requested
            if show_details:
                if getattr(result, 'generated_sql', None):
                    output.append(f"\n🔍 Generated SQL:\n{result.generated_sql}")
                if getattr(result, 'error', None):
                    output.append(f"\n❌ Error: {result.error}")
                if getattr(result, 'message', None):
                    output.append(f"\n📝 {result.message}")
            return "\n".join(output)
        except Exception as e:
            return f"❌ Error formatting result: {e}"
    
    def format_multi_result(self, results: List[QueryResult], query: str) -> str:
        """FIXED: Format multiple complete results with MULTITOOL support. Filters out irrelevant/failed results."""
        try:
            if not results:
                return "❌ No results to display"
            output = [f"🎯 MULTITOOL COMPLETE RESULTS FOR: '{query}'", "=" * 80]
            # Filter out irrelevant/failed/None results
            relevant_tools = {'execute_dynamic_sql', 'execute_dynamic_sql_with_graph', 'get_database_status'}
            filtered_results = [r for r in results if r and getattr(r, 'success', False) and r.data is not None and r.tool_used in relevant_tools]
            if not filtered_results:
                return "❌ No successful or relevant results to display. Please try rephrasing your query."
            # Group results by query if they have query_index
            results_by_query = {}
            for result in filtered_results:
                query_index = getattr(result, 'query_index', 1)
                if query_index not in results_by_query:
                    results_by_query[query_index] = []
                results_by_query[query_index].append(result)
            # Display results grouped by query
            for query_index in sorted(results_by_query.keys()):
                query_results = results_by_query[query_index]
                if len(results_by_query) > 1:
                    output.append(f"\n--- Query {query_index} Results ---")
                for i, result in enumerate(query_results):
                    if len(query_results) > 1:
                        output.append(f"\n  -- Result {i+1} --")
                    single_result = self.format_single_result(result)
                    if len(results_by_query) > 1:
                        single_result = '\n'.join(f"  {line}" for line in single_result.split('\n'))
                    output.append(single_result)
            # Summary
            total_success = len(filtered_results)
            total_results = len(results)
            output.append(f"\n📊 MULTITOOL Summary: {total_success}/{total_results} successful and relevant results")
            output.append(f"🔗 Processed {len(results_by_query)} individual queries")
            return "\n".join(output)
        except Exception as e:
            logger.error(f"Error formatting MULTITOOL results: {e}")
            return f"❌ Error formatting MULTITOOL results: {str(e)}"
    
    def _format_complete_graph_info(self, result: QueryResult) -> str:
        """Format complete graph information with enhanced details."""
        try:
            graph_data = result.graph_data
            graph_type = graph_data.get('graph_type', 'unknown').title()
            title = graph_data.get('title', 'Complete Data Visualization')
            
            output = [
                "📊 COMPLETE GRAPH GENERATED",
                "=" * 50,
                f"Graph Type: {graph_type}",
                f"Title: {title}"
            ]
            
            if hasattr(result, 'graph_filepath') and result.graph_filepath:
                output.append(f"Saved to: {result.graph_filepath}")
                output.append("🎨 Graph opened in default image viewer")
            
            # Add complete data summary
            data_summary = graph_data.get('data_summary', {})
            if data_summary:
                output.append(f"Data Points: {data_summary.get('total_rows', 'N/A')}")
                output.append(f"Columns: {', '.join(data_summary.get('columns', []))}")
                output.append(f"Chart Optimization: {data_summary.get('chart_optimization', 'N/A')}")
            
            return "\n".join(output)
        except Exception as e:
            logger.error(f"Error formatting complete graph info: {e}")
            return "📊 Complete graph generated (formatting error)"
    
    def _format_complete_main_data(self, result: QueryResult) -> str:
        """Format main data with complete enhanced presentation."""
        try:
            is_dynamic = result.tool_used in ['execute_dynamic_sql', 'execute_dynamic_sql_with_graph']
            
            header = f"📊 COMPLETE DYNAMIC QUERY RESULT" if is_dynamic else f"📊 COMPLETE RESULT FROM TOOL: {result.tool_used.upper()}"
            output = [header, "=" * len(header)]
            
            if isinstance(result.data, list) and len(result.data) > 0:
                table_output = self._format_complete_enhanced_table(result.data)
                output.extend(table_output)
                
                # Add complete intelligent insights
                insights = self._generate_complete_smart_insights(result.data, result.original_query)
                if insights:
                    output.extend(insights)
                    
            elif isinstance(result.data, dict):
                output.extend(self._format_complete_dict_data(result.data))
            else:
                output.append(str(result.data))
            
            return "\n".join(output)
            
        except Exception as e:
            logger.error(f"Error formatting complete main data: {e}")
            return f"Error formatting complete data: {str(e)}"
    
    def _format_complete_enhanced_table(self, data: List[Dict]) -> List[str]:
        """Format table with complete enhanced styling and smart column handling."""
        try:
            if not data:
                return ["No data to display"]
            
            headers = list(data[0].keys())
            if not headers:
                return ["No columns to display"]
            
            # Complete smart column width calculation
            col_widths = self._calculate_complete_smart_column_widths(data, headers)
            
            output = []
            
            # Create header
            header_line = " | ".join(h.ljust(col_widths[h]) for h in headers)
            output.append(header_line)
            output.append("-" * len(header_line))
            
            # Add data rows (limit for readability)
            display_rows = data[:50]
            for row in display_rows:
                formatted_row = []
                for h in headers:
                    val = self._format_complete_cell_value(row.get(h), col_widths[h])
                    formatted_row.append(val.ljust(col_widths[h]))
                output.append(" | ".join(formatted_row))
            
            # Add complete summary
            output.append("")
            total_rows = len(data)
            if total_rows > 50:
                output.append(f"📈 Showing 50 of {total_rows} total rows")
            else:
                output.append(f"📈 Total rows: {total_rows}")
            
            return output
            
        except Exception as e:
            logger.error(f"Error formatting complete table: {e}")
            return [f"Error formatting complete table: {str(e)}"]
    
    def _calculate_complete_smart_column_widths(self, data: List[Dict], headers: List[str]) -> Dict[str, int]:
        """Calculate complete smart column widths based on content."""
        col_widths = {}
        
        for h in headers:
            try:
                # Start with header length
                max_width = len(str(h))
                
                # Check first 20 rows for content width
                for row in data[:20]:
                    val_len = len(self._format_complete_cell_value(row.get(h), 50))
                    max_width = max(max_width, val_len)
                
                # Set reasonable bounds
                col_widths[h] = max(10, min(max_width, 30))
                
            except Exception:
                col_widths[h] = 15  # Default width
        
        return col_widths
    
    def _format_complete_cell_value(self, value, max_width: int) -> str:
        """Format individual cell values with complete smart truncation."""
        try:
            if value is None:
                return ""
            
            str_val = str(value)
            
            # Handle different value types
            if isinstance(value, float):
                str_val = f"{value:.1f}" if value != int(value) else str(int(value))
            
            # Truncate if too long
            if len(str_val) > max_width:
                str_val = str_val[:max_width-3] + "..."
            
            return str_val
            
        except Exception:
            return str(value)[:max_width] if value else ""
    
    def _format_complete_dict_data(self, data: Dict) -> List[str]:
        """Format dictionary data with complete enhanced presentation."""
        try:
            output = []
            for key, value in data.items():
                try:
                    if isinstance(value, (dict, list)):
                        output.append(f"{key}:")
                        output.append(f"  {json.dumps(value, indent=2, default=str)}")
                    else:
                        output.append(f"{key}: {value}")
                except Exception:
                    output.append(f"{key}: [Error displaying value]")
            return output
        except Exception as e:
            return [f"Error formatting complete dictionary: {str(e)}"]
    
    def _generate_complete_smart_insights(self, data: List[Dict], original_query: str) -> List[str]:
        """Generate complete smart insights about the data."""
        try:
            if not data or len(data) < 2:
                return []
            
            insights = []
            headers = list(data[0].keys())
            query_lower = original_query.lower() if original_query else ""
            
            # Identify numeric columns
            numeric_cols = self._identify_complete_numeric_columns(data, headers)
            date_cols = self._identify_complete_date_columns(data, headers)
            
            # Generate complete dataset insights
            if len(data) > 100:
                insights.append(f"\n💡 **Large Dataset**: {len(data)} records - consider filtering for specific analysis")
            
            # Generate complete time series insights
            if date_cols and len(data) > 10:
                insights.append(f"\n📅 **Time Series Data**: Consider grouping by day/week/month for trend analysis")
            
            # Generate complete numeric insights
            for col in numeric_cols[:2]:  # Limit to first 2 numeric columns
                try:
                    values = [float(row.get(col, 0)) for row in data 
                             if row.get(col) is not None and isinstance(row.get(col), (int, float))]
                    if values:
                        avg_val = sum(values) / len(values)
                        max_val = max(values)
                        min_val = min(values)
                        if 'rate' in col.lower() or 'percent' in col.lower():
                            insights.append(f"\n📊 **{col}**: Average: {avg_val:.1f}%, Range: {min_val:.1f}% - {max_val:.1f}%")
                        else:
                            insights.append(f"\n📊 **{col}**: Average: {avg_val:.1f}, Maximum: {max_val}")
                except Exception:
                    continue
            
            # Complete query-specific insights
            if 'success' in query_lower and 'rate' in query_lower:
                success_data = [row for row in data if 'success' in str(row).lower()]
                if success_data:
                    insights.append(f"\n🎯 **Success Analysis**: Found {len(success_data)} success-related data points")
            
            if 'merchant' in query_lower:
                merchant_data = [row for row in data if any('merchant' in str(v).lower() for v in row.values())]
                if merchant_data:
                    insights.append(f"\n🏪 **Merchant Analysis**: Found {len(merchant_data)} merchant-related data points")
            
            return insights
            
        except Exception as e:
            logger.debug(f"Could not generate complete insights: {e}")
            return []

    def _identify_complete_numeric_columns(self, data: List[Dict], headers: List[str]) -> List[str]:
        """Identify numeric columns in the data with complete logic."""
        numeric_cols = []
        
        for col in headers:
            try:
                # Check first 5 values
                sample_values = [row.get(col) for row in data[:5] if row.get(col) is not None]
                if sample_values:
                    numeric_count = sum(1 for val in sample_values if isinstance(val, (int, float)))
                    if numeric_count >= len(sample_values) * 0.8:  # 80% numeric
                        numeric_cols.append(col)
            except Exception:
                continue
        
        return numeric_cols

    def _identify_complete_date_columns(self, data: List[Dict], headers: List[str]) -> List[str]:
        """Identify date columns in the data with complete logic."""
        date_cols = []
        
        for col in headers:
            if 'date' in col.lower() or 'time' in col.lower():
                date_cols.append(col)
        
        return date_cols

# Complete Enhanced Universal Client with all fixes
class CompleteEnhancedUniversalClient:
    """COMPLETE universal client with full semantic learning and proper schema handling."""
    
    def __init__(self, config: dict):
        self.config = config
        self.nlp = CompleteSmartNLPProcessor()
        self.session = None
        self.formatter = CompleteEnhancedResultFormatter()
        self.history = []
        self.graph_generator = CompleteGraphGenerator()
        self.max_history_length = 8  # Increased for better context
        self.ssl_disabled = False
        self.context = {}  # Store key results for context awareness

    async def __aenter__(self):
        try:
            try:
                connector = aiohttp.TCPConnector(
                    ssl=ssl.create_default_context(),
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
                return self
            except Exception as ssl_error:
                print("⚠️ SSL verification failed, retrying with SSL disabled (not secure)...")
                connector = aiohttp.TCPConnector(ssl=False)
                self.session = aiohttp.ClientSession(
                    connector=connector,
                    timeout=aiohttp.ClientTimeout(total=120),
                    headers={'Connection': 'close'}
                )
                self.ssl_disabled = True
                return self
        except Exception as e:
            logger.error(f"Failed to initialize complete client: {e}")
            raise

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        try:
            if self.session:
                await self.session.close()
        except Exception as e:
            logger.warning(f"Error closing complete session: {e}")

    async def call_tool(self, tool_name: str, parameters: Dict = None, original_query: str = "", wants_graph: bool = False) -> QueryResult:
        """Complete enhanced tool calling with smart graph handling."""
        if tool_name == 'execute_dynamic_sql_with_graph':
            return await self._handle_complete_smart_sql_with_graph(parameters, original_query)
        
        headers = {
            "Authorization": f"Bearer {self.config['API_KEY_1']}",
            "Connection": "close"
        }
        payload = {"tool_name": tool_name, "parameters": parameters or {}}
        server_url = self.config['SUBSCRIPTION_API_URL']
        max_retries = 3
        
        for attempt in range(max_retries):
            try:
                async with self.session.post(f"{server_url}/execute", json=payload, headers=headers) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        if attempt < max_retries - 1:
                            logger.warning(f"HTTP {response.status} on attempt {attempt + 1}, retrying...")
                            await asyncio.sleep(1)
                            continue
                        return QueryResult(
                            success=False,
                            error=f"HTTP {response.status}: {error_text}",
                            tool_used=tool_name
                        )
                    
                    result_data = await response.json()
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
            except ssl.SSLError as ssl_error:
                if not self.ssl_disabled:
                    print("⚠️ SSL error encountered, retrying with SSL disabled (not secure)...")
                    connector = aiohttp.TCPConnector(ssl=False)
                    self.session = aiohttp.ClientSession(
                        connector=connector,
                        timeout=aiohttp.ClientTimeout(total=120),
                        headers={'Connection': 'close'}
                    )
                    self.ssl_disabled = True
                    continue
                else:
                    return QueryResult(success=False, error=f"SSL error: {ssl_error}", tool_used=tool_name)
            except Exception as e:
                if attempt < max_retries - 1:
                    logger.warning(f"Complete attempt {attempt + 1} failed: {e}, retrying...")
                    await asyncio.sleep(1)
                    continue
                else:
                    return QueryResult(
                        success=False,
                        error=f"Complete connection error after {max_retries} attempts: {str(e)}",
                        tool_used=tool_name
                    )
        
        return QueryResult(
            success=False,
            error="All complete retry attempts failed",
            tool_used=tool_name
        )

    async def _handle_complete_smart_sql_with_graph(self, parameters: Dict, original_query: str) -> QueryResult:
        """Complete smart SQL with graph generation handling."""
        try:
            # Execute SQL first
            sql_result = await self.call_tool('execute_dynamic_sql', {
                'sql_query': parameters['sql_query']
            }, original_query)
            
            if not sql_result.success or not sql_result.data:
                if sql_result.success:
                    sql_result.message = (sql_result.message or "") + "\n💡 No data returned - cannot generate complete graph"
                return sql_result
            
            logger.info(f"📊 Complete SQL returned {len(sql_result.data)} rows for graph analysis")
            
            if not self.graph_generator.can_generate_graphs():
                sql_result.message = (sql_result.message or "") + "\n⚠️ Complete graph generation unavailable (matplotlib not installed)"
                return sql_result
            
            # Generate graph with complete enhanced handling
            try:
                graph_config = {
                    'data': sql_result.data,
                    'graph_type': parameters.get('graph_type', 'bar'),
                    'custom_config': {
                        'title': self._generate_complete_smart_title(original_query),
                        'description': f"Generated from: {original_query}"
                    }
                }
                
                graph_result = await self.call_tool('generate_graph_data', graph_config)
                
                if graph_result.success and graph_result.data:
                    graph_filepath = self.graph_generator.generate_graph(
                        graph_result.data, original_query
                    )
                    
                    sql_result.graph_data = graph_result.data
                    sql_result.graph_generated = graph_filepath is not None
                    
                    if graph_filepath:
                        sql_result.graph_filepath = graph_filepath
                        sql_result.message = (sql_result.message or "") + f"\n📊 Complete graph generated successfully"
                    else:
                        sql_result.message = (sql_result.message or "") + f"\n⚠️ Complete graph data generated but file creation failed"
                else:
                    error_msg = graph_result.error if graph_result.error else "Complete graph generation service error"
                    sql_result.message = (sql_result.message or "") + f"\n⚠️ Complete graph generation failed: {error_msg}"
            
            except Exception as graph_error:
                logger.error(f"Complete graph generation error: {graph_error}")
                sql_result.message = (sql_result.message or "") + f"\n⚠️ Complete graph generation failed: {str(graph_error)}"
            
            return sql_result
            
        except Exception as e:
            logger.error(f"Error in complete smart SQL with graph: {e}")
            # Fallback to regular SQL
            return await self.call_tool('execute_dynamic_sql', {
                'sql_query': parameters['sql_query']
            }, original_query)

    def _generate_complete_smart_title(self, query: str) -> str:
        """Generate complete smart title from query."""
        try:
            query_lower = query.lower()
            
            if 'success' in query_lower and 'rate' in query_lower:
                return "Complete Payment Success Analysis"
            elif 'merchant' in query_lower and 'transaction' in query_lower:
                return "Complete Merchant Transaction Analysis"
            elif 'trend' in query_lower:
                return "Complete Trend Analysis"
            elif 'pie' in query_lower or 'distribution' in query_lower:
                return "Complete Distribution Analysis"
            elif 'payment' in query_lower:
                return "Complete Payment Analysis"
            elif 'subscription' in query_lower:
                return "Complete Subscription Analysis"
            else:
                return "Complete Data Analysis"
        except Exception:
            return "Complete Analysis"

    async def query(self, user_query: str) -> Union[QueryResult, List[QueryResult]]:
        """Complete enhanced query processing with smart AI and MULTITOOL support. Includes post-processing for metric queries."""
        try:
            parsed_calls = await self.nlp.parse_query(user_query, self.history, client=self)
            if len(parsed_calls) > 1:
                results = []
                for call in parsed_calls:
                    try:
                        result = await self.call_tool(
                            call['tool'], 
                            call['parameters'], 
                            call['original_query'],
                            call.get('wants_graph', False)
                        )
                        # Add multitool metadata
                        result.query_index = call.get('query_index', 1)
                        result.total_queries = call.get('total_queries', len(parsed_calls))
                        result.is_multitool = call.get('is_multitool', True)
                        results.append(result)
                    except Exception as e:
                        logger.error(f"Error calling complete tool {call['tool']}: {e}")
                        error_result = QueryResult(
                            success=False,
                            error=f"Complete tool {call['tool']} failed: {str(e)}",
                            tool_used=call['tool']
                        )
                        error_result.query_index = call.get('query_index', 1)
                        error_result.total_queries = call.get('total_queries', len(parsed_calls))
                        error_result.is_multitool = call.get('is_multitool', True)
                        results.append(error_result)
                # Post-process for metric queries: if user asked for ARPU/metric and got a breakdown, retry with explicit prompt
                metric_keywords = ['arpu', 'average revenue per user', 'average revenue', 'mean revenue', 'arppu', 'arpau', 'total revenue', 'sum', 'average', 'mean']
                user_query_lower = user_query.lower()
                is_metric_query = any(k in user_query_lower for k in metric_keywords)
                if is_metric_query:
                    for result in results:
                        if result.data and isinstance(result.data, list) and len(result.data) > 1 and any('category' in row for row in result.data if isinstance(row, dict)):
                            logger.info("Detected breakdown for metric query; retrying with explicit single-value prompt.")
                            explicit_query = user_query + " (Return only a single ARPU value, not a breakdown or category table.)"
                            return await self.query(explicit_query)
                return results
            else:
                call = parsed_calls[0]
                result = await self.call_tool(
                    call['tool'], 
                    call['parameters'], 
                    call['original_query'],
                    call.get('wants_graph', False)
                )
                # Add single query metadata
                result.query_index = 1
                result.total_queries = 1
                result.is_multitool = False
                # Post-process for metric queries: if user asked for ARPU/metric and got a breakdown, retry with explicit prompt
                metric_keywords = ['arpu', 'average revenue per user', 'average revenue', 'mean revenue', 'arppu', 'arpau', 'total revenue', 'sum', 'average', 'mean']
                user_query_lower = user_query.lower()
                is_metric_query = any(k in user_query_lower for k in metric_keywords)
                if is_metric_query:
                    if result.data and isinstance(result.data, list) and len(result.data) > 1 and any('category' in row for row in result.data if isinstance(row, dict)):
                        logger.info("Detected breakdown for metric query; retrying with explicit single-value prompt.")
                        explicit_query = user_query + " (Return only a single ARPU value, not a breakdown or category table.)"
                        return await self.query(explicit_query)
                return result
        except Exception as e:
            logger.error(f"Error in complete query processing: {e}", exc_info=True)
            return QueryResult(
                success=False,
                error=f"Complete query processing failed: {e}",
                tool_used="complete_query_processor"
            )

    def manage_history(self, query: str, response: str):
        """Complete enhanced history management with smart filtering."""
        try:
            self.history.extend([f"User: {query}", f"Assistant: {response[:200]}..."])  # Truncate long responses
            self.history = self.history[-self.max_history_length:]  # Keep configurable amount
        except Exception as e:
            logger.warning(f"Error managing complete history: {e}")

    async def submit_feedback(self, result: QueryResult, helpful: bool, improvement_suggestion: str = None):
        """Complete enhanced feedback submission with better error handling."""
        if result.is_dynamic and result.generated_sql and result.original_query:
            try:
                feedback_params = {
                    'original_question': result.original_query,
                    'sql_query': result.generated_sql,
                    'was_helpful': helpful
                }
                
                if not helpful and improvement_suggestion:
                    feedback_params['improvement_suggestion'] = improvement_suggestion.strip()
                
                feedback_result = await self.call_tool('record_query_feedback', feedback_params)
                
                if feedback_result.success and feedback_result.message:
                    print(f"✅ {feedback_result.message}")
                else:
                    print(f"✅ Complete feedback recorded successfully")
                    
            except Exception as e:
                logger.warning(f"Could not submit complete feedback: {str(e)}")
                print("⚠️ Complete feedback noted locally")

# Complete Enhanced Interactive Mode
async def complete_enhanced_interactive_mode():
    """COMPLETE interactive mode with full functionality."""
    print("✨ COMPLETE Subscription Analytics AI Agent ✨")
    print("=" * 70)
    
    config_manager = ConfigManager()
    
    try:
        user_config = config_manager.get_config()
        genai.configure(api_key=user_config['GOOGLE_API_KEY'])
        
        print(f"🔗 Connected to server: {user_config['SUBSCRIPTION_API_URL']}")
        print("🧠 COMPLETE Smart AI with full semantic learning!")
        print("🛡️ Production-ready error handling and recovery!")
        print("🔧 COMPLETE SQL generation with perfect schema handling!")
        print("📊 COMPLETE semantic learning with feedback system!")
        print("🎯 COMPLETE chart type awareness and learning!")
        print("🔗 MULTITOOL functionality for complex queries!")
        
        if MATPLOTLIB_AVAILABLE:
            print("📈 COMPLETE advanced graph generation with smart type detection")
        else:
            print("⚠️ Graph generation disabled (install matplotlib: pip install matplotlib)")
        
        async with CompleteEnhancedUniversalClient(config=user_config) as client:
            print("\n💬 Enter questions in natural language. Type 'quit' to exit.")
            print("\n📚 COMPLETE Examples:")
            print("  • Show me a pie chart of payment success rates")
            print("  • Give me subscription trends as a line chart")
            print("  • Visualize payment data with a bar chart")
            print("  • Create a pie chart breakdown of successful vs failed payments")
            print("  • Show payment success rate for merchants with more than 1 transaction visually")
            print("  • Show payment trends over time")
            print("  • Tell me the last date for which data is available")
            print("  • Compare subscribers with more than 1 and more than 2 subscriptions")
            print("  • Multiple queries: Get database status and show recent payment summary")
            print("\n💡 The COMPLETE AI has semantic learning and MULTITOOL support!")
            print("=" * 70)
            
            while True:
                try:
                    query = input("\n> ").strip()
                    if query.lower() in ['quit', 'exit', 'q']:
                        break
                    if not query:
                        continue
                    
                    print("🤔 Processing your query with COMPLETE AI...")
                    result = await client.query(query)
                    
                    try:
                        if isinstance(result, list):
                            output = client.formatter.format_multi_result(result, query)
                        else:
                            output = client.formatter.format_single_result(result)
                        
                        print(f"\n{output}")
                        
                        client.manage_history(query, output)
                        
                        # COMPLETE enhanced feedback for dynamic queries
                        if isinstance(result, list):
                            # Handle multitool feedback
                            for i, individual_result in enumerate(result, 1):
                                if (individual_result.is_dynamic and 
                                    individual_result.success and 
                                    individual_result.data is not None):
                                    
                                    print(f"\n" + "="*50)
                                    print(f"📝 Result {i} was generated using COMPLETE AI with semantic learning!")
                                    print("Your feedback helps the system learn and improve over time.")
                                    
                                    while True:
                                        try:
                                            feedback_input = input(f"Was Result {i} helpful? (y/n/skip): ").lower().strip()
                                            if feedback_input in ['y', 'yes']:
                                                await client.submit_feedback(individual_result, True)
                                                print("🧠 Positive feedback recorded in semantic learning system!")
                                                print_section('💡 Your feedback will be used to improve future answers to similar queries.')
                                                break
                                            elif feedback_input in ['n', 'no']:
                                                improvement = input("How can this be improved? (e.g., 'use pie chart instead', 'fix SQL error'): ").strip()
                                                if improvement.lower() not in ['skip', 's', '']:
                                                    await client.submit_feedback(individual_result, False, improvement)
                                                    print("🧠 Negative feedback and improvement recorded - the system will learn!")
                                                    print_section('💡 Your feedback will be used to improve future answers to similar queries.')
                                                else:
                                                    await client.submit_feedback(individual_result, False)
                                                    print("🧠 Negative feedback recorded!")
                                                    print_section('💡 Your feedback will be used to improve future answers to similar queries.')
                                                break
                                            elif feedback_input in ['s', 'skip', '']:
                                                break
                                            else:
                                                print("Please enter 'y', 'n', or 'skip'.")
                                        except (KeyboardInterrupt, EOFError):
                                            break
                        else:
                            # Handle single result feedback
                            if (result.is_dynamic and 
                                result.success and 
                                result.data is not None):
                                
                                print("\n" + "="*50)
                                print("📝 This answer was generated using COMPLETE AI with semantic learning!")
                                print("Your feedback helps the system learn and improve over time.")
                                
                                while True:
                                    try:
                                        feedback_input = input("Was this helpful? (y/n/skip): ").lower().strip()
                                        if feedback_input in ['y', 'yes']:
                                            await client.submit_feedback(result, True)
                                            print("🧠 Positive feedback recorded in semantic learning system!")
                                            print_section('💡 Your feedback will be used to improve future answers to similar queries.')
                                            break
                                        elif feedback_input in ['n', 'no']:
                                            improvement = input("How can this be improved? (e.g., 'use pie chart instead', 'fix SQL error'): ").strip()
                                            if improvement.lower() not in ['skip', 's', '']:
                                                await client.submit_feedback(result, False, improvement)
                                                print("🧠 Negative feedback and improvement recorded - the system will learn!")
                                                print_section('💡 Your feedback will be used to improve future answers to similar queries.')
                                            else:
                                                await client.submit_feedback(result, False)
                                                print("🧠 Negative feedback recorded!")
                                                print_section('💡 Your feedback will be used to improve future answers to similar queries.')
                                            break
                                        elif feedback_input in ['s', 'skip', '']:
                                            break
                                        else:
                                            print("Please enter 'y', 'n', or 'skip'.")
                                    except (KeyboardInterrupt, EOFError):
                                        break
                    except Exception as format_error:
                        logger.error(f"Error formatting COMPLETE output: {format_error}")
                        print(f"❌ Error displaying COMPLETE results: {format_error}")
                        if isinstance(result, QueryResult):
                            print(f"Raw result: Success={result.success}, Error={result.error}")
                        elif isinstance(result, list):
                            print(f"Raw results: {len(result)} results, {sum(1 for r in result if r.success)} successful")
                
                except (KeyboardInterrupt, EOFError):
                    break
                except Exception as e:
                    logger.error("Error in COMPLETE interactive loop", exc_info=True)
                    print(f"❌ Error: {e}")
                    print("💡 The COMPLETE system will continue - try a different query")
                    
    except Exception as e:  
        logger.error("COMPLETE client failed to initialize", exc_info=True)
        print(f"❌ Critical Error: {e}")
        print("\n🔧 Troubleshooting:")
        print("1. Check your internet connection")
        print("2. Verify your API keys in config.json")
        print("3. Ensure the server is running")
        if not MATPLOTLIB_AVAILABLE:
            print("4. For COMPLETE graphs: pip install matplotlib")
        print("5. For COMPLETE semantic learning: pip install sentence-transformers faiss-cpu")
    
    print("\n👋 Goodbye from the COMPLETE system!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Subscription Analytics Universal Client")
    parser.add_argument('query', nargs='?', help='Natural language query to run')
    parser.add_argument('--test-connection', action='store_true', help='Test API server connection and credentials')
    parser.add_argument('--show-config', action='store_true', help='Show current config (without API keys)')
    parser.add_argument('--show-server-status', action='store_true', help='Show current server status')
    parser.add_argument('--show-gemini-key', action='store_true', help='Show Gemini API key status (masked)')
    parser.add_argument('--show-details', action='store_true', help='Show technical/log details (SQL, tool info, etc.)')
    args = parser.parse_args()
    
    config_manager = ConfigManager()
    config = config_manager.get_config()
    print(f"✅ Config loaded from: {config_manager.config_path}")
    print(f"✅ Server URL: {config.get('SUBSCRIPTION_API_URL', '[not set]')}")
    
    # Gemini API key check
    gemini_key = config.get('GOOGLE_API_KEY', '')
    if not gemini_key or not isinstance(gemini_key, str) or len(gemini_key) < 20 or not gemini_key.startswith('AI'):
        print("❌ Gemini API key is missing or invalid!")
        print("   Please set it in client/config.json as 'GOOGLE_API_KEY' or export GOOGLE_API_KEY=your_key")
        print("   Get your key from https://ai.google.dev/")
        sys.exit(1)
    
    import google.generativeai as genai
    try:
        genai.configure(api_key=gemini_key)
    except Exception as e:
        print(f"❌ Failed to configure Gemini API: {e}")
        sys.exit(1)
    
    if args.show_config:
        safe_config = {k: ('***' if 'key' in k.lower() else v) for k, v in config.items()}
        print(json.dumps(safe_config, indent=2))
        sys.exit(0)
    
    if args.show_server_status:
        print("Checking server status...")
        import requests
        try:
            resp = requests.get(config['SUBSCRIPTION_API_URL'] + '/health', timeout=10)
            print(f"Server status: {resp.status_code} {resp.text}")
        except Exception as e:
            print(f"❌ Could not reach server: {e}")
        sys.exit(0)
    
    if args.show_gemini_key:
        print(f"Gemini API key: {'*' * (len(gemini_key) - 4) + gemini_key[-4:] if gemini_key else '[not set]'}")
        sys.exit(0)
    
    if args.test_connection:
        print("Testing API connection...")
        import requests
        try:
            resp = requests.get(config['SUBSCRIPTION_API_URL'] + '/health', timeout=10)
            if resp.status_code == 200:
                print("✅ API server is reachable and healthy.")
            else:
                print(f"❌ API server returned status {resp.status_code}: {resp.text}")
        except Exception as e:
            print(f"❌ Could not connect to API server: {e}")
        sys.exit(0)

    EXAMPLE_QUERIES = [
        "Visualize payment data with a bar chart",
        "Create a pie chart breakdown of successful vs failed payments",
        "Show payment success rate for merchants with more than 1 transaction visually",
        "Show payment trends over time",
        "Tell me the last date for which data is available",
        "Compare subscribers with more than 1 and more than 2 subscriptions",
        "Number of subscriptions on 24 april 2025",
        "Show me database status and recent subscription summary",
        "How many new subscriptions did we get this month?",
        "Multiple: Get database status; show payment success rates; create pie chart",
    ]

    def print_example_queries():
        print("\n💡 Example queries:")
        for q in EXAMPLE_QUERIES:
            print(f"  • {q}")

    async def run_query_loop():
        print_header("✨ COMPLETE Subscription Analytics AI Agent ✨")
        print_section("Welcome! Type your questions below. Type 'exit' to quit.")
        async with CompleteEnhancedUniversalClient(config) as client:
            while True:
                try:
                    user_query = args.query
                    if not user_query:
                        print_section("Example queries:")
                        for q in EXAMPLE_QUERIES:
                            print(f"  {CYAN}•{RESET} {q}")
                        print_separator()
                        try:
                            user_query = input(f"{BOLD}{YELLOW}\nEnter your query (or 'exit' to quit): {RESET}").strip()
                        except (KeyboardInterrupt, EOFError):
                            print_success("\n👋 Goodbye from COMPLETE system!")
                            break
                        if user_query.lower() in ['exit', 'quit', 'q']:
                            print_success("\n👋 Goodbye from COMPLETE system!")
                            break
                        if not user_query:
                            continue
                    # Always reload improvement suggestions before each query for immediate feedback effect
                    if hasattr(client.nlp, '_last_best_chart_type'):
                        del client.nlp._last_best_chart_type
                    # Context-aware query resolution
                    resolved_query = user_query
                    if 'that day' in user_query.lower() and 'last_date' in client.context:
                        resolved_query = user_query.lower().replace('that day', client.context['last_date'])
                    if 'that merchant' in user_query.lower() and 'last_merchant' in client.context:
                        resolved_query = user_query.lower().replace('that merchant', client.context['last_merchant'])
                    if 'that count' in user_query.lower() and 'last_count' in client.context:
                        resolved_query = user_query.lower().replace('that count', str(client.context['last_count']))
                    # Process query (handles both single and multitool automatically)
                    result = await client.query(resolved_query)
                    print_separator()
                    # Format and display results
                    if isinstance(result, list):
                        print_header(f"MULTITOOL RESULTS FOR: '{resolved_query}'")
                        output = client.formatter.format_multi_result(result, resolved_query)
                        print(f"{output}")
                        print_separator()
                        # Update context with key results from all queries
                        for individual_result in result:
                            if individual_result.data and isinstance(individual_result.data, list) and len(individual_result.data) > 0:
                                row = individual_result.data[0]
                                if isinstance(row, dict):
                                    for k, v in row.items():
                                        if 'date' in k.lower():
                                            client.context['last_date'] = str(v)
                                        if 'merchant' in k.lower():
                                            client.context['last_merchant'] = str(v)
                                        if 'count' in k.lower() or 'num' in k.lower():
                                            client.context['last_count'] = v
                        # Feedback for multitool results
                        for i, individual_result in enumerate(result, 1):
                            if (getattr(individual_result, 'is_dynamic', False) and 
                                getattr(individual_result, 'success', False) and 
                                getattr(individual_result, 'data', None) is not None):
                                print_feedback_prompt(f"\n📝 Feedback for Query {i} (generated by AI):")
                                print_feedback_prompt("Your feedback helps the system learn and improve over time.")
                                feedback_given = False
                                while not feedback_given:
                                    try:
                                        feedback_input = input(f"{MAGENTA}Was Query {i} helpful? (y/n/skip): {RESET}").lower().strip()
                                        if feedback_input in ['y', 'yes']:
                                            await client.submit_feedback(individual_result, True)
                                            print_success("🧠 Positive feedback recorded in semantic learning system!")
                                            print_section('💡 Your feedback will be used to improve future answers to similar queries.')
                                            feedback_given = True
                                        elif feedback_input in ['n', 'no']:
                                            improvement = input(f"{MAGENTA}How can this be improved? (e.g., 'use bar chart instead', 'fix SQL error'): {RESET}").strip()
                                            if improvement.lower() not in ['skip', 's', '']:
                                                await client.submit_feedback(individual_result, False, improvement)
                                                print_success("🧠 Negative feedback and improvement recorded - the system will learn!")
                                                print_section('💡 Your feedback will be used to improve future answers to similar queries.')
                                            else:
                                                await client.submit_feedback(individual_result, False)
                                                print_success("🧠 Negative feedback recorded!")
                                                print_section('💡 Your feedback will be used to improve future answers to similar queries.')
                                            feedback_given = True
                                        elif feedback_input in ['s', 'skip', '']:
                                            feedback_given = True
                                        else:
                                            print_warning("Please enter 'y', 'n', or 'skip'.")
                                    except (KeyboardInterrupt, EOFError):
                                        feedback_given = True
                                print_separator()
                    else:
                        print_header("RESULT")
                        output = client.formatter.format_single_result(result, show_details=args.show_details)
                        print(f"{output}")
                        print_separator()
                        # Update context with key results
                        if result.data and isinstance(result.data, list) and len(result.data) > 0:
                            row = result.data[0]
                            if isinstance(row, dict):
                                for k, v in row.items():
                                    if 'date' in k.lower():
                                        client.context['last_date'] = str(v)
                                    if 'merchant' in k.lower():
                                        client.context['last_merchant'] = str(v)
                                    if 'count' in k.lower() or 'num' in k.lower():
                                        client.context['last_count'] = v
                        # Feedback for single result
                        if True:
                            print_feedback_prompt("\n📝 This answer was generated using COMPLETE AI with semantic learning!")
                            print_feedback_prompt("Your feedback helps the system learn and improve over time.")
                            while True:
                                try:
                                    feedback_input = input(f"{MAGENTA}Was this helpful? (y/n/skip): {RESET}").lower().strip()
                                    if feedback_input in ['y', 'yes']:
                                        await client.submit_feedback(result, True)
                                        print("🧠 Positive feedback recorded in semantic learning system!")
                                        print_section('💡 Your feedback will be used to improve future answers to similar queries.')
                                        break
                                    elif feedback_input in ['n', 'no']:
                                        improvement = input(f"{MAGENTA}How can this be improved? (e.g., 'use pie chart instead', 'fix SQL error'): {RESET}").strip()
                                        if improvement.lower() not in ['skip', 's', '']:
                                            await client.submit_feedback(result, False, improvement)
                                            print("🧠 Negative feedback and improvement recorded - the system will learn!")
                                            print_section('💡 Your feedback will be used to improve future answers to similar queries.')
                                        else:
                                            await client.submit_feedback(result, False)
                                            print("🧠 Negative feedback recorded!")
                                            print_section('💡 Your feedback will be used to improve future answers to similar queries.')
                                        break
                                    elif feedback_input in ['s', 'skip', '']:
                                        break
                                    else:
                                        print("Please enter 'y', 'n', or 'skip'.")
                                except (KeyboardInterrupt, EOFError):
                                    break
                    client.manage_history(user_query, output)
                except Exception as e:
                    print_error(f"❌ Error running query: {e}")
                    logger.error(f"Query error: {e}", exc_info=True)
                args.query = None  # After first run, always prompt interactively

    # Run interactive mode or single query
    import asyncio
    asyncio.run(run_query_loop())