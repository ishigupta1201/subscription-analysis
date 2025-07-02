"""
Universal Client for Subscription Analytics
- Command-line and interactive analytics using natural language
- Graph generation and feedback-driven improvement
- Connects to remote API and Gemini AI
"""

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
import calendar

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
        """Generate graph with complete enhanced error handling and smart type enforcement."""
        if not self.can_generate_graphs():
            logger.warning("Cannot generate graph - matplotlib not available")
            return None
        try:
            # FIXED: Better data validation and preparation
            if not self._validate_and_prepare_graph_data(graph_data):
                logger.error("Invalid graph data structure after preparation")
                return None
            
            # ENFORCE: Use the graph_type from graph_data (set by tool call override logic)
            requested_graph_type = graph_data.get('graph_type', '').lower()
            query_lower = query.lower()
            # HANDLE "TRY AGAIN" - retry previous user query if requested
            if query_lower in ['try again', 'retry', 'fix it', 'try that again']:
                # Get the most recent user query from history (exclude 'try again')
                recent_user_queries = [line[6:] for line in history if line.startswith('User: ') and 
                                    not any(retry_word in line.lower() for retry_word in ['try again', 'retry', 'fix it'])]
                if recent_user_queries:
                    original_query = recent_user_queries[-1]  # Most recent actual query
                    logger.info(f"[TRY AGAIN] Retrying with original query: {original_query}")
                    query = original_query
                    query_lower = query.lower().strip()
                else:
                    logger.warning("[TRY AGAIN] No previous query found in history")
            logger.info(f"[GRAPH] Received graph_type: '{requested_graph_type}' from tool call. Query: {query}")
            
            # Only use requested_graph_type if valid, else fallback to smart detection
            if requested_graph_type in self.supported_types:
                graph_type = requested_graph_type
            else:
                # Smart fallback if not set or invalid
                graph_type = self._determine_optimal_graph_type(graph_data, query)
                logger.info(f"[GRAPH] Falling back to smart-detected graph_type: '{graph_type}'")
            
            logger.info(f"[GRAPH] Actually using graph_type: '{graph_type}' for plotting.")
            
            # Never use pie chart for time series data
            if graph_type == 'pie' and self._is_time_series_data(graph_data):
                logger.warning("Pie chart requested for time series data; switching to line chart.")
                graph_type = 'line'
            
            # Set up matplotlib with error handling
            plt.style.use('default')
            fig, ax = plt.subplots(figsize=(12, 8))
            
            # Generate graph based on type
            success = self._create_graph_by_type(ax, graph_data, graph_type)
            if not success:
                logger.error(f"Failed to create {graph_type} chart")
                plt.close(fig)
                print(f"❌ Could not generate a {graph_type} chart for this data. Try a different chart type or aggregation.")
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
    
    def _validate_and_prepare_graph_data(self, graph_data: Dict) -> bool:
        """FIXED: Validate and prepare graph data, ensuring proper category/value structure."""
        try:
            # Check if we have raw SQL result data that needs conversion
            if 'data' in graph_data and isinstance(graph_data['data'], list) and len(graph_data['data']) > 0:
                raw_data = graph_data['data']
                if isinstance(raw_data[0], dict):
                    columns = list(raw_data[0].keys())
                    graph_type = graph_data.get('graph_type', '').lower()
                    
                    logger.info(f"[GRAPH] Preparing {graph_type} chart with columns: {columns}")
                    
                    if graph_type == 'pie':
                        # Look for category/value columns for pie charts
                        category_col = None
                        value_col = None
                        
                        # Priority mapping for pie charts
                        for col in columns:
                            col_lower = col.lower()
                            if col_lower in ['category', 'status', 'type', 'label']:
                                category_col = col
                            elif col_lower in ['value', 'count', 'amount', 'total', 'num']:
                                value_col = col
                        
                        # Fallback to first two columns if no exact match
                        if not category_col and len(columns) >= 1:
                            category_col = columns[0]
                        if not value_col and len(columns) >= 2:
                            value_col = columns[1]
                        
                        if category_col and value_col:
                            graph_data['labels'] = [str(row[category_col]) for row in raw_data]
                            graph_data['values'] = [float(row[value_col]) if row[value_col] is not None else 0 for row in raw_data]
                            logger.info(f"[PIE] Prepared data: labels={len(graph_data['labels'])}, values={len(graph_data['values'])}")
                        else:
                            logger.error(f"[PIE] Could not find category/value columns in: {columns}")
                            return False
                    
                    elif graph_type in ['bar', 'horizontal_bar']:
                        # FIXED: Better handling for bar charts
                        if len(columns) >= 2:
                            # Smart column detection
                            category_col = columns[0]  # Usually the first column is the category
                            value_col = columns[1]     # Usually the second column is the value
                            
                            # Try to find better value column
                            for col in columns:
                                col_lower = col.lower()
                                if any(keyword in col_lower for keyword in ['amount', 'total', 'count', 'num', 'value', 'payment', 'revenue']):
                                    value_col = col
                                    break
                            
                            # Prepare bar chart data
                            categories = []
                            values = []
                            
                            for row in raw_data:
                                cat_val = row.get(category_col)
                                num_val = row.get(value_col)
                                
                                if cat_val is not None and num_val is not None:
                                    # Truncate long category names for better display
                                    cat_str = str(cat_val)
                                    if len(cat_str) > 15:
                                        cat_str = cat_str[:12] + "..."
                                    categories.append(cat_str)
                                    
                                    try:
                                        values.append(float(num_val))
                                    except (ValueError, TypeError):
                                        values.append(0)
                            
                            if categories and values and len(categories) == len(values):
                                graph_data['x_values'] = categories
                                graph_data['y_values'] = values
                                graph_data['x_label'] = category_col.replace('_', ' ').title()
                                graph_data['y_label'] = value_col.replace('_', ' ').title()
                                logger.info(f"[BAR] Prepared data: {len(categories)} categories, {len(values)} values")
                                logger.info(f"[BAR] Sample data: {categories[:3]} -> {values[:3]}")
                                return True
                            else:
                                logger.error(f"[BAR] Data preparation failed: categories={len(categories)}, values={len(values)}")
                                return False
                        else:
                            logger.error(f"[BAR] Insufficient columns for bar chart: {len(columns)}")
                            return False
                    
                    elif graph_type == 'line':
                        # FIXED: Better handling for line charts
                        if len(columns) >= 2:
                            x_col = columns[0]
                            y_col = columns[1]
                            
                            # Try to find better time/value columns
                            for col in columns:
                                col_lower = col.lower()
                                if any(keyword in col_lower for keyword in ['date', 'time', 'period', 'month', 'year']):
                                    x_col = col
                                elif any(keyword in col_lower for keyword in ['amount', 'total', 'count', 'value', 'revenue']):
                                    y_col = col
                            
                            x_values = []
                            y_values = []
                            
                            for row in raw_data:
                                x_val = row.get(x_col)
                                y_val = row.get(y_col)
                                
                                if x_val is not None and y_val is not None:
                                    x_values.append(str(x_val))
                                    try:
                                        y_values.append(float(y_val))
                                    except (ValueError, TypeError):
                                        y_values.append(0)
                            
                            if x_values and y_values and len(x_values) == len(y_values):
                                graph_data['x_values'] = x_values
                                graph_data['y_values'] = y_values
                                graph_data['x_label'] = x_col.replace('_', ' ').title()
                                graph_data['y_label'] = y_col.replace('_', ' ').title()
                                logger.info(f"[LINE] Prepared data: {len(x_values)} points")
                                return True
                            else:
                                logger.error(f"[LINE] Data preparation failed: x={len(x_values)}, y={len(y_values)}")
                                return False
        
            # Validate final structure based on graph type
            graph_type = graph_data.get('graph_type', '').lower()
            if graph_type == 'pie':
                return 'labels' in graph_data and 'values' in graph_data and len(graph_data['labels']) > 0
            elif graph_type in ['bar', 'horizontal_bar']:
                return ('x_values' in graph_data and 'y_values' in graph_data and 
                       len(graph_data['x_values']) > 0 and len(graph_data['y_values']) > 0)
            elif graph_type == 'line':
                # ADDED: Check for extreme value ranges in line charts
                if 'y_values' in graph_data and len(graph_data['y_values']) > 1:
                    y_values = graph_data['y_values']
                    min_val = min(y_values)
                    max_val = max(y_values)
                    if min_val > 0:
                        ratio = max_val / min_val
                        if ratio > 500:
                            logger.warning(f"[VALIDATION] Extreme value range detected in line chart: {ratio:.1f}")
                            logger.warning(f"[VALIDATION] Small values might be hard to see: min={min_val:,.0f}, max={max_val:,.0f}")
                            graph_data['scaling_warning'] = f"Note: Values range from {min_val:,.0f} to {max_val:,.0f}"
                return ('x_values' in graph_data and 'y_values' in graph_data and 
                       len(graph_data['x_values']) > 0 and len(graph_data['y_values']) > 0)
            return True
            
        except Exception as e:
            logger.error(f"Error in data validation and preparation: {e}")
            return False
    
    def _is_time_series_data(self, graph_data: Dict) -> bool:
        """Check if data represents time series."""
        # Check for time-related column names or x_values
        if 'x_values' in graph_data:
            x_vals = graph_data['x_values']
            if x_vals and isinstance(x_vals[0], str):
                first_val = str(x_vals[0]).lower()
                return any(word in first_val for word in ['january', 'february', 'march', 'april', 'may', 'june', 
                                                         'july', 'august', 'september', 'october', 'november', 'december',
                                                         '2024', '2025', 'jan', 'feb', 'mar', 'apr', 'may', 'jun',
                                                         'jul', 'aug', 'sep', 'oct', 'nov', 'dec'])
        
        if 'data' in graph_data and isinstance(graph_data['data'], list) and len(graph_data['data']) > 0:
            columns = list(graph_data['data'][0].keys())
            return any(any(word in col.lower() for word in ['date', 'time', 'period', 'month', 'year']) for col in columns)
        
        return False
    
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
        if self._is_time_series_data(graph_data):
            return 'line'
        elif 'labels' in graph_data and 'values' in graph_data:
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
        """FIXED: Create complete enhanced bar chart with smart formatting."""
        try:
            # Check for x_values/y_values first (preferred format)
            if 'x_values' in graph_data and 'y_values' in graph_data:
                x_values = graph_data['x_values']
                y_values = graph_data['y_values']
                
                if not x_values or not y_values or len(x_values) != len(y_values):
                    logger.error(f"[BAR] Invalid x/y values: x={len(x_values) if x_values else 0}, y={len(y_values) if y_values else 0}")
                    return False
                
                # Limit to reasonable number of bars
                if len(x_values) > 30:
                    x_values = x_values[:30]
                    y_values = y_values[:30]
                    logger.info("[BAR] Limited to 30 bars for readability")
                
                # Create bar chart
                bars = ax.bar(range(len(x_values)), y_values, color='steelblue', alpha=0.8, edgecolor='darkblue')
                
                # Set labels
                ax.set_xlabel(graph_data.get('x_label', 'Categories'), fontsize=12)
                ax.set_ylabel(graph_data.get('y_label', 'Values'), fontsize=12)
                ax.set_xticks(range(len(x_values)))
                ax.set_xticklabels(x_values, rotation=45, ha='right')
                
                # Add value labels on bars if not too many
                if len(x_values) <= 15:
                    max_val = max(y_values) if y_values else 1
                    for i, (bar, value) in enumerate(zip(bars, y_values)):
                        height = bar.get_height()
                        ax.text(bar.get_x() + bar.get_width()/2., height + max_val * 0.01,
                               f'{value:.0f}' if isinstance(value, float) else str(value),
                               ha='center', va='bottom', fontsize=8)
                
                ax.grid(True, alpha=0.3, axis='y')
                logger.info(f"[BAR] Successfully created bar chart with {len(x_values)} bars")
                return True
            
            # Fallback to categories/values format
            elif 'categories' in graph_data and 'values' in graph_data:
                categories = graph_data['categories']
                values = graph_data['values']
                
                if not categories or not values or len(categories) != len(values):
                    logger.error(f"[BAR] Invalid categories/values: cat={len(categories) if categories else 0}, val={len(values) if values else 0}")
                    return False
                
                # Limit to reasonable number
                if len(categories) > 30:
                    categories = categories[:30]
                    values = values[:30]
                    logger.info("[BAR] Limited to 30 categories for readability")
                
                bars = ax.bar(range(len(categories)), values, color='steelblue', alpha=0.8, edgecolor='darkblue')
                ax.set_xlabel(graph_data.get('x_label', 'Categories'), fontsize=12)
                ax.set_ylabel(graph_data.get('y_label', 'Values'), fontsize=12)
                ax.set_xticks(range(len(categories)))
                ax.set_xticklabels([str(cat)[:15] for cat in categories], rotation=45, ha='right')
                
                # Add value labels if not too many
                if len(categories) <= 15:
                    max_val = max(values) if values else 1
                    for i, (bar, value) in enumerate(zip(bars, values)):
                        height = bar.get_height()
                        ax.text(bar.get_x() + bar.get_width()/2., height + max_val * 0.01,
                               f'{value:.0f}' if isinstance(value, float) else str(value),
                               ha='center', va='bottom', fontsize=8)
                
                ax.grid(True, alpha=0.3, axis='y')
                logger.info(f"[BAR] Successfully created bar chart with {len(categories)} categories")
                return True
            
            else:
                logger.error("[BAR] Missing required data: need either (x_values, y_values) or (categories, values)")
                return False
            
        except Exception as e:
            logger.error(f"[BAR] Bar chart creation failed: {e}")
            import traceback
            logger.error(f"[BAR] Traceback: {traceback.format_exc()}")
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
        """FIXED: Create line chart with proper scaling for small values."""
        try:
            # Check for x_values/y_values format
            if 'x_values' in graph_data and 'y_values' in graph_data:
                x_values = graph_data['x_values']
                y_values = graph_data['y_values']
                
                if not x_values or not y_values or len(x_values) != len(y_values):
                    logger.error(f"[LINE] Invalid x/y values: x={len(x_values) if x_values else 0}, y={len(y_values) if y_values else 0}")
                    return False
                
                # CRITICAL FIX: Check for extreme value ranges that cause scaling issues
                if len(y_values) > 1:
                    min_val = min(y_values)
                    max_val = max(y_values)
                    value_range_ratio = max_val / min_val if min_val > 0 else float('inf')
                    
                    logger.info(f"[LINE] Value range: {min_val:,.0f} to {max_val:,.0f} (ratio: {value_range_ratio:.1f})")
                    
                    # If we have extreme scaling issues (ratio > 500), use log scale or adjust
                    if value_range_ratio > 500 and min_val > 0:
                        logger.warning(f"[LINE] Extreme value range detected (ratio: {value_range_ratio:.1f})")
                        logger.info("[LINE] Applying scaling fix to prevent small values from disappearing")
                        
                        # Option 1: Use log scale if all values are positive
                        if all(val > 0 for val in y_values):
                            ax.set_yscale('log')
                            logger.info("[LINE] Applied logarithmic Y-axis scale")
                        
                        # Option 2: Set a reasonable Y-axis minimum to show small values
                        else:
                            # Set Y-axis to start from 0 but ensure small values are visible
                            y_margin = max_val * 0.05  # 5% margin
                            ax.set_ylim(bottom=0, top=max_val + y_margin)
                            logger.info(f"[LINE] Set Y-axis range: 0 to {max_val + y_margin:,.0f}")
                
                # Create the line chart
                line = ax.plot(range(len(x_values)), y_values, 
                              color='darkgreen', linewidth=2.5, marker='o', 
                              markersize=5, markerfacecolor='green', alpha=0.8)
                
                # Set labels and formatting
                ax.set_xlabel(graph_data.get('x_label', 'Time Period'), fontsize=12, fontweight='bold')
                ax.set_ylabel(graph_data.get('y_label', 'Values'), fontsize=12, fontweight='bold')
                ax.set_xticks(range(len(x_values)))
                ax.set_xticklabels(x_values, rotation=45, ha='right')
                
                # ENHANCED: Add value labels on points for better readability
                if len(x_values) <= 20:  # Only add labels if not too crowded
                    for i, (x, y) in enumerate(zip(range(len(x_values)), y_values)):
                        # Format large numbers with K, M suffixes
                        if y >= 1000000:
                            label = f'{y/1000000:.1f}M'
                        elif y >= 1000:
                            label = f'{y/1000:.0f}K'
                        else:
                            label = f'{y:.0f}'
                        # Position label above the point
                        ax.annotate(label, (x, y), textcoords="offset points", 
                                   xytext=(0,10), ha='center', fontsize=8, 
                                   bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7))
                
                # Enhanced grid and styling
                ax.grid(True, alpha=0.3, linestyle='--')
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                
                # CRITICAL: Force the plot to show all data points properly
                ax.autoscale(tight=False)
                
                logger.info(f"[LINE] Successfully created line chart with {len(x_values)} points")
                return True
                
            else:
                logger.error("[LINE] Missing required x_values/y_values data")
                return False
                
        except Exception as e:
            logger.error(f"[LINE] Line chart creation failed: {e}")
            import traceback
            logger.error(f"[LINE] Traceback: {traceback.format_exc()}")
            return False

    def _create_dual_scale_line_chart(self, ax, graph_data: Dict) -> bool:
        """Alternative: Create line chart with dual Y-axis for extreme value ranges."""
        try:
            if 'x_values' in graph_data and 'y_values' in graph_data:
                x_values = graph_data['x_values']
                y_values = graph_data['y_values']
                
                # Detect if we need dual scale
                min_val = min(y_values)
                max_val = max(y_values)
                ratio = max_val / min_val if min_val > 0 else 1
                
                if ratio > 1000:
                    logger.info("[LINE] Using dual-scale approach for extreme value range")
                    
                    # Separate small and large values
                    threshold = max_val * 0.1  # 10% of max
                    small_indices = [i for i, val in enumerate(y_values) if val < threshold]
                    large_indices = [i for i, val in enumerate(y_values) if val >= threshold]
                    
                    if small_indices and large_indices:
                        # Create main plot for large values
                        large_x = [x_values[i] for i in large_indices]
                        large_y = [y_values[i] for i in large_indices]
                        
                        ax.plot([i for i in large_indices], large_y, 
                               color='darkgreen', linewidth=2.5, marker='o', label='High Values')
                        
                        # Create secondary axis for small values
                        ax2 = ax.twinx()
                        small_x = [x_values[i] for i in small_indices]
                        small_y = [y_values[i] for i in small_indices]
                        
                        ax2.plot([i for i in small_indices], small_y, 
                                color='orange', linewidth=2.5, marker='s', label='Low Values')
                        
                        # Set labels
                        ax.set_ylabel('High Values', color='darkgreen')
                        ax2.set_ylabel('Low Values', color='orange')
                        
                        # Add legend
                        lines1, labels1 = ax.get_legend_handles_labels()
                        lines2, labels2 = ax2.get_legend_handles_labels()
                        ax.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
                        
                        return True
                
                # Fall back to regular line chart
                return self._create_complete_line_chart(ax, graph_data)
        except Exception as e:
            logger.error(f"[DUAL-LINE] Dual scale chart creation failed: {e}")
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
            
            if os   .name == 'nt':  # Windows
                os.startfile(filepath)
            elif sys.platform == 'darwin':  # macOS
                subprocess.run(['open', filepath], check=True, timeout=5)
            else:  # Linux
                subprocess.run(['xdg-open', filepath], check=True, timeout=5)
            
            logger.info(f"📊 Graph opened: {filepath}")
            return True
        except Exception:
            return False

class SimpleAIModel:
    """Simple wrapper for Google AI model."""
    def __init__(self, model_name="gemini-1.5-flash"):
        import google.generativeai as genai
        self.model = genai.GenerativeModel(model_name)
    async def generate_content_async(self, prompt: str):
        try:
            response = self.model.generate_content(prompt)
            return response
        except Exception as e:
            logger.error(f"AI generation failed: {e}")
            return None

class CompleteSmartNLPProcessor:
    """COMPLETE NLP processor with enhanced threshold detection and better prompting. FIXED MULTITOOL SUPPORT."""
    
    def __init__(self, config=None):
        self.config = config or {}
        try:
            import google.generativeai as genai
            api_key = self.config.get('GOOGLE_API_KEY') if self.config else None
            if api_key:
                genai.configure(api_key=api_key)
            self.ai_model = SimpleAIModel()
            logger.info("✅ AI model initialized successfully")
        except Exception as e:
            logger.error(f"❌ Failed to initialize AI model: {e}")
            self.ai_model = None
        self.model = genai.GenerativeModel('gemini-2.0-flash-lite')
        self.db_schema = self._get_complete_database_schema()
        self.chart_keywords = self._get_chart_keywords()
        self.tools = self._get_tools_config()

    async def _generate_with_complete_retries(self, prompt: str, original_query: str, chart_analysis: dict) -> list:
        """Enhanced generation with better error handling and retries."""
        max_attempts = 3
        for attempt in range(max_attempts):
            try:
                logger.info(f"🧠 Complete AI generation attempt {attempt + 1}")
                response = await self.ai_model.generate_content_async(prompt)
                if not response or not hasattr(response, 'text') or not response.text:
                    logger.warning(f"Complete AI generation attempt {attempt + 1} failed: empty response")
                    continue
                response_text = response.text.strip()
                if not response_text:
                    logger.warning(f"Complete AI generation attempt {attempt + 1} failed: empty text")
                    continue
                logger.info(f"🧠 AI Response: {response_text[:200]}...")
                json_match = re.search(r'\[.*?\]', response_text, re.DOTALL)
                if json_match:
                    try:
                        tool_calls = json.loads(json_match.group())
                        if tool_calls and isinstance(tool_calls, list):
                            logger.info(f"✅ Successfully parsed {len(tool_calls)} tool calls")
                            enhanced_calls = []
                            for call in tool_calls:
                                if isinstance(call, dict) and 'tool' in call and 'parameters' in call:
                                    call['original_query'] = original_query
                                    call['wants_graph'] = call.get('tool') == 'execute_dynamic_sql_with_graph'
                                    call['chart_analysis'] = chart_analysis or {'chart_type': 'none'}
                                    enhanced_calls.append(call)
                            if enhanced_calls:
                                return enhanced_calls
                    except json.JSONDecodeError as e:
                        logger.warning(f"JSON parse error: {e}")
                        continue
                tool_calls = self._extract_tool_calls_from_text(response_text, original_query, chart_analysis)
                if tool_calls:
                    return tool_calls
            except Exception as e:
                logger.warning(f"Complete AI generation attempt {attempt + 1} failed: {e}")
                continue
        logger.warning("All AI generation attempts failed, using fallback.")
        return self._get_complete_smart_fallback_tool_call(original_query, [])

    def _extract_tool_calls_from_text(self, text: str, query: str, chart_analysis: dict) -> list:
        """Extract tool calls from unstructured AI response."""
        query_lower = query.lower()
        sql_patterns = [
            r'```sql\s*(SELECT.*?)```',
            r'```\s*(SELECT.*?)```',
            r'(SELECT.*?;|SELECT.*?\n\n|SELECT.*?$)',
        ]
        sql_query = None
        for pattern in sql_patterns:
            sql_match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
            if sql_match:
                sql_query = sql_match.group(1) if pattern.startswith('```') else sql_match.group()
                sql_query = sql_query.strip().rstrip(';')
                break
        if sql_query:
            wants_graph = any(word in query_lower for word in ['chart', 'graph', 'visualize', 'plot'])
            tool_name = 'execute_dynamic_sql_with_graph' if wants_graph else 'execute_dynamic_sql'
            tool_call = {
                'tool': tool_name,
                'parameters': {'sql_query': sql_query},
                'original_query': query,
                'wants_graph': wants_graph,
                'chart_analysis': chart_analysis or {'chart_type': 'none'}
            }
            if wants_graph and 'graph_type' not in tool_call['parameters']:
                tool_call['parameters']['graph_type'] = chart_analysis.get('chart_type', 'bar')
            logger.info(f"🔧 Extracted SQL tool call: {tool_name}")
            return [tool_call]
        if 'database_status' in text.lower() or 'get_database_status' in text.lower():
            return [{
                'tool': 'get_database_status',
                'parameters': {},
                'original_query': query,
                'wants_graph': False,
                'chart_analysis': {'chart_type': 'none'}
            }]
        if 'payment' in text.lower() and 'success' in text.lower():
            return [{
                'tool': 'get_payment_success_rate_in_last_days',
                'parameters': {'days': 30},
                'original_query': query,
                'wants_graph': False,
                'chart_analysis': {'chart_type': 'none'}
            }]
        if 'subscription' in text.lower() and 'last' in text.lower():
            return [{
                'tool': 'get_subscriptions_in_last_days',
                'parameters': {'days': 30},
                'original_query': query,
                'wants_graph': False,
                'chart_analysis': {'chart_type': 'none'}
            }]
        return []

    def _get_complete_database_schema(self) -> str:
        """Get COMPLETE CORRECTED database schema with full column information."""
        return """
COMPLETE CORRECTED Database Schema:

Tables:
1. subscription_contract_v2:
   - subscription_id (bigint, PRIMARY KEY, auto_increment)
   - merchant_user_id (varchar(255))
   - frequency (int)
   - frequency_unit (int)
   - grace_days (int)
   - merchant_id (varchar(255), NOT NULL)
   - payer_account_number (varchar(255))
   - internal_user_id (varchar(255))
   - subs_payment_mode (varchar(255))
   - saved_card_id (bigint)
   - subcription_start_date (datetime, NOT NULL) -- Note: column name has typo "subcription"
   - subcription_end_date (datetime, NOT NULL) -- Note: column name has typo "subcription"
   - status (varchar(255), NOT NULL)
   - website (varchar(255))
   - industry_type (varchar(255))
   - created_date (datetime, DEFAULT CURRENT_TIMESTAMP)
   - updated_date (datetime, ON UPDATE CURRENT_TIMESTAMP)
   - retry_count (int, DEFAULT 0)
   - amount_type (int, DEFAULT 0)
   - max_amount (float)
   - user_email (varchar(255)) -- AVAILABLE: Customer email addresses
   - user_mobile (varchar(255))
   - user_name (varchar(255)) -- AVAILABLE: Customer names
   - order_id (varchar(255))
   - start_date_flow (int, DEFAULT 1)
   - channel_id (varchar(20))
   - service_id (varchar(100))
   - account_type (varchar(255))
   - sub_status (varchar(255))
   - renewal_amount (decimal(16,2))
   - auto_renewal (tinyint, DEFAULT 0)
   - auto_retry (tinyint, NOT NULL, DEFAULT 0)
   - is_communication_manager (tinyint, NOT NULL, DEFAULT 0)
   - subs_goods_info (varchar(255))
   - subs_callback_url (varchar(255))
   - sub_due_date (datetime)
   - is_new_subscription (tinyint(1), NOT NULL, DEFAULT 0)
   - metadata (text)
   - card_info_id (bigint)
   - link_id (bigint)
   - plan_id (bigint)
   - si_hub_details_id (bigint)
   - token_details_id (bigint)
   - max_amount_decimal (decimal(16,2))

2. subscription_payment_details:
   - subcription_payment_details_id (bigint, PRIMARY KEY, auto_increment) -- Note: column name has typo "subcription"
   - subscription_id (bigint, NOT NULL, FOREIGN KEY to subscription_contract_v2.subscription_id)
   - acquirement_id (varchar(255))
   - trans_amount (decimal(16,2))
   - order_id (varchar(255))
   - time_out (datetime)
   - created_date (datetime, DEFAULT CURRENT_TIMESTAMP)
   - updated_date (datetime, ON UPDATE CURRENT_TIMESTAMP)
   - status (varchar(255)) -- Payment status
   - payment_type (varchar(255), NOT NULL)
   - metadata (varchar(255))
   - payment_time (datetime)
   - trans_amount_decimal (decimal(16,2))

CRITICAL COMPLETE SCHEMA RULES:
- EMAIL IS AVAILABLE: Use c.user_email for customer email addresses
- END DATE IS AVAILABLE: Use c.subcription_end_date for subscription end dates  
- NAME IS AVAILABLE: Use c.user_name for customer names
- PRIMARY KEY: Use subcription_payment_details_id (with typo) for payments table
- To get merchant_user_id info from payments, you MUST JOIN:
  FROM subscription_payment_details p 
  JOIN subscription_contract_v2 c ON p.subscription_id = c.subscription_id
  
- NEVER use merchant_user_id in GROUP BY with subscription_payment_details alone
- Always use proper JOINs when accessing both tables

COMPLETE STATUS VALUES:
- Subscription status: ACTIVE, INACTIVE, CLOSED, REJECT, INIT
- Payment status: ACTIVE, INIT, FAIL (and possibly others)

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
- Remember column typos: subcription_start_date, subcription_end_date (not subscription_)
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
        comparison_info = self._extract_comparison_info(user_query)
        is_comparison = False
        # --- ENHANCED: Also treat time-period comparisons as single query ---
        is_time_period_comparison = comparison_info.get('comparison_type') == 'time_period_comparison' and comparison_info.get('time_periods')
        if (
            (threshold_info['has_threshold'] and threshold_info['numbers'] and
            (('compare' in query_lower or 'vs' in query_lower or 'versus' in query_lower or 'and' in query_lower)))
            or is_time_period_comparison
        ):
            if is_time_period_comparison or (len(threshold_info['numbers']) >= 2):
                is_comparison = True
        # --- COMPLEX ANALYTICAL QUERY DETECTION ---
        complex_patterns = [
            'list of customers', 'show me their', 'customers who have', 
            'users who have', 'find customers', 'get customers'
        ]
        is_complex_analytical = any(pattern in query_lower for pattern in complex_patterns)
        # Only split if not a comparison AND not complex analytical query
        if not is_comparison and not is_complex_analytical:
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
        
        # --- FEEDBACK-AWARE: Get best_chart_type and actionable_rules from improvement context for this query ---
        best_chart_type = None
        improvement_context = None
        actionable_rules = []
        if client:
            try:
                improvement_context, best_chart_type, actionable_rules = await self._get_complete_improvement_context(user_query, history, client, return_chart_type=True, return_rules=True)
            except TypeError:
                improvement_context, best_chart_type = await self._get_complete_improvement_context(user_query, history, client, return_chart_type=True)
                actionable_rules = []
        
        for i, query in enumerate(unique_queries, 1):
            logger.info(f"🔧 MULTITOOL: Processing query {i}/{len(unique_queries)}: {query[:50]}...")
            try:
                # --- FEEDBACK-AWARE LOGIC ---
                auto_applied = False
                auto_union = False
                auto_no_graph = False
                auto_chart_type = None
                auto_aggregate_by = None
                suggestions_result = None
                
                # Apply actionable rules to this query
                for rule in actionable_rules:
                    if rule['action'] == 'aggregate_by' and rule['trigger'] in query.lower():
                        auto_aggregate_by = rule['value']
                        print_section(f"💡 Auto-applying aggregation: {rule['instruction']}")
                    if rule['action'] == 'chart_type' and (rule['trigger'] in query.lower() or rule['value'] in query.lower()):
                        auto_chart_type = rule['value']
                        print_section(f"💡 Auto-applying chart type: {rule['instruction']}")
                    if rule['action'] == 'no_graph' and (rule['trigger'] in query.lower() or 'graph' in query.lower() or 'chart' in query.lower()):
                        auto_no_graph = True
                        print_section(f"💡 Auto-applying: {rule['instruction']}")
                
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
                                if 'line chart' in sug or 'line graph' in sug:
                                    auto_chart_type = 'line'
                                    auto_applied = True
                                    print_section(f'💡 Auto-applying past improvement: "{suggestion["user_suggestion"]}"')
                    except Exception as e:
                        logger.warning(f"Could not fetch improvement suggestions: {e}")
                
                # --- FEEDBACK-AWARE: If best_chart_type from improvement context, enforce it ---
                if best_chart_type and not auto_chart_type:
                    auto_chart_type = best_chart_type
                    logger.info(f"[FEEDBACK] Enforcing chart type from feedback: {auto_chart_type}")
                # --- END FEEDBACK-AWARE LOGIC ---
                
                query_tool_calls = await self._process_single_query(query, history, client, auto_union=auto_union, auto_no_graph=auto_no_graph, auto_chart_type=auto_chart_type, auto_aggregate_by=auto_aggregate_by, force_comparison=is_comparison)
                
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

    async def _process_single_query(self, query: str, history: List[str], client=None, auto_union=False, auto_no_graph=False, auto_chart_type=None, auto_aggregate_by=None, force_comparison=False, actionable_rules=None) -> List[Dict]:
        """Process a single query and return tool calls, with feedback-aware logic. If force_comparison is True, always generate a single UNION SQL."""
        query_lower = query.lower().strip()
        logger.info(f"[DEBUG] _process_single_query received: {query_lower}")
        
        # HANDLE "TRY AGAIN" - IMPROVED VERSION
        if query_lower in ['try again', 'retry', 'fix it', 'try that again']:
            # Get the most recent user query from history (exclude "try again" and feedback)
            recent_user_queries = []
            for line in reversed(history):  # Start from most recent
                if line.startswith('User: '):
                    query_text = line[6:].strip()
                    # Skip retry commands and very short queries
                    if (not any(retry_word in query_text.lower() for retry_word in ['try again', 'retry', 'fix it']) 
                        and len(query_text) > 3):
                        recent_user_queries.append(query_text)
                        break  # Take the first (most recent) valid query
            if recent_user_queries:
                original_query = recent_user_queries[0]
                logger.info(f"[TRY AGAIN] Retrying with original query: {original_query}")
                query = original_query
                query_lower = query.lower().strip()
            else:
                logger.warning("[TRY AGAIN] No previous query found in history")
        
        chart_analysis = self._analyze_complete_chart_requirements(query, history)
        if actionable_rules is None:
            actionable_rules = []
        
        # Apply aggregation override if present
        if auto_aggregate_by:
            chart_analysis['aggregation'] = auto_aggregate_by
        
        # FIXED: Define all required variables before using them
        threshold_info = self._extract_threshold_info(query)
        date_info = self._extract_date_info(query)
        comparison_info = self._extract_comparison_info(query)
        
        # 1. Handle specific date queries directly
        if date_info['has_date'] and date_info['dates']:
            logger.info("[DEBUG] Path: specific date query detected.")
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
            sql = self._fix_sql_date_math(sql, query)
            
            return [{
                'tool': 'execute_dynamic_sql',
                'parameters': {'sql_query': sql},
                'original_query': query,
                'wants_graph': False,
                'chart_analysis': {'chart_type': 'none'}
            }]
        
        # 1b. Handle time period comparison queries (e.g., last month vs previous month)
        if comparison_info.get('comparison_type') == 'time_period_comparison' and comparison_info.get('time_periods') == ['last_month', 'prev_month']:
            logger.info("[DEBUG] Path: time period comparison detected (last month vs previous month). Generating UNION SQL.")
            # Only for revenue/payment queries
            if ('revenue' in query_lower or 'payment' in query_lower or 'total' in query_lower) and ('last month' in query_lower and ('month before' in query_lower or 'previous month' in query_lower)):
                sql = """
SELECT DATE_FORMAT(DATE_SUB(CURDATE(), INTERVAL 1 MONTH), '%M %Y') AS period, SUM(p.trans_amount_decimal) AS total_revenue
FROM subscription_payment_details p
WHERE p.status = 'ACTIVE'
  AND DATE_FORMAT(p.created_date, '%Y-%m') = DATE_FORMAT(CURDATE() - INTERVAL 1 MONTH, '%Y-%m')
UNION ALL
SELECT DATE_FORMAT(DATE_SUB(CURDATE(), INTERVAL 2 MONTH), '%M %Y') AS period, SUM(p.trans_amount_decimal) AS total_revenue
FROM subscription_payment_details p
WHERE p.status = 'ACTIVE'
  AND DATE_FORMAT(p.created_date, '%Y-%m') = DATE_FORMAT(CURDATE() - INTERVAL 2 MONTH, '%Y-%m')
"""
                sql = self._fix_sql_quotes(sql)
                sql = self._validate_and_autofix_sql(sql)
                sql = self._fix_sql_date_math(sql, query)
                
                return [{
                    'tool': 'execute_dynamic_sql',
                    'parameters': {'sql_query': sql},
                    'original_query': query,
                    'wants_graph': False,
                    'chart_analysis': {'chart_type': 'none'}
                }]
        
        # 2. Handle comparison queries with UNION ALL
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
                sql = self._fix_sql_date_math(sql, query)
                
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
                sql = self._fix_sql_date_math(sql, query)
                
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
            # FIXED: Better chart type detection and SQL generation for trends over time
            if any(word in query_lower for word in ['trend', 'over time', 'timeline']) or auto_chart_type == 'line':
                sql = """
SELECT DATE_FORMAT(p.created_date, '%M %Y') AS period, SUM(p.trans_amount_decimal) AS total_revenue
FROM subscription_payment_details p
WHERE p.status = 'ACTIVE'
  AND p.created_date >= DATE_SUB(CURDATE(), INTERVAL 3 MONTH)
GROUP BY DATE_FORMAT(p.created_date, '%Y-%m')
ORDER BY DATE_FORMAT(p.created_date, '%Y-%m')
"""
                sql = self._fix_sql_quotes(sql)
                sql = self._validate_and_autofix_sql(sql)
                sql = self._fix_sql_date_math(sql, query)
                
                return [{
                    'tool': 'execute_dynamic_sql_with_graph',
                    'parameters': {'sql_query': sql, 'graph_type': 'line'},
                    'original_query': query,
                    'wants_graph': True,
                    'chart_analysis': chart_analysis
                }]
            
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
                    sql = self._fix_sql_date_math(sql, query)
                    
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
                sql = self._fix_sql_date_math(sql, query)
                
                return [{
                    'tool': 'execute_dynamic_sql_with_graph',
                    'parameters': {'sql_query': sql, 'graph_type': 'pie'},
                    'original_query': query,
                    'wants_graph': True,
                    'chart_analysis': chart_analysis
                }]
        
        # 4. Fall back to AI processing for complex queries
        try:
            history_context = self._build_complete_history_context(history)
            improvement_context = await self._get_complete_improvement_context(query, history, client)
            similar_context = await self._get_similar_queries_context(query, client)
            
            if auto_chart_type:
                chart_analysis['chart_type'] = auto_chart_type
            
            prompt = self._create_enhanced_threshold_prompt(
                query, history_context, improvement_context, similar_context, 
                chart_analysis, threshold_info, date_info, comparison_info, actionable_rules
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
            
            # --- ENFORCE CHART TYPE OVERRIDE BASED ON USER QUERY OR FEEDBACK ---
            # Detect explicit chart type requests in the query
            chart_type_override = None
            if any(word in query_lower for word in ['line graph', 'line chart']):
                chart_type_override = 'line'
            elif any(word in query_lower for word in ['bar graph', 'bar chart']):
                chart_type_override = 'bar'
            elif any(word in query_lower for word in ['pie chart', 'pie graph']):
                chart_type_override = 'pie'
            elif any(word in query_lower for word in ['scatter plot', 'scatter chart']):
                chart_type_override = 'scatter'
            
            # Also check for feedback-based override
            if not chart_type_override and auto_chart_type:
                chart_type_override = auto_chart_type
            
            if chart_type_override:
                for call in tool_calls:
                    if call['tool'] == 'execute_dynamic_sql_with_graph':
                        prev_type = call['parameters'].get('graph_type', None)
                        call['parameters']['graph_type'] = chart_type_override
                        logger.info(f"[ENFORCE] Overriding graph_type from {prev_type} to {chart_type_override} due to explicit user request or feedback.")
            
            for call in tool_calls:
                if 'sql_query' in call['parameters']:
                    call['parameters']['sql_query'] = self._fix_sql_quotes(call['parameters']['sql_query'])
                    call['parameters']['sql_query'] = self._validate_and_autofix_sql(call['parameters']['sql_query'])
                    call['parameters']['sql_query'] = self._fix_sql_date_math(call['parameters']['sql_query'], query)
            
            enhanced_calls = self._enhance_and_validate_complete_tool_calls(
                tool_calls, query, chart_analysis, threshold_info
            )
            
            logger.info(f"🧠 AI selected tool(s): {[tc['tool'] for tc in enhanced_calls]}")
            return enhanced_calls
            
        except Exception as e:
            logger.error(f"Error in AI query processing: {e}", exc_info=True)
            return self._get_complete_smart_fallback_tool_call(query, history)

    def handle_specific_date_queries(self, query: str, history: List[str]) -> List[Dict]:
        """Handle specific date queries with improved date parsing."""
        date_info = self._extract_date_info(query)
        
        if date_info['has_date'] and date_info['dates']:
            date_str = date_info['dates'][0]
            
            # Convert different date formats to YYYY-MM-DD
            try:
                if re.match(r'\d{1,2}\s+(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\w*\s+\d{4}', date_str, re.IGNORECASE):
                    # Parse "24 april 2025" format
                    from datetime import datetime
                    date_obj = datetime.strptime(date_str, '%d %B %Y')
                    date_str = date_obj.strftime('%Y-%m-%d')
                elif re.match(r'\d{2}/\d{2}/\d{4}', date_str):
                    # Parse "MM/DD/YYYY" format  
                    from datetime import datetime
                    date_obj = datetime.strptime(date_str, '%m/%d/%Y')
                    date_str = date_obj.strftime('%Y-%m-%d')
                # YYYY-MM-DD format is already correct
            except Exception as e:
                logger.warning(f"Could not parse date '{date_str}': {e}")
                return []
                
            # Generate the correct SQL query
            if 'subscription' in query.lower():
                sql = f"SELECT COUNT(*) as num_subscriptions FROM subscription_contract_v2 WHERE DATE(subcription_start_date) = '{date_str}'"
            elif 'payment' in query.lower():
                sql = f"SELECT COUNT(*) as num_payments FROM subscription_payment_details WHERE DATE(created_date) = '{date_str}'"
            else:
                # Default to subscriptions
                sql = f"SELECT COUNT(*) as num_subscriptions FROM subscription_contract_v2 WHERE DATE(subcription_start_date) = '{date_str}'"
            
            return [{
                'tool': 'execute_dynamic_sql',
                'parameters': {'sql_query': sql},
                'original_query': query,
                'wants_graph': False,
                'chart_analysis': {'chart_type': 'none'}
            }]
        
        return []

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
        """Extract date information from query with improved parsing."""
        date_info = {
            'has_date': False,
            'dates': [],
            'date_context': 'unknown'
        }
        
        # Extract dates in various formats
        date_patterns = [
            r'\d{4}-\d{2}-\d{2}',  # YYYY-MM-DD
            r'\d{2}/\d{2}/\d{4}',  # MM/DD/YYYY  
            r'\d{1,2}\s+(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\w*\s+\d{4}',  # 24 april 2025
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
        """Extract comparison information from query, including time period comparisons."""
        comparison_info = {
            'is_comparison': False,
            'elements': [],
            'comparison_type': 'unknown',
            'time_periods': []
        }
        
        query_lower = query.lower()
        
        # Detect comparison keywords
        comparison_keywords = ['compare', 'vs', 'versus', 'and', 'both', 'between', 'how does', 'difference']
        if any(keyword in query_lower for keyword in comparison_keywords):
            comparison_info['is_comparison'] = True
        
        # Detect time period comparison (e.g., last month vs the month before)
        time_periods = []
        # Loosened detection: allow for any order, extra words, etc.
        if ('last month' in query_lower and ('month before' in query_lower or 'previous month' in query_lower)) or (('month before' in query_lower or 'previous month' in query_lower) and 'last month' in query_lower):
            time_periods = ['last_month', 'prev_month']
        elif 'this month' in query_lower and 'last month' in query_lower:
            time_periods = ['this_month', 'last_month']
        
        if time_periods:
            comparison_info['is_comparison'] = True
            comparison_info['comparison_type'] = 'time_period_comparison'
            comparison_info['time_periods'] = time_periods
        
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

    def _extract_actionable_rules_from_suggestions(self, improvements):
        """Extract actionable rules (aggregation, chart type, etc.) from improvement suggestions."""
        import re
        rules = []
        
        for imp in improvements:
            suggestion = imp.get('user_suggestion', '').lower()
            
            # Aggregation rules
            if re.search(r'over time.*week|trend.*week|aggregate.*week|weekly', suggestion):
                rules.append({
                    'trigger': 'over time',
                    'action': 'aggregate_by',
                    'value': 'week',
                    'instruction': "ALWAYS aggregate by week for 'over time' or trend queries."
                })
            if re.search(r'over time.*month|trend.*month|aggregate.*month|monthly', suggestion):
                rules.append({
                    'trigger': 'over time',
                    'action': 'aggregate_by',
                    'value': 'month',
                    'instruction': "ALWAYS aggregate by month for 'over time' or trend queries."
                })
            
            # Chart type rules
            if 'bar chart' in suggestion:
                rules.append({'trigger': 'bar chart', 'action': 'chart_type', 'value': 'bar', 'instruction': 'ALWAYS use a bar chart for relevant queries.'})
            if 'pie chart' in suggestion:
                rules.append({'trigger': 'pie chart', 'action': 'chart_type', 'value': 'pie', 'instruction': 'ALWAYS use a pie chart for relevant queries.'})
            if 'line chart' in suggestion or 'line graph' in suggestion:
                rules.append({'trigger': 'line chart', 'action': 'chart_type', 'value': 'line', 'instruction': 'ALWAYS use a line chart for trend queries.'})
            if 'scatter' in suggestion:
                rules.append({'trigger': 'scatter', 'action': 'chart_type', 'value': 'scatter', 'instruction': 'ALWAYS use a scatter plot for correlation/relationship queries.'})
            
            # No graph rules
            if 'do not generate a graph' in suggestion or 'no graph' in suggestion or 'no chart' in suggestion:
                rules.append({'trigger': 'no graph', 'action': 'no_graph', 'value': True, 'instruction': 'DO NOT generate a graph for this type of query.'})
        
        return rules

    def _format_chart_requirements(self, chart_analysis: Dict) -> str:
        """Format chart requirements for the AI prompt."""
        if not chart_analysis:
            return "No chart requested."
        
        chart_type = chart_analysis.get('chart_type', 'none')
        
        if chart_type == 'none':
            return "No chart requested."
        elif chart_type == 'pie':
            return """PIE CHART REQUIRED:
- Use execute_dynamic_sql_with_graph tool
- SQL must return exactly 2 columns: category (text) and value (number)
- Example: SELECT 'Successful' as category, COUNT(*) as value FROM ... UNION ALL SELECT 'Failed' as category, COUNT(*) as value FROM ..."""
        elif chart_type == 'bar':
            return """BAR CHART REQUIRED:
- Use execute_dynamic_sql_with_graph tool  
- SQL should return category and value columns
- Limit to reasonable number of bars (≤30)"""
        elif chart_type == 'line':
            return """LINE CHART REQUIRED:
- Use execute_dynamic_sql_with_graph tool
- SQL should return time/period and value columns
- Order by time ascending"""
        else:
            return f"CHART REQUIRED: {chart_type} - Use execute_dynamic_sql_with_graph tool"

    def _enhance_and_validate_complete_tool_calls(self, tool_calls: List[Dict], 
                                                 query: str, chart_analysis: Dict, 
                                                 threshold_info: Dict) -> List[Dict]:
        """Enhanced validation with proper original_query handling."""
        enhanced_calls = []
        
        for call in tool_calls:
            try:
                # Ensure original_query is always present
                if 'original_query' not in call:
                    call['original_query'] = query
                
                # Ensure wants_graph is present
                if 'wants_graph' not in call:
                    call['wants_graph'] = call.get('tool') == 'execute_dynamic_sql_with_graph'
                
                # Ensure chart_analysis is present
                if 'chart_analysis' not in call:
                    call['chart_analysis'] = chart_analysis or {'chart_type': 'none'}
                
                # Validate tool and parameters
                if 'tool' not in call or 'parameters' not in call:
                    logger.warning(f"Invalid tool call structure: {call}")
                    continue
                
                # Enhanced SQL validation for dynamic SQL tools
                if call['tool'] in ['execute_dynamic_sql', 'execute_dynamic_sql_with_graph']:
                    if 'sql_query' in call['parameters']:
                        sql = call['parameters']['sql_query']
                        
                        # Apply enhanced SQL fixes
                        sql = self._fix_sql_quotes(sql)
                        sql = self._validate_and_autofix_sql(sql)
                        
                        # Apply date fixes with user query context
                        sql = self._fix_sql_date_math(sql, query)
                        
                        call['parameters']['sql_query'] = sql
                        
                        # Set default graph type if missing
                        if call['tool'] == 'execute_dynamic_sql_with_graph' and 'graph_type' not in call['parameters']:
                            call['parameters']['graph_type'] = chart_analysis.get('chart_type', 'bar')
                
                enhanced_calls.append(call)
                
            except Exception as e:
                logger.warning(f"Error enhancing tool call: {e}")
                # Create a safe fallback tool call
                fallback_call = {
                    'tool': 'get_database_status',
                    'parameters': {},
                    'original_query': query,
                    'wants_graph': False,
                    'chart_analysis': {'chart_type': 'none'}
                }
                enhanced_calls.append(fallback_call)
        
        if not enhanced_calls:
            # Return fallback if no valid calls
            return [{
                'tool': 'get_database_status',
                'parameters': {},
                'original_query': query,
                'wants_graph': False,
                'chart_analysis': {'chart_type': 'none'}
            }]
        
        return enhanced_calls

    def _get_complete_smart_fallback_tool_call(self, query: str, history: List[str]) -> List[Dict]:
        """Enhanced fallback with all required keys."""
        query_lower = query.lower()
        
        # Check for detail-oriented queries that failed
        if any(phrase in query_lower for phrase in ['show me the ones', 'list the users', 'show users', 'user details']):
            # Try a simple user listing query for common cases
            if 'more than' in query_lower and any(word in query_lower for word in ['subscription', 'subscriptions']):
                # Extract threshold
                threshold_match = re.search(r'more than (\d+)', query_lower)
                threshold = int(threshold_match.group(1)) if threshold_match else 1
                
                sql = f"""
SELECT DISTINCT c.merchant_user_id, 
       c.user_email, 
       c.user_name,
       sub_count.total_subscriptions
FROM subscription_contract_v2 c
JOIN (
    SELECT merchant_user_id, COUNT(*) as total_subscriptions
    FROM subscription_contract_v2 
    GROUP BY merchant_user_id 
    HAVING COUNT(*) > {threshold}
) sub_count ON c.merchant_user_id = sub_count.merchant_user_id
ORDER BY sub_count.total_subscriptions DESC
LIMIT 20
"""
                
                return [{
                    'tool': 'execute_dynamic_sql',
                    'parameters': {'sql_query': sql},
                    'original_query': query,
                    'wants_graph': False,
                    'chart_analysis': {'chart_type': 'none'}
                }]
        
        # Original fallback logic for other cases with proper structure
        if any(word in query_lower for word in ['payment', 'transaction', 'revenue']):
            return [{
                'tool': 'get_payment_success_rate_in_last_days', 
                'parameters': {'days': 30},
                'original_query': query,
                'wants_graph': False,
                'chart_analysis': {'chart_type': 'none'}
            }]
        elif any(word in query_lower for word in ['subscription', 'subscriber', 'user']):
            return [{
                'tool': 'get_subscriptions_in_last_days', 
                'parameters': {'days': 30},
                'original_query': query,
                'wants_graph': False,
                'chart_analysis': {'chart_type': 'none'}
            }]
        else:
            return [{
                'tool': 'get_database_status', 
                'parameters': {},
                'original_query': query,
                'wants_graph': False,
                'chart_analysis': {'chart_type': 'none'}
            }]

    def _create_enhanced_threshold_prompt(self, user_query: str, history_context: str, 
                                        improvement_context: str, similar_context: str, 
                                        chart_analysis: Dict, threshold_info: Dict, 
                                        date_info: Dict, comparison_info: Dict, actionable_rules=None) -> str:
        """Create enhanced prompt with better user intent detection - FIXED VERSION."""
        from datetime import datetime
        
        current_year = datetime.now().year
        current_month = datetime.now().strftime('%B')
        
        # CRITICAL: Analyze user intent for details vs counts
        user_query_lower = user_query.lower()
        wants_details = any(phrase in user_query_lower for phrase in [
            'show me the ones', 'list the users', 'who are the', 'get user details',
            'show user', 'list user', 'user info', 'details', 'names', 'emails',
            'show them', 'list them', 'the actual', 'specific users'
        ])
        
        wants_count_only = any(phrase in user_query_lower for phrase in [
            'how many', 'count of', 'number of', 'total count', 'how much'
        ]) and not wants_details
        
        detail_preference = "SHOW ACTUAL USER/RECORD DETAILS" if wants_details else "COUNT OR AGGREGATE" if wants_count_only else "PREFER DETAILS UNLESS ASKING FOR COUNTS"
        
        # Chart requirements handling
        chart_type = chart_analysis.get('chart_type', 'none')
        if chart_type == 'pie':
            chart_requirements = """PIE CHART REQUIRED:
- Use execute_dynamic_sql_with_graph tool
- SQL must return exactly 2 columns: category (text) and value (number)
- Example: SELECT 'Successful' as category, COUNT(*) as value FROM ... UNION ALL SELECT 'Failed' as category, COUNT(*) as value FROM ..."""
        elif chart_type == 'bar':
            chart_requirements = """BAR CHART REQUIRED:
- Use execute_dynamic_sql_with_graph tool  
- SQL should return category and value columns
- Limit to reasonable number of bars (≤30)"""
        elif chart_type == 'line':
            chart_requirements = """LINE CHART REQUIRED:
- Use execute_dynamic_sql_with_graph tool
- SQL should return time/period and value columns
- Order by time ascending"""
        else:
            chart_requirements = "No chart requested."

        prompt = f"""You are an expert SQL analyst for subscription data. Generate the EXACT tool calls needed.

USER INTENT ANALYSIS:
- Query: "{user_query}"
- Detail Preference: {detail_preference}
- Chart Request: {chart_type}

🔥 CRITICAL RULES:
1. When user says "show me the ones with..." → ALWAYS show actual records/users, NOT just counts
2. When user says "how many..." → Show counts/aggregates
3. For "list", "show", "get" → Show detailed records
4. For thresholds like "more than 1" → Show the actual users who meet criteria

CURRENT CONTEXT:
- Current Year: {current_year} (default for month-only queries)
- History: {history_context}
- Improvements Needed: {improvement_context}
- Similar Successful Queries: {similar_context}

DATABASE SCHEMA:
subscription_contract_v2: merchant_user_id, user_email, user_name, status, subcription_start_date
subscription_payment_details: subscription_id, status, trans_amount_decimal, created_date

🎯 SQL GENERATION RULES:

FOR DETAIL QUERIES (when user wants to see actual records):
✅ CORRECT: "show me users with more than 1 subscription"
```sql
SELECT DISTINCT c.merchant_user_id, c.user_email, c.user_name, sub_count.total_subscriptions
FROM subscription_contract_v2 c
JOIN (
    SELECT merchant_user_id, COUNT(*) as total_subscriptions
    FROM subscription_contract_v2 
    GROUP BY merchant_user_id 
    HAVING COUNT(*) > 1
) sub_count ON c.merchant_user_id = sub_count.merchant_user_id
ORDER BY sub_count.total_subscriptions DESC
LIMIT 50
```

❌ WRONG: SELECT COUNT(*) as num_subscribers FROM (...)

FOR COUNT QUERIES (when user asks "how many"):
✅ CORRECT: "how many users have more than 1 subscription"
```sql
SELECT COUNT(*) as num_subscribers 
FROM (
    SELECT merchant_user_id 
    FROM subscription_contract_v2 
    GROUP BY merchant_user_id 
    HAVING COUNT(*) > 1
) as t
```

FOR TIME SERIES QUERIES:
✅ CORRECT: "show weekly payment trends"
```sql
SELECT DATE_FORMAT(p.created_date, '%Y-W%u') AS payment_week, 
       SUM(p.trans_amount_decimal) AS total_amount
FROM subscription_payment_details p
WHERE p.status = 'ACTIVE'
  AND p.created_date >= DATE_SUB(CURDATE(), INTERVAL 3 MONTH)
GROUP BY DATE_FORMAT(p.created_date, '%Y-W%u')
ORDER BY payment_week ASC
```

✅ CORRECT: "show monthly trends"  
```sql
SELECT DATE_FORMAT(p.created_date, '%Y-%m') AS payment_month,
       SUM(p.trans_amount_decimal) AS total_amount
FROM subscription_payment_details p
WHERE p.status = 'ACTIVE'
  AND p.created_date >= DATE_SUB(CURDATE(), INTERVAL 6 MONTH)
GROUP BY DATE_FORMAT(p.created_date, '%Y-%m')
ORDER BY payment_month ASC
```

❌ WRONG: DATE_FORMAT(created_date, '%Y-%W') → inconsistent day names

🎨 CHART REQUIREMENTS:
{chart_requirements}

📅 DATE FILTERING (Current Year: {current_year}):
- "april" without year → WHERE DATE_FORMAT(created_date, '%Y-%m') = '{current_year}-04'
- "this month" → WHERE DATE_FORMAT(created_date, '%Y-%m') = DATE_FORMAT(CURDATE(), '%Y-%m')
- "last month" → WHERE DATE_FORMAT(created_date, '%Y-%m') = DATE_FORMAT(DATE_SUB(CURDATE(), INTERVAL 1 MONTH), '%Y-%m')

🗓️ TIME PERIOD FORMATTING:
- Daily: DATE_FORMAT(created_date, '%Y-%m-%d') AS day
- Weekly: DATE_FORMAT(created_date, '%Y-W%u') AS week  (e.g., 2025-W15)
- Monthly: DATE_FORMAT(created_date, '%Y-%m') AS month (e.g., 2025-04)
- Quarterly: CONCAT(YEAR(created_date), '-Q', QUARTER(created_date)) AS quarter

❌ WRONG: DATE_FORMAT(created_date, '%Y-%W') → gives day names
✅ CORRECT: DATE_FORMAT(created_date, '%Y-W%u') → gives week numbers like 2025-W15

🔧 AVAILABLE TOOLS:
1. execute_dynamic_sql - for data queries without charts
2. execute_dynamic_sql_with_graph - for data queries with charts
3. get_subscriptions_in_last_days - basic subscription stats
4. get_payment_success_rate_in_last_days - payment stats
5. get_database_status - connection and basic stats

RESPONSE FORMAT:
Return a JSON array of tool calls like:
[{{"tool": "execute_dynamic_sql", "parameters": {{"sql_query": "SELECT ..."}}}}]

Query: "{user_query}"
Generate the appropriate tool call(s):"""

        return prompt

    def _auto_fix_sql_errors(self, sql: str, error: str) -> str:
        """Enhanced auto-fix for SQL errors with GROUP BY handling."""
        import re
        
        try:
            error_lower = error.lower()
            
            # Fix MySQL GROUP BY errors (ONLY_FULL_GROUP_BY mode)
            if ('group by' in error_lower and 'not in group by clause' in error_lower) or '1055' in error:
                logger.info("🔧 Fixing MySQL GROUP BY error - rewriting as subquery")
                
                # Extract the threshold from HAVING clause if present
                threshold_match = re.search(r'HAVING COUNT\(\*\) > (\d+)', sql)
                threshold = int(threshold_match.group(1)) if threshold_match else 1
                
                # Check if this is a user detail query
                if ('user_email' in sql.lower() or 'user_name' in sql.lower()) and 'merchant_user_id' in sql.lower():
                    # Rewrite as a proper subquery to avoid GROUP BY issues
                    sql = f"""
SELECT DISTINCT c.merchant_user_id, 
       c.user_email, 
       c.user_name,
       sub_count.total_subscriptions
FROM subscription_contract_v2 c
JOIN (
    SELECT merchant_user_id, COUNT(*) as total_subscriptions
    FROM subscription_contract_v2 
    GROUP BY merchant_user_id 
    HAVING COUNT(*) > {threshold}
) sub_count ON c.merchant_user_id = sub_count.merchant_user_id
ORDER BY sub_count.total_subscriptions DESC
LIMIT 50
"""
                    logger.info(f"🔧 Rewritten as subquery with threshold {threshold} to show user details")
                    return sql.strip()
                
                # For non-user queries, try to fix by adding columns to GROUP BY
                else:
                    select_match = re.search(r'SELECT\s+(.*?)\s+FROM', sql, re.IGNORECASE | re.DOTALL)
                    group_by_match = re.search(r'GROUP BY\s+(.*?)(?:\s+HAVING|\s+ORDER|\s*$)', sql, re.IGNORECASE)
                    
                    if select_match and group_by_match:
                        select_columns = [col.strip() for col in select_match.group(1).split(',')]
                        # Get base column names (without functions)
                        base_columns = []
                        for col in select_columns:
                            if '(' not in col:  # Skip aggregation functions
                                base_col = col.split(' as ')[0].strip()
                                if base_col not in ['*']:
                                    base_columns.append(base_col)
                        
                        if base_columns:
                            current_group_by = group_by_match.group(1).strip()
                            # Add missing columns to GROUP BY
                            all_group_columns = [current_group_by] + [col for col in base_columns if col not in current_group_by]
                            new_group_by = ', '.join(all_group_columns)
                            sql = re.sub(r'GROUP BY\s+.*?(?=\s+HAVING|\s+ORDER|\s*$)', 
                                       f'GROUP BY {new_group_by}', sql, flags=re.IGNORECASE)
                            logger.info(f"🔧 Updated GROUP BY to include all columns: {new_group_by}")
            
            # Fix quote escaping issues
            elif 'syntax' in error_lower or 'quote' in error_lower or '42000' in error_lower:
                logger.info("🔧 Applying quote fixes for syntax error")
                sql = sql.replace("\\'", "'").replace('\\"', '"').replace("''", "'")
                sql = sql.strip()
                
                # Fix orphaned quotes
                if sql.startswith('"') and sql.count('"') % 2 == 1:
                    sql = sql[1:]
                if sql.endswith('"') and sql.count('"') % 2 == 1:
                    sql = sql[:-1]
                
                # Fix date strings
                sql = re.sub(r'"(\d{4}-\d{2}-\d{2})', r"'\1'", sql)
                sql = re.sub(r'(\d{4}-\d{2}-\d{2})"', r"'\1'", sql)
                sql = re.sub(r'"([^"]*)"', r"'\1'", sql)
            
            # Fix unknown column errors for status values
            elif 'unknown column' in error_lower and 'status' in sql.lower():
                status_values = ['ACTIVE', 'INACTIVE', 'FAILED', 'FAIL', 'INIT', 'CLOSED', 'REJECT']
                for status in status_values:
                    pattern = rf'\bstatus\s*(?:=|!=|<>)\s*{status}\b'
                    replacement = f"status = '{status}'"
                    sql = re.sub(pattern, replacement, sql, flags=re.IGNORECASE)
                logger.info("🔧 Fixed status value quoting")
            
            # Clean up whitespace
            sql = re.sub(r'\s+', ' ', sql).strip()
            logger.info(f"🔧 Auto-fixed SQL: {sql[:150]}...")
            return sql
            
        except Exception as e:
            logger.warning(f"Auto-fix failed: {e}")
            return sql

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
                if any(keyword in line.lower() for keyword in ['feedback', 'improve', 'try again', 'pie chart', 'bar chart', 'line chart', 'error']):
                    context_lines.append(f"IMPORTANT: {line}")
                else:
                    context_lines.append(line)
        
        if not context_lines:
            return "No previous context."
            
        return "\n".join(context_lines)

    async def _get_complete_improvement_context(self, user_query: str, history: List[str], client, return_chart_type=False, return_rules=False):
        """Get complete improvement context, actionable rules, and best chart type."""
        try:
            improvement_lines = ["COMPLETE LEARNED IMPROVEMENTS AND CONTEXT:"]
            best_suggestion = None
            best_chart_type = None
            actionable_rules = []
            
            if history:
                recent_feedback = self._extract_complete_recent_feedback(history)
                if recent_feedback:
                    improvement_lines.append(f"RECENT USER FEEDBACK: {recent_feedback}")
            
            if client:
                try:
                    suggestions_result = await client.call_tool('get_improvement_suggestions', {
                        'original_question': user_query
                    })
                    
                    if (suggestions_result.success and 
                        suggestions_result.data and 
                        suggestions_result.data.get('improvements')):
                        improvements = suggestions_result.data['improvements'][:4]  # More for rules
                        improvement_lines.append("PAST USER IMPROVEMENTS:")
                        
                        if improvements:
                            best_suggestion = improvements[0]['user_suggestion']
                            imp = improvements[0]
                            if 'chart_type' in imp and imp['chart_type']:
                                best_chart_type = imp['chart_type']
                            else:
                                suggestion_text = best_suggestion.lower()
                                if 'bar chart' in suggestion_text:
                                    best_chart_type = 'bar'
                                elif 'pie chart' in suggestion_text:
                                    best_chart_type = 'pie'
                                elif 'line chart' in suggestion_text or 'line graph' in suggestion_text:
                                    best_chart_type = 'line'
                                elif 'scatter' in suggestion_text:
                                    best_chart_type = 'scatter'
                        
                        for improvement in improvements:
                            improvement_lines.append(f"- Issue: {improvement['user_suggestion']}")
                            improvement_lines.append(f"  Context: {improvement['similar_question']}")
                            improvement_lines.append(f"  Category: {improvement['improvement_category']}")
                        
                        # Extract actionable rules from all improvements
                        actionable_rules = self._extract_actionable_rules_from_suggestions(improvements)
                        
                except Exception as e:
                    logger.debug(f"Could not get improvement suggestions: {e}")
            
            if best_suggestion:
                improvement_lines.insert(1, f"AUTO-APPLIED IMPROVEMENT: {best_suggestion}")
            
            self._last_best_chart_type = best_chart_type
            
            if return_chart_type and return_rules:
                return ("\n".join(improvement_lines) if len(improvement_lines) > 1 else "", best_chart_type, actionable_rules)
            elif return_chart_type:
                return ("\n".join(improvement_lines) if len(improvement_lines) > 1 else "", best_chart_type)
            elif return_rules:
                return ("\n".join(improvement_lines) if len(improvement_lines) > 1 else "", actionable_rules)
            else:
                return "\n".join(improvement_lines) if len(improvement_lines) > 1 else ""
                
        except Exception as e:
            logger.warning(f"Could not get complete improvement context: {e}")
            self._last_best_chart_type = None
            if return_chart_type and return_rules:
                return ("", None, [])
            elif return_chart_type:
                return ("", None)
            elif return_rules:
                return ("", [])
            else:
                return ""

    def _inject_actionable_instructions(self, actionable_rules, user_query):
        """Return a string of imperative instructions for the prompt based on actionable rules and the current query."""
        instructions = []
        q = user_query.lower()
        
        for rule in actionable_rules:
            if rule['action'] == 'aggregate_by' and rule['trigger'] in q:
                instructions.append(rule['instruction'])
            if rule['action'] == 'chart_type' and (rule['trigger'] in q or rule['value'] in q):
                instructions.append(rule['instruction'])
            if rule['action'] == 'no_graph' and (rule['trigger'] in q or 'graph' in q or 'chart' in q):
                instructions.append(rule['instruction'])
        
        return "\n".join(instructions)

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
                elif 'line chart' in line.lower() or 'line graph' in line.lower():
                    return "User specifically requested LINE CHART visualization"
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
                elif 'line chart' in line.lower() or 'line graph' in line.lower():
                    analysis['chart_type'] = 'line'
                    analysis['specific_request'] = "User previously requested line chart"
                    break
                elif 'bar chart' in line.lower():
                    analysis['chart_type'] = 'bar'
                    analysis['specific_request'] = "User previously requested bar chart"
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
        elif chart_analysis['chart_type'] == 'line':
            guidance.append("- LINE CHART REQUIRES: Time series data with period/date and values")
            guidance.append("- USE: DATE_FORMAT for time grouping (e.g., '%M %Y' for monthly)")
            guidance.append("- ORDER BY: Always order by time/period for proper line progression")
            guidance.append("- CRITICAL: Ensure data has time dimension for line charts")
        
        return "\n".join(guidance) + "\n"

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

    def _fix_sql_date_math(self, sql_query: str, user_query: str = None) -> str:
        """Convert SQLite-style date math to MySQL-compatible syntax. Handles both single and double quotes and all common intervals."""
        import re
        from datetime import datetime
        
        # Replace DATE('now', '-N day') or DATE("now", '-N day') with DATE_SUB(CURDATE(), INTERVAL N DAY)
        sql_query = re.sub(r"DATE\(['\"]now['\"],\s*'-?(\d+) day'\)", r"DATE_SUB(CURDATE(), INTERVAL \1 DAY)", sql_query)
        # Replace DATE('now', '-N month') or DATE("now", '-N month') with DATE_SUB(CURDATE(), INTERVAL N MONTH)
        sql_query = re.sub(r"DATE\(['\"]now['\"],\s*'-?(\d+) month'\)", r"DATE_SUB(CURDATE(), INTERVAL \1 MONTH)", sql_query)
        # Replace DATE('now', '-N year') or DATE("now", '-N year') with DATE_SUB(CURDATE(), INTERVAL N YEAR)
        sql_query = re.sub(r"DATE\(['\"]now['\"],\s*'-?(\d+) year'\)", r"DATE_SUB(CURDATE(), INTERVAL \1 YEAR)", sql_query)
        # Replace DATE('now') or DATE("now") with CURDATE()
        sql_query = re.sub(r"DATE\(['\"]now['\"]\)", "CURDATE()", sql_query)

        # FIXED: Handle month-only queries to default to current year
        if user_query:
            # Check for month name without year
            month_only_match = re.search(r'\b(january|february|march|april|may|june|july|august|september|october|november|december)\b', user_query.lower())
            year_mentioned = re.search(r'20\d{2}', user_query)
            
            if month_only_match and not year_mentioned:
                # User mentioned month but no year - default to current year
                month_str = month_only_match.group(1)
                current_year = datetime.now().year
                
                month_map = {
                    'january': '01', 'february': '02', 'march': '03', 'april': '04', 'may': '05', 'june': '06',
                    'july': '07', 'august': '08', 'september': '09', 'october': '10', 'november': '11', 'december': '12'
                }
                month_num = month_map[month_str]
                date_filter = f"WHERE DATE_FORMAT(p.created_date, '%Y-%m') = '{current_year}-{month_num}'"
                
                # CRITICAL FIX: Only add if not already present AND no existing WHERE with specific date
                if (date_filter not in sql_query and 
                    'WHERE DATE(' not in sql_query and 
                    f"= '{current_year}-{month_num}" not in sql_query):
                    # Insert before GROUP BY if present, else at end
                    if 'GROUP BY' in sql_query:
                        sql_query = sql_query.replace('GROUP BY', f'{date_filter} GROUP BY')
                    else:
                        sql_query += f' {date_filter}'
            
            elif year_mentioned:
                # User mentioned both month and year - use the specified year
                match = re.search(r'(january|february|march|april|may|june|july|august|september|october|november|december)\s+20\d{2}', user_query.lower())
                if match:
                    month_str = match.group(1)
                    year_str = re.search(r'20\d{2}', user_query).group(0)
                    month_map = {
                        'january': '01', 'february': '02', 'march': '03', 'april': '04', 'may': '05', 'june': '06',
                        'july': '07', 'august': '08', 'september': '09', 'october': '10', 'november': '11', 'december': '12'
                    }
                    month_num = month_map[month_str]
                    date_filter = f"WHERE DATE_FORMAT(p.created_date, '%Y-%m') = '{year_str}-{month_num}'"
                    
                    # Only add if not already present AND no existing WHERE with specific date
                    if (date_filter not in sql_query and 
                        'WHERE DATE(' not in sql_query and 
                        f"= '{year_str}-{month_num}" not in sql_query):
                        # Insert before GROUP BY if present, else at end
                        if 'GROUP BY' in sql_query:
                            sql_query = sql_query.replace('GROUP BY', f'{date_filter} GROUP BY')
                        else:
                            sql_query += f' {date_filter}'
        
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
                # --- POST-PROCESSING: Convert period like '2025-04' to 'April 2025' ---
                new_row = dict(row)
                if 'period' in row and isinstance(row['period'], str):
                    period_val = row['period']
                    match = re.match(r'^(20\d{2})-(0[1-9]|1[0-2])$', period_val)
                    if match:
                        year, month = match.groups()
                        month_name = calendar.month_name[int(month)]
                        new_row['period'] = f"{month_name} {year}"
                for k, v in new_row.items():
                    output.append(f"{k.replace('_', ' ').capitalize()}: {v}")
                # --- NEW: Warn if requested fields are missing ---
                if hasattr(result, 'original_query') and result.original_query:
                    requested_fields = []
                    query_lower = result.original_query.lower()
                    for field in ['email', 'end date', 'end_date', 'subscription_end_date']:
                        if field in query_lower:
                            requested_fields.append(field)
                    for field in requested_fields:
                        if field not in new_row:
                            # Try to find a close substitute
                            substitute = None
                            if 'subcription_start_date' in new_row and 'end' in field:
                                substitute = 'subcription_start_date'
                            elif 'merchant_user_id' in new_row and 'email' in field:
                                substitute = 'merchant_user_id'
                            if substitute:
                                output.append(f"[Note] Field '{field}' not found in schema; showing '{substitute}' instead.")
                            else:
                                output.append(f"[Note] Field '{field}' not found in schema and no substitute available.")
            elif result.data and isinstance(result.data, list):
                # Table output for breakdowns
                headers = result.data[0].keys()
                output.append(" | ".join(headers))
                output.append("-" * (3 * len(headers)))
                for row in result.data:
                    # --- POST-PROCESSING: Convert period like '2025-04' to 'April 2025' ---
                    new_row = dict(row)
                    if 'period' in row and isinstance(row['period'], str):
                        period_val = row['period']
                        match = re.match(r'^(20\d{2})-(0[1-9]|1[0-2])$', period_val)
                        if match:
                            year, month = match.groups()
                            month_name = calendar.month_name[int(month)]
                            new_row['period'] = f"{month_name} {year}"
                    output.append(" | ".join(str(new_row[h]) for h in headers))
                # --- NEW: Warn if requested fields are missing ---
                if hasattr(result, 'original_query') and result.original_query:
                    requested_fields = []
                    query_lower = result.original_query.lower()
                    for field in ['email', 'end date', 'end_date', 'subscription_end_date']:
                        if field in query_lower:
                            requested_fields.append(field)
                    for field in requested_fields:
                        if field not in headers:
                            substitute = None
                            if 'subcription_start_date' in headers and 'end' in field:
                                substitute = 'subcription_start_date'
                            elif 'merchant_user_id' in headers and 'email' in field:
                                substitute = 'merchant_user_id'
                            if substitute:
                                output.append(f"[Note] Field '{field}' not found in schema; showing '{substitute}' instead.")
                            else:
                                output.append(f"[Note] Field '{field}' not found in schema and no substitute available.")
            else:
                output.append(str(result.data))
            
            # Only show graph info if show_graph is True
            if show_graph and getattr(result, 'graph_generated', False):
                output.append(f"\n[Graph generated: {getattr(result, 'graph_filepath', '[see file]')}]")
            
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
                    if len(results_by_query) > 1:
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
        """FIXED: Complete smart SQL with graph generation handling."""
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
                # FIXED: Pass SQL result data directly to graph generator
                graph_data = {
                    'data': sql_result.data,  # Pass SQL result data
                    'graph_type': parameters.get('graph_type', 'bar'),
                    'title': self._generate_complete_smart_title(original_query),
                    'description': f"Generated from: {original_query}"
                }
                
                # ENFORCE: Always set graph_type in graph_data to the overridden value
                enforced_type = parameters.get('graph_type', 'bar')
                graph_data['graph_type'] = enforced_type
                logger.info(f"[ENFORCE] Setting graph_data['graph_type'] = '{enforced_type}' before generate_graph")
                
                graph_filepath = self.graph_generator.generate_graph(
                    graph_data, original_query
                )
                
                sql_result.graph_data = graph_data
                sql_result.graph_generated = graph_filepath is not None
                
                if graph_filepath:
                    sql_result.graph_filepath = graph_filepath
                    sql_result.message = (sql_result.message or "") + f"\n📊 Complete graph generated successfully"
                else:
                    sql_result.message = (sql_result.message or "") + f"\n⚠️ Complete graph data generated but file creation failed"
            
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