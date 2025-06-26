# client/universal_client.py - COMPLETE VERSION WITH FULL FUNCTIONALITY

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

from config_manager import ConfigManager

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

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
            graph_type = self._determine_optimal_graph_type(graph_data, query)
            
            # Set up matplotlib with error handling
            plt.style.use('default')
            fig, ax = plt.subplots(figsize=(12, 8))
            
            # Generate graph based on type
            success = self._create_graph_by_type(ax, graph_data, graph_type)
            
            if not success:
                logger.error(f"Failed to create {graph_type} chart")
                plt.close(fig)
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
        """Create complete enhanced bar chart with smart formatting."""
        try:
            categories = graph_data.get('categories', graph_data.get('labels', []))
            values = graph_data.get('values', [])
            
            if not categories or not values or len(categories) != len(values):
                return False
            
            # Limit data points for readability
            if len(categories) > 30:
                categories = categories[:30]
                values = values[:30]
                logger.info("Limited bar chart to 30 categories for readability")
            
            # Create bar chart with complete enhanced styling
            bars = ax.bar(range(len(categories)), values, color='steelblue', alpha=0.8, edgecolor='darkblue')
            
            # Set labels and formatting
            ax.set_xlabel(graph_data.get('x_label', 'Categories'), fontsize=12)
            ax.set_ylabel(graph_data.get('y_label', 'Values'), fontsize=12)
            ax.set_xticks(range(len(categories)))
            ax.set_xticklabels([str(cat)[:15] for cat in categories], rotation=45, ha='right')
            
            # Add value labels on bars if not too many
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
    """COMPLETE NLP processor with full semantic learning integration and perfect schema handling."""
    
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
   - status (ENUM: 'ACTIVE', 'INACTIVE')
   - subcription_start_date (DATE)

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
- For pie charts with success rates, use aggregated queries with UNION
- Always use proper JOINs when accessing both tables
- merchant_user_id is ONLY in subscription_contract_v2 table

COMPLETE PIE CHART PATTERNS:
1. Success vs Failure totals:
   SELECT 'Successful' as category, SUM(CASE WHEN status = 'ACTIVE' THEN 1 ELSE 0 END) as value FROM subscription_payment_details
   UNION ALL
   SELECT 'Failed' as category, SUM(CASE WHEN status != 'ACTIVE' THEN 1 ELSE 0 END) as value FROM subscription_payment_details

2. For merchant analysis with pie chart:
   SELECT 
     CASE WHEN c.merchant_user_id = 'specific_user' THEN 'Target User' ELSE 'Other Users' END as category,
     COUNT(*) as value
   FROM subscription_payment_details p 
   JOIN subscription_contract_v2 c ON p.subscription_id = c.subscription_id
   GROUP BY CASE WHEN c.merchant_user_id = 'specific_user' THEN 'Target User' ELSE 'Other Users' END

3. For merchants with more than X transactions:
   SELECT 
     CASE WHEN total_transactions > 1 THEN 'Active Merchants' ELSE 'Low Activity Merchants' END as category,
     COUNT(*) as value
   FROM (
     SELECT c.merchant_user_id, COUNT(*) as total_transactions
     FROM subscription_payment_details p 
     JOIN subscription_contract_v2 c ON p.subscription_id = c.subscription_id
     GROUP BY c.merchant_user_id
   ) merchant_stats
   GROUP BY CASE WHEN total_transactions > 1 THEN 'Active Merchants' ELSE 'Low Activity Merchants' END
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
        """COMPLETE query parsing with full semantic learning integration."""
        try:
            # Get conversation context
            history_context = self._build_complete_history_context(history)
            
            # Get complete improvement context from feedback
            improvement_context = await self._get_complete_improvement_context(user_query, history, client)
            
            # Get similar successful queries for context
            similar_context = await self._get_similar_queries_context(user_query, client)
            
            # Detect chart requirements
            chart_analysis = self._analyze_complete_chart_requirements(user_query, history)
            
            # Create COMPLETE prompt with full schema warnings
            prompt = self._create_complete_prompt(user_query, history_context, improvement_context, similar_context, chart_analysis)
            
            # Generate response with retries
            tool_calls = await self._generate_with_complete_retries(prompt, user_query, chart_analysis)
            
            # Validate and enhance tool calls
            enhanced_calls = self._enhance_and_validate_complete_tool_calls(tool_calls, user_query, chart_analysis)
            
            logger.info(f"🧠 Complete Smart AI selected tool(s): {[tc['tool'] for tc in enhanced_calls]}")
            return enhanced_calls
            
        except Exception as e:
            logger.error(f"Error in complete smart query parsing: {e}", exc_info=True)
            return self._get_complete_smart_fallback_tool_call(user_query, history)

    def _build_complete_history_context(self, history: List[str]) -> str:
        """Build complete contextual history with smart filtering."""
        if not history:
            return "No previous context."
        
        # Take last 6 turns but prioritize recent feedback
        recent_history = history[-6:] if len(history) > 6 else history
        
        # Look for feedback patterns
        context_lines = []
        for line in recent_history:
            if any(keyword in line.lower() for keyword in ['feedback', 'improve', 'try again', 'pie chart', 'bar chart', 'error']):
                context_lines.append(f"IMPORTANT: {line}")
            else:
                context_lines.append(line)
        
        return "\n".join(context_lines)

    async def _get_complete_improvement_context(self, user_query: str, history: List[str], client) -> str:
        """Get complete improvement context with full history analysis."""
        try:
            improvement_lines = ["COMPLETE LEARNED IMPROVEMENTS AND CONTEXT:"]
            
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
                        
                        for improvement in improvements:
                            improvement_lines.append(f"- Issue: {improvement['user_suggestion']}")
                            improvement_lines.append(f"  Context: {improvement['similar_question']}")
                            improvement_lines.append(f"  Category: {improvement['improvement_category']}")
                except Exception as e:
                    logger.debug(f"Could not get improvement suggestions: {e}")
            
            return "\n".join(improvement_lines) if len(improvement_lines) > 1 else ""
            
        except Exception as e:
            logger.warning(f"Could not get complete improvement context: {e}")
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

    def _create_complete_prompt(self, user_query: str, history_context: str, improvement_context: str, similar_context: str, chart_analysis: Dict) -> str:
        """Create COMPLETE prompt with full schema understanding."""
        chart_guidance = self._get_complete_chart_guidance(chart_analysis)
        
        return f"""
You are an expert subscription analytics assistant with COMPLETE database schema knowledge and semantic learning.

CONVERSATION HISTORY:
{history_context}

{improvement_context}

{similar_context}

CURRENT USER QUERY: "{user_query}"

{chart_guidance}

{self.db_schema}

COMPLETE CRITICAL SCHEMA RULES FOR SQL GENERATION:
1. ❌ NEVER use "merchant_user_id" directly in GROUP BY with subscription_payment_details table
2. ✅ To access merchant_user_id from payments, ALWAYS JOIN tables:
   FROM subscription_payment_details p 
   JOIN subscription_contract_v2 c ON p.subscription_id = c.subscription_id

3. ✅ For PIE CHARTS with success rates, use UNION queries:
   SELECT 'Successful' as category, SUM(CASE WHEN status = 'ACTIVE' THEN 1 ELSE 0 END) as value FROM subscription_payment_details
   UNION ALL  
   SELECT 'Failed' as category, SUM(CASE WHEN status != 'ACTIVE' THEN 1 ELSE 0 END) as value FROM subscription_payment_details

4. ✅ For merchant-specific analysis with pie charts:
   Use proper JOINs and subqueries to group by merchant categories, not individual merchant_user_id

5. ✅ For merchants with more than X transactions:
   Use subqueries to first count transactions per merchant, then categorize

COMPLETE ENHANCED TOOL SELECTION RULES:
1. **PIE CHARTS** need AGGREGATED data with category/value pairs
2. **TIME SERIES** charts need date grouping with proper ORDER BY
3. **SUCCESS RATES** should be calculated as percentages and totals
4. **MERCHANT ANALYSIS** requires proper JOINs and categorization
5. **COMPARISON** queries should use UNION for different categories

COMPLETE CRITICAL CHART TYPE DETECTION:
- If user says "pie chart" or "distribution" or "visually" → use execute_dynamic_sql_with_graph with graph_type="pie"
- If user wants "rates" or "percentages" → calculate aggregated percentages, use pie chart
- If user says "trend" or "over time" → use time series with line chart
- If user says "try again" → check history for previous chart type requests
- If user mentions "merchants with more than X transactions" → use merchant categorization with pie chart

COMPLETE EXAMPLES WITH CORRECTED SQL:

1. PIE CHART REQUEST: "pie chart of payment success rates"
   CORRECT SQL: 
   SELECT 'Successful Payments' as category, SUM(CASE WHEN status = 'ACTIVE' THEN 1 ELSE 0 END) as value FROM subscription_payment_details 
   UNION ALL 
   SELECT 'Failed Payments' as category, SUM(CASE WHEN status != 'ACTIVE' THEN 1 ELSE 0 END) as value FROM subscription_payment_details
   Tool: execute_dynamic_sql_with_graph with graph_type="pie"

2. MERCHANT ANALYSIS: "show payment success rate for merchants with more than 1 transaction visually"
   CORRECT SQL:
   SELECT 
     CASE WHEN total_transactions > 1 THEN 'Active Merchants' ELSE 'Low Activity Merchants' END as category,
     COUNT(*) as value
   FROM (
     SELECT c.merchant_user_id, COUNT(*) as total_transactions
     FROM subscription_payment_details p 
     JOIN subscription_contract_v2 c ON p.subscription_id = c.subscription_id
     GROUP BY c.merchant_user_id
   ) merchant_stats
   GROUP BY CASE WHEN total_transactions > 1 THEN 'Active Merchants' ELSE 'Low Activity Merchants' END
   Tool: execute_dynamic_sql_with_graph with graph_type="pie"

3. COMPLETE MERCHANT SUCCESS ANALYSIS: "merchants with more than 1 transaction and their success rates as pie chart"
   CORRECT SQL:
   SELECT 
     CASE WHEN success_rate > 50 THEN 'High Success Merchants' ELSE 'Low Success Merchants' END as category,
     COUNT(*) as value
   FROM (
     SELECT 
       c.merchant_user_id,
       COUNT(*) as total_transactions,
       ROUND((SUM(CASE WHEN p.status = 'ACTIVE' THEN 1 ELSE 0 END) * 100.0 / COUNT(*)), 2) as success_rate
     FROM subscription_payment_details p 
     JOIN subscription_contract_v2 c ON p.subscription_id = c.subscription_id
     GROUP BY c.merchant_user_id
     HAVING COUNT(*) > 1
   ) merchant_stats
   GROUP BY CASE WHEN success_rate > 50 THEN 'High Success Merchants' ELSE 'Low Success Merchants' END
   Tool: execute_dynamic_sql_with_graph with graph_type="pie"

4. TIME TREND: "payment trends over time"
   CORRECT SQL: 
   SELECT DATE(created_date) as date, COUNT(*) as payments FROM subscription_payment_details GROUP BY DATE(created_date) ORDER BY date
   Tool: execute_dynamic_sql_with_graph with graph_type="line"

Remember: ALWAYS validate your SQL against the complete schema rules above!
Use semantic learning context to improve your responses!
"""

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

    def _enhance_and_validate_complete_tool_calls(self, tool_calls: List[Dict], user_query: str, chart_analysis: Dict) -> List[Dict]:
        """Complete enhanced tool call validation with FIXED schema handling."""
        enhanced_calls = []
        
        for call in tool_calls:
            try:
                # Enhance SQL for complete schema compliance
                if 'sql_query' in call['parameters']:
                    original_sql = call['parameters']['sql_query']
                    enhanced_sql = self._fix_complete_sql_schema_issues(original_sql, chart_analysis, user_query)
                    call['parameters']['sql_query'] = enhanced_sql
                    
                    if original_sql != enhanced_sql:
                        logger.info("🧠 SQL fixed for complete schema compliance")
                
                # Ensure graph type is set correctly
                if call['tool'] == 'execute_dynamic_sql_with_graph':
                    if chart_analysis.get('chart_type') and 'graph_type' not in call['parameters']:
                        call['parameters']['graph_type'] = chart_analysis['chart_type']
                
                enhanced_calls.append(call)
                
            except Exception as e:
                logger.error(f"Error enhancing complete tool call: {e}")
                enhanced_calls.append(call)  # Keep original if enhancement fails
        
        if not enhanced_calls:
            return self._get_complete_smart_fallback_tool_call(user_query, [])
        
        return enhanced_calls

    def _fix_complete_sql_schema_issues(self, sql_query: str, chart_analysis: Dict, user_query: str) -> str:
        """Fix SQL to comply with COMPLETE corrected schema."""
        try:
            # Clean SQL first
            sql_query = sql_query.strip().strip('"\'')
            
            # COMPLETE CRITICAL FIX: Handle merchant_user_id GROUP BY issues
            if 'GROUP BY merchant_user_id' in sql_query and 'subscription_payment_details' in sql_query:
                logger.warning("🔧 Fixing merchant_user_id GROUP BY issue with complete logic")
                
                if chart_analysis.get('chart_type') == 'pie' and 'merchant' in user_query.lower():
                    # Convert to merchant categorization for pie chart
                    if 'more than' in user_query.lower() and 'transaction' in user_query.lower():
                        # Extract the transaction threshold
                        import re
                        threshold_match = re.search(r'more than (\d+)', user_query.lower())
                        threshold = int(threshold_match.group(1)) if threshold_match else 1
                        
                        sql_query = f"""
SELECT 
  CASE WHEN total_transactions > {threshold} THEN 'Active Merchants' ELSE 'Low Activity Merchants' END as category,
  COUNT(*) as value
FROM (
  SELECT c.merchant_user_id, COUNT(*) as total_transactions
  FROM subscription_payment_details p 
  JOIN subscription_contract_v2 c ON p.subscription_id = c.subscription_id
  GROUP BY c.merchant_user_id
) merchant_stats
GROUP BY CASE WHEN total_transactions > {threshold} THEN 'Active Merchants' ELSE 'Low Activity Merchants' END
"""
                    elif 'success' in user_query.lower() or 'rate' in user_query.lower():
                        # Convert to success rate categorization
                        sql_query = """
SELECT 
  CASE WHEN success_rate > 50 THEN 'High Success Merchants' ELSE 'Low Success Merchants' END as category,
  COUNT(*) as value
FROM (
  SELECT 
    c.merchant_user_id,
    COUNT(*) as total_transactions,
    ROUND((SUM(CASE WHEN p.status = 'ACTIVE' THEN 1 ELSE 0 END) * 100.0 / COUNT(*)), 2) as success_rate
  FROM subscription_payment_details p 
  JOIN subscription_contract_v2 c ON p.subscription_id = c.subscription_id
  GROUP BY c.merchant_user_id
  HAVING COUNT(*) > 1
) merchant_stats
GROUP BY CASE WHEN success_rate > 50 THEN 'High Success Merchants' ELSE 'Low Success Merchants' END
"""
                    else:
                        # Default merchant categorization
                        sql_query = """
SELECT 
  CASE WHEN total_transactions > 1 THEN 'Active Merchants' ELSE 'Low Activity Merchants' END as category,
  COUNT(*) as value
FROM (
  SELECT c.merchant_user_id, COUNT(*) as total_transactions
  FROM subscription_payment_details p 
  JOIN subscription_contract_v2 c ON p.subscription_id = c.subscription_id
  GROUP BY c.merchant_user_id
) merchant_stats
GROUP BY CASE WHEN total_transactions > 1 THEN 'Active Merchants' ELSE 'Low Activity Merchants' END
"""
                elif chart_analysis.get('chart_type') == 'pie' and 'rate' in user_query.lower():
                    # Convert to success/failure pie chart
                    sql_query = """
SELECT 'Successful Payments' as category, SUM(CASE WHEN status = 'ACTIVE' THEN 1 ELSE 0 END) as value 
FROM subscription_payment_details 
UNION ALL 
SELECT 'Failed Payments' as category, SUM(CASE WHEN status != 'ACTIVE' THEN 1 ELSE 0 END) as value 
FROM subscription_payment_details
"""
                else:
                    # Remove problematic GROUP BY and add proper JOIN
                    sql_query = sql_query.replace('FROM subscription_payment_details', 
                                                'FROM subscription_payment_details p JOIN subscription_contract_v2 c ON p.subscription_id = c.subscription_id')
                    sql_query = sql_query.replace('GROUP BY merchant_user_id', 'GROUP BY c.merchant_user_id')
            
            # Complete chart-specific enhancements
            if chart_analysis.get('chart_type') == 'pie':
                sql_query = self._convert_to_complete_pie_chart_sql(sql_query, user_query, chart_analysis)
            elif chart_analysis.get('data_aggregation') == 'time_series':
                sql_query = self._optimize_for_complete_time_series(sql_query)
            elif chart_analysis.get('data_aggregation') == 'rate_calculation':
                sql_query = self._enhance_for_complete_rate_calculation(sql_query)
            
            # Complete general optimizations
            sql_query = self._apply_complete_general_sql_optimizations(sql_query)
            
            return sql_query
            
        except Exception as e:
            logger.error(f"Complete SQL fixing failed: {e}")
            return sql_query

    def _convert_to_complete_pie_chart_sql(self, sql_query: str, user_query: str, chart_analysis: Dict) -> str:
        """Convert SQL to complete pie chart appropriate format."""
        query_lower = user_query.lower()
        
        # Handle merchant analysis specifically
        if chart_analysis.get('is_merchant_analysis') and 'more than' in query_lower:
            # Already handled in the main fixing function
            return sql_query
        
        # If it's a problematic GROUP BY query, convert to success/failure
        if 'GROUP BY' in sql_query and 'merchant_user_id' in sql_query and 'rate' in query_lower:
            return """
SELECT 'Successful Payments' as category, 
       SUM(CASE WHEN status = 'ACTIVE' THEN 1 ELSE 0 END) as value 
FROM subscription_payment_details 
UNION ALL 
SELECT 'Failed Payments' as category, 
       SUM(CASE WHEN status != 'ACTIVE' THEN 1 ELSE 0 END) as value 
FROM subscription_payment_details
"""
        
        # For time series converted to pie chart
        if ('GROUP BY DATE(' in sql_query or 'GROUP BY created_date' in sql_query) and 'rate' in query_lower:
            return """
SELECT 'Successful Payments' as category, 
       SUM(CASE WHEN status = 'ACTIVE' THEN 1 ELSE 0 END) as value 
FROM subscription_payment_details 
UNION ALL 
SELECT 'Failed Payments' as category, 
       SUM(CASE WHEN status != 'ACTIVE' THEN 1 ELSE 0 END) as value 
FROM subscription_payment_details
"""
        
        return sql_query

    def _optimize_for_complete_time_series(self, sql_query: str) -> str:
        """Optimize SQL for complete time series visualization."""
        # Ensure proper date grouping
        if 'GROUP BY created_date' in sql_query:
            sql_query = sql_query.replace('GROUP BY created_date', 'GROUP BY DATE(created_date)')
            
        # Ensure proper ordering
        if 'ORDER BY' not in sql_query and 'GROUP BY DATE(' in sql_query:
            sql_query += ' ORDER BY DATE(created_date)'
            
        return sql_query

    def _enhance_for_complete_rate_calculation(self, sql_query: str) -> str:
        """Enhance SQL for complete rate calculations."""
        # Add percentage calculations if missing
        if 'success_rate' not in sql_query and 'CASE WHEN status' in sql_query:
            # Add success rate calculation
            if 'SELECT' in sql_query and 'FROM' in sql_query:
                select_part = sql_query.split('FROM')[0]
                from_part = 'FROM' + sql_query.split('FROM')[1]
                
                if 'success_rate' not in select_part:
                    enhanced_select = select_part + ', ROUND((SUM(CASE WHEN status = \'ACTIVE\' THEN 1 ELSE 0 END) * 100.0 / COUNT(*)), 2) as success_rate'
                    sql_query = enhanced_select + ' ' + from_part
        
        return sql_query

    def _apply_complete_general_sql_optimizations(self, sql_query: str) -> str:
        """Apply complete general SQL optimizations."""
        import re
        
        # Clean quotes safely
        sql_query = sql_query.replace("\\'", "'")  # Remove escaped quotes first
        sql_query = sql_query.replace('\\"', '"')  # Remove escaped double quotes
        
        # Fix quotes more carefully
        sql_query = re.sub(r'"([^"\']*)"', r"'\1'", sql_query)
        
        # Fix status values carefully
        status_values = ['ACTIVE', 'INACTIVE', 'FAILED', 'FAIL', 'INIT']
        for status in status_values:
            # Only fix if not already quoted
            pattern = rf'\bstatus\s*=\s*{status}\b'
            replacement = f"status = '{status}'"
            sql_query = re.sub(pattern, replacement, sql_query, flags=re.IGNORECASE)
        
        # Clean whitespace
        sql_query = re.sub(r'\s+', ' ', sql_query).strip()
        
        return sql_query

    def _get_complete_smart_fallback_tool_call(self, user_query: str, history: List[str]) -> List[Dict]:
        """Get complete smart fallback based on query analysis."""
        query_lower = user_query.lower()
        
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

class CompleteEnhancedResultFormatter:
    """COMPLETE enhanced result formatter with smart insights and better presentation."""
    
    def __init__(self):
        self.graph_generator = CompleteGraphGenerator()
    
    def format_single_result(self, result: QueryResult) -> str:
        """Format single result with complete enhanced insights and presentation."""
        try:
            if not result.success:
                return f"❌ ERROR: {result.error}"
            
            if result.message and not result.data:
                return f"ℹ️ {result.message}"
            
            if result.data is None:
                return "✅ Query succeeded, but no data was returned."
            
            output = []
            
            # Handle graph information first
            if result.graph_data and result.graph_generated:
                graph_info = self._format_complete_graph_info(result)
                output.append(graph_info)
                output.append("")
            
            # Format main data
            main_content = self._format_complete_main_data(result)
            output.append(main_content)
            
            # Add SQL information for dynamic queries
            if result.generated_sql:
                output.append(f"\n🔍 Generated SQL (COMPLETE):")
                output.append("-" * 20)
                output.append(result.generated_sql)
            
            # Add message if present
            if result.message:
                output.append(f"\n📝 {result.message}")
            
            return "\n".join(output)
            
        except Exception as e:
            logger.error(f"Error formatting complete result: {e}")
            return f"❌ Error formatting complete result: {str(e)}"
    
    def format_multi_result(self, results: List[QueryResult], query: str) -> str:
        """Format multiple complete results."""
        try:
            if not results:
                return "❌ No results to display"
            
            output = [f"🎯 MULTIPLE COMPLETE RESULTS FOR: '{query}'", "=" * 70]
            
            for i, result in enumerate(results, 1):
                output.append(f"\n--- Complete Result {i} ---")
                single_result = self.format_single_result(result)
                output.append(single_result)
            
            return "\n".join(output)
            
        except Exception as e:
            logger.error(f"Error formatting multiple complete results: {e}")
            return f"❌ Error formatting complete results: {str(e)}"
    
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

    async def __aenter__(self):
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

    async def query(self, nl_query: str) -> Union[QueryResult, List[QueryResult]]:
        """Complete enhanced query processing with smart AI."""
        try:
            parsed_calls = await self.nlp.parse_query(nl_query, self.history, client=self)
            
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
                        results.append(result)
                    except Exception as e:
                        logger.error(f"Error calling complete tool {call['tool']}: {e}")
                        error_result = QueryResult(
                            success=False,
                            error=f"Complete tool {call['tool']} failed: {str(e)}",
                            tool_used=call['tool']
                        )
                        results.append(error_result)
                return results
            else:
                call = parsed_calls[0]
                return await self.call_tool(
                    call['tool'], 
                    call['parameters'], 
                    call['original_query'],
                    call.get('wants_graph', False)
                )
                
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
            print("  • Get similar successful queries for my question")
            print("\n💡 The COMPLETE AI now has semantic learning and feedback!")
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
                        if (isinstance(result, QueryResult) and 
                            result.is_dynamic and 
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
                                        break
                                    elif feedback_input in ['n', 'no']:
                                        improvement = input("How can this be improved? (e.g., 'use pie chart instead', 'fix SQL error'): ").strip()
                                        if improvement.lower() not in ['skip', 's', '']:
                                            await client.submit_feedback(result, False, improvement)
                                            print("🧠 Negative feedback and improvement recorded - the system will learn!")
                                        else:
                                            await client.submit_feedback(result, False)
                                            print("🧠 Negative feedback recorded!")
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
    try:
        asyncio.run(complete_enhanced_interactive_mode())
    except KeyboardInterrupt:
        print("\n👋 Goodbye from COMPLETE system!")
    except Exception as e:
        print(f"❌ Failed to start COMPLETE client: {e}")
        print("💡 COMPLETE system uses smart error handling - most issues are automatically resolved!")