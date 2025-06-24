# client/universal_client.py - ENHANCED WITH GRAPH GENERATION CAPABILITIES

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

# FIXED: Remove relative import and use absolute import
# Add current directory to path to ensure imports work
from pathlib import Path
current_dir = Path(__file__).parent.absolute()
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

# Now import config manager
from config_manager import ConfigManager

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
    graph_data: Optional[Dict] = None  # NEW: Graph data
    graph_generated: bool = False      # NEW: Flag for graph generation

# Graph Generation Class
class GraphGenerator:
    """Handles graph generation and display."""
    
    def __init__(self):
        # Try multiple locations for the graphs directory
        possible_dirs = [
            Path.cwd() / "generated_graphs",
            Path(__file__).parent / "generated_graphs", 
            Path.home() / "subscription_graphs",
            Path("/tmp") / "subscription_graphs" if os.name != 'nt' else Path.cwd() / "temp_graphs"
        ]
        
        self.graphs_dir = None
        for directory in possible_dirs:
            try:
                directory.mkdir(parents=True, exist_ok=True)
                # Test write permissions
                test_file = directory / "test_write.tmp"
                test_file.write_text("test")
                test_file.unlink()  # Delete test file
                
                self.graphs_dir = directory
                logger.info(f"📊 Graph directory set to: {self.graphs_dir}")
                break
            except Exception as e:
                logger.warning(f"⚠️ Cannot use directory {directory}: {e}")
                continue
        
        if not self.graphs_dir:
            logger.error("❌ Could not create any graph directory!")
            # Fallback to current directory without subdirectory
            self.graphs_dir = Path.cwd()
            logger.info(f"📊 Fallback: Using current directory for graphs: {self.graphs_dir}")
        
    def can_generate_graphs(self) -> bool:
        """Check if graph generation is available."""
        return MATPLOTLIB_AVAILABLE
    
    def should_generate_graph(self, data: List[Dict], query: str) -> bool:
        """Determine if data is suitable for graphing."""
        if not self.can_generate_graphs() or not data:
            return False
        
        # Check for graph indicators in query
        graph_keywords = [
            'chart', 'graph', 'plot', 'visualize', 'show trend', 'compare', 
            'distribution', 'over time', 'by month', 'by year', 'trend',
            'performance', 'analysis', 'breakdown', 'ranking'
        ]
        
        query_lower = query.lower()
        has_graph_keywords = any(keyword in query_lower for keyword in graph_keywords)
        
        # Check data structure - good for graphing if:
        # 1. Has numeric columns
        # 2. Has reasonable number of rows (2-100)
        # 3. Has categorical or date columns for x-axis
        if len(data) < 2 or len(data) > 100:
            return False
        
        # Analyze columns
        columns = list(data[0].keys())
        numeric_cols = []
        categorical_cols = []
        date_cols = []
        
        for col in columns:
            sample_values = [row.get(col) for row in data[:5] if row.get(col) is not None]
            if not sample_values:
                continue
                
            sample_value = sample_values[0]
            
            if isinstance(sample_value, (int, float)):
                numeric_cols.append(col)
            elif 'date' in col.lower() or 'time' in col.lower():
                date_cols.append(col)
            else:
                # Check if it's a reasonable categorical variable
                unique_values = len(set(str(row.get(col)) for row in data))
                if unique_values <= len(data) * 0.8:  # Not too many unique values
                    categorical_cols.append(col)
        
        # Good for graphing if we have at least one numeric column and one categorical/date column
        suitable_for_graphing = len(numeric_cols) >= 1 and (len(categorical_cols) >= 1 or len(date_cols) >= 1)
        
        return has_graph_keywords or suitable_for_graphing
    
    def generate_graph(self, graph_data: Dict, query: str) -> Optional[str]:
        """Generate and save a graph, return the file path."""
        if not self.can_generate_graphs():
            logger.warning("Cannot generate graph - matplotlib not available")
            return None
        
        try:
            # Ensure the directory exists
            if not self.graphs_dir.exists():
                try:
                    self.graphs_dir.mkdir(parents=True, exist_ok=True)
                    logger.info(f"📁 Created graphs directory: {self.graphs_dir}")
                except Exception as e:
                    logger.error(f"❌ Cannot create graphs directory: {e}")
                    # Try current directory as fallback
                    self.graphs_dir = Path.cwd()
                    logger.info(f"📁 Using current directory instead: {self.graphs_dir}")
            
            # Set up matplotlib for non-interactive use
            plt.style.use('default')
            fig, ax = plt.subplots(figsize=(12, 8))
            
            graph_type = graph_data.get('graph_type', 'bar')
            
            if graph_type == 'line':
                self._create_line_chart(ax, graph_data)
            elif graph_type == 'bar':
                self._create_bar_chart(ax, graph_data)
            elif graph_type == 'horizontal_bar':
                self._create_horizontal_bar_chart(ax, graph_data)
            elif graph_type == 'pie':
                self._create_pie_chart(ax, graph_data)
            elif graph_type == 'scatter':
                self._create_scatter_plot(ax, graph_data)
            else:
                logger.warning(f"Unknown graph type: {graph_type}")
                plt.close(fig)
                return None
            
            # Set title and description
            title = graph_data.get('title', 'Data Visualization')
            description = graph_data.get('description', '')
            
            if description:
                fig.suptitle(f"{title}\n{description}", fontsize=12, y=0.98)
                ax.set_title("")  # Remove the axis title since we have suptitle
            else:
                ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
            
            # Apply final layout adjustments to prevent overlapping
            plt.tight_layout()
            
            # Give extra space for rotated labels if needed
            plt.subplots_adjust(bottom=0.15)  # Add space at bottom for rotated x-labels
            
            # Generate filename and full path
            import time
            timestamp = int(time.time())
            filename = f"graph_{graph_type}_{timestamp}.png"
            filepath = self.graphs_dir / filename
            
            # Save the graph with error handling
            try:
                logger.info(f"📊 Saving graph to: {filepath}")
                plt.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white')
                logger.info(f"✅ Graph successfully saved to: {filepath}")
            except Exception as save_error:
                logger.error(f"❌ Failed to save graph to {filepath}: {save_error}")
                # Try saving to current directory as fallback
                fallback_path = Path.cwd() / filename
                try:
                    plt.savefig(fallback_path, dpi=300, bbox_inches='tight', facecolor='white')
                    filepath = fallback_path
                    logger.info(f"✅ Graph saved to fallback location: {filepath}")
                except Exception as fallback_error:
                    logger.error(f"❌ Fallback save also failed: {fallback_error}")
                    plt.close(fig)
                    return None
            
            # Also try to display the graph in a window (if display is available)
            try:
                # Check if we have a display available
                import os
                if os.environ.get('DISPLAY') or os.name == 'nt' or (os.name == 'posix' and os.uname().sysname == 'Darwin'):
                    # Try to show the plot in a window
                    plt.show(block=False)  # Non-blocking show
                    logger.info("📊 Graph displayed in matplotlib window")
            except Exception as e:
                logger.debug(f"Could not display graph in window: {e}")
            
            plt.close(fig)  # Important: close the figure to free memory
            
            # Verify the file was actually created
            if not filepath.exists():
                logger.error(f"❌ Graph file was not created at {filepath}")
                return None
            
            logger.info(f"📊 Graph generation completed: {filepath}")
            return str(filepath)
            
        except Exception as e:
            logger.error(f"❌ Graph generation failed: {e}")
            import traceback
            logger.error(f"❌ Traceback: {traceback.format_exc()}")
            plt.close('all')  # Close any open figures
            return None
    
    def _create_line_chart(self, ax, graph_data):
        """Create a line chart with improved x-axis handling."""
        x_values = graph_data.get('x_values', [])
        y_values = graph_data.get('y_values', [])
        
        ax.plot(x_values, y_values, marker='o', linewidth=2, markersize=6)
        ax.set_xlabel(graph_data.get('x_label', 'X Axis'))
        ax.set_ylabel(graph_data.get('y_label', 'Y Axis'))
        ax.grid(True, alpha=0.3)
        
        # Improved x-axis label handling to prevent overlapping
        if x_values:
            # Check if we have many x-values or long labels
            num_labels = len(x_values)
            max_label_length = max(len(str(x)) for x in x_values) if x_values else 0
            
            if num_labels > 10 or max_label_length > 8:
                # Rotate labels for better readability
                ax.tick_params(axis='x', rotation=45)
                
                # If still too many labels, show every nth label
                if num_labels > 20:
                    step = max(1, num_labels // 15)  # Show ~15 labels max
                    indices = range(0, len(x_values), step)
                    ax.set_xticks([x_values[i] for i in indices])
                    ax.set_xticklabels([x_values[i] for i in indices], rotation=45, ha='right')
            elif max_label_length > 6:
                # Just rotate if labels are long
                ax.tick_params(axis='x', rotation=30)
    
    def _create_bar_chart(self, ax, graph_data):
        """Create a bar chart with improved label handling."""
        categories = graph_data.get('categories', [])
        values = graph_data.get('values', [])
        
        bars = ax.bar(categories, values, color='steelblue', alpha=0.8)
        ax.set_xlabel(graph_data.get('x_label', 'Categories'))
        ax.set_ylabel(graph_data.get('y_label', 'Values'))
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + max(values) * 0.01,
                   f'{value:.1f}' if isinstance(value, float) else str(value),
                   ha='center', va='bottom')
        
        # Improved x-axis label handling to prevent overlapping
        if categories:
            num_labels = len(categories)
            max_label_length = max(len(str(cat)) for cat in categories) if categories else 0
            
            if num_labels > 8 or max_label_length > 10:
                # Rotate labels for better readability
                ax.tick_params(axis='x', rotation=45)
                for label in ax.get_xticklabels():
                    label.set_ha('right')
                
                # If too many categories, consider showing fewer
                if num_labels > 15:
                    # For time series data, show every nth label
                    step = max(1, num_labels // 12)  # Show ~12 labels max
                    indices = range(0, len(categories), step)
                    ax.set_xticks(indices)
                    ax.set_xticklabels([categories[i] for i in indices], rotation=45, ha='right')
            elif max_label_length > 6:
                # Just rotate if labels are moderately long
                ax.tick_params(axis='x', rotation=30)
                for label in ax.get_xticklabels():
                    label.set_ha('right')
    
    def _create_horizontal_bar_chart(self, ax, graph_data):
        """Create a horizontal bar chart."""
        categories = graph_data.get('categories', [])
        values = graph_data.get('values', [])
        
        bars = ax.barh(categories, values, color='lightcoral', alpha=0.8)
        ax.set_xlabel(graph_data.get('x_label', 'Values'))
        ax.set_ylabel(graph_data.get('y_label', 'Categories'))
        
        # Add value labels
        for bar, value in zip(bars, values):
            width = bar.get_width()
            ax.text(width + max(values) * 0.01, bar.get_y() + bar.get_height()/2,
                   f'{value:.1f}' if isinstance(value, float) else str(value),
                   ha='left', va='center')
    
    def _create_pie_chart(self, ax, graph_data):
        """Create a pie chart."""
        labels = graph_data.get('labels', [])
        values = graph_data.get('values', [])
        
        # Filter out zero values
        filtered_data = [(label, value) for label, value in zip(labels, values) if value > 0]
        if not filtered_data:
            ax.text(0.5, 0.5, 'No data to display', ha='center', va='center', transform=ax.transAxes)
            return
        
        labels, values = zip(*filtered_data)
        
        wedges, texts, autotexts = ax.pie(values, labels=labels, autopct='%1.1f%%', startangle=90)
        
        # Improve text readability
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
    
    def _create_scatter_plot(self, ax, graph_data):
        """Create a scatter plot."""
        x_values = graph_data.get('x_values', [])
        y_values = graph_data.get('y_values', [])
        
        ax.scatter(x_values, y_values, alpha=0.6, s=50, color='darkgreen')
        ax.set_xlabel(graph_data.get('x_label', 'X Axis'))
        ax.set_ylabel(graph_data.get('y_label', 'Y Axis'))
        ax.grid(True, alpha=0.3)
    
    def display_graph_info(self, graph_data: Dict, filepath: str = None) -> str:
        """Generate a text description of the graph for display."""
        graph_type = graph_data.get('graph_type', 'unknown')
        title = graph_data.get('title', 'Data Visualization')
        description = graph_data.get('description', '')
        
        output = [
            "📊 GRAPH GENERATED",
            "=" * 50,
            f"Graph Type: {graph_type.title()}",
            f"Title: {title}"
        ]
        
        if description:
            output.append(f"Description: {description}")
        
        if filepath:
            output.append(f"Saved to: {filepath}")
            # Auto-open the graph
            if self._auto_open_graph(filepath):
                output.append("🎨 Graph opened in default image viewer")
            else:
                output.append("💡 Please open the graph file manually to view")
        
        # Add data summary
        data_summary = graph_data.get('data_summary', {})
        if data_summary:
            output.append(f"Data Points: {data_summary.get('total_rows', 'N/A')}")
            output.append(f"Columns: {', '.join(data_summary.get('columns', []))}")
        
        # Add metadata about other available graph types
        metadata = graph_data.get('metadata', {})
        recommendations = metadata.get('all_recommendations', [])
        if len(recommendations) > 1:
            output.append("\n💡 Other visualization options available:")
            for i, rec in enumerate(recommendations[1:4], 1):  # Show up to 3 alternatives
                output.append(f"   {i}. {rec['type'].title()}: {rec['description']}")
        
        return "\n".join(output)
    
    def _auto_open_graph(self, filepath: str) -> bool:
        """Automatically open the graph file in the default image viewer."""
        try:
            import subprocess
            import os
            
            if os.name == 'nt':  # Windows
                os.startfile(filepath)
            elif os.name == 'posix':  # macOS/Linux
                if os.uname().sysname == 'Darwin':  # macOS
                    subprocess.run(['open', filepath], check=True)
                else:  # Linux
                    subprocess.run(['xdg-open', filepath], check=True)
            else:
                return False
            
            logger.info(f"📊 Graph opened: {filepath}")
            return True
            
        except Exception as e:
            logger.warning(f"⚠️ Could not auto-open graph: {e}")
            return False

# Enhanced Formatting Logic
class ResultFormatter:
    def __init__(self):
        self.graph_generator = GraphGenerator()
    
    def format_single_result(self, result: QueryResult) -> str:
        if not result.success:
            return f"❌ ERROR: {result.error}"
        
        if result.message and not result.data:
            return f"ℹ️ {result.message}"
        
        if result.data is None:
            return "✅ Query succeeded, but the result set was empty."
        
        output = []
        is_dynamic = result.tool_used == 'execute_dynamic_sql' or result.tool_used == 'execute_dynamic_sql_with_graph'
        
        # Handle graph data first if available
        if result.graph_data and result.graph_generated:
            graph_info = self.graph_generator.display_graph_info(
                result.graph_data,
                getattr(result, 'graph_filepath', None)
            )
            output.append(graph_info)
            output.append("")  # Add spacing
        
        # ALWAYS show the data table for dynamic queries
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
            
            # Check for zero values in comparison data
            if len(result.data) == 1 and len(headers) >= 2:
                all_zeros = all(result.data[0].get(col, 0) == 0 for col in headers if isinstance(result.data[0].get(col), (int, float)))
                if all_zeros:
                    output.append("\n⚠️ **Data Issue:** All values are zero - check date ranges or data availability")
                    output.append("💡 Suggestion: Try different years (2023, 2024, 2025) or verify data exists for these time periods")
            
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
        
        # Add any messages at the end
        if result.message:
            output.append(f"\n📝 {result.message}")
        
        return "\n".join(output)

    def format_multi_result(self, results: List[QueryResult], original_query: str) -> str:
        output = [f"🎯 RESULTS FOR COMPARISON: '{original_query}'", "="*70]
        for i, res in enumerate(results, 1):
            output.append(f"\n--- Result {i}/{len(results)} ---")
            output.append(self.format_single_result(res))
        return "\n".join(output)

# Enhanced AI Logic with Graph Detection
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
        
        # Define available tools for the AI (enhanced with graph generation)
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
                    ),
                    genai.protos.FunctionDeclaration(
                        name="execute_dynamic_sql_with_graph",
                        description="Execute a custom SQL query AND generate a graph visualization. Use this when the user asks for charts, graphs, plots, trends, comparisons, or visual analysis.",
                        parameters=genai.protos.Schema(
                            type=genai.protos.Type.OBJECT,
                            properties={
                                "sql_query": genai.protos.Schema(
                                    type=genai.protos.Type.STRING, 
                                    description="The SELECT SQL query to execute"
                                ),
                                "graph_type": genai.protos.Schema(
                                    type=genai.protos.Type.STRING,
                                    description="Optional: Preferred graph type. Valid options: 'line', 'bar', 'horizontal_bar', 'pie', 'scatter'. Leave blank or omit for auto-detection."
                                )
                            },
                            required=["sql_query"]
                        )
                    )
                ]
            )
        ]

    async def parse_query(self, user_query: str, history: List[str], client=None) -> List[Dict]:
        """Parse user query and determine which tool(s) to use with REAL improvement suggestions and graph detection"""
        
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
            "same time", "same period", "same timeframe", "try again"
        ]
        
        # Check for graph/visualization keywords
        graph_keywords = [
            "chart", "graph", "plot", "visualize", "show trend", "compare", "visual",
            "distribution", "over time", "by month", "by year", "trend", "trends",
            "performance chart", "analysis chart", "breakdown chart", "ranking chart",
            "pie chart", "bar chart", "line chart", "scatter plot", "plot data"
        ]
        
        # Check if this is a contextual follow-up
        is_contextual = any(indicator in user_query.lower() for indicator in follow_up_indicators)
        
        # Check if user wants visualization
        wants_graph = any(keyword in user_query.lower() for keyword in graph_keywords)
        
        # Enhanced context extraction from history
        context_data = self._extract_context_from_history(history)
        
        # FIXED: Actually fetch real improvement suggestions from the API
        improvement_context = ""
        improvement_found = False
        if client:
            try:
                logger.info(f"🔍 Fetching improvement suggestions for: {user_query}")
                suggestions_result = await client.call_tool('get_improvement_suggestions', {
                    'original_question': user_query
                })
                
                if (suggestions_result.success and 
                    suggestions_result.data and 
                    suggestions_result.data.get('improvements')):
                    
                    improvements = suggestions_result.data['improvements']
                    improvement_lines = [
                        "🚨🚨🚨 CRITICAL: PAST USER COMPLAINTS ABOUT SIMILAR QUERIES 🚨🚨🚨",
                        "=" * 80,
                        "THE USER HAS ALREADY COMPLAINED ABOUT THESE EXACT ISSUES:",
                        ""
                    ]
                    
                    for i, improvement in enumerate(improvements[:3], 1):
                        improvement_lines.append(f"{i}. PREVIOUS FAILURE:")
                        improvement_lines.append(f"   Question: {improvement['similar_question']}")
                        improvement_lines.append(f"   Failed SQL: {improvement['what_failed']}")
                        improvement_lines.append(f"   USER COMPLAINT: \"{improvement['user_suggestion']}\"")
                        improvement_lines.append(f"   🚨 YOU MUST FIX THIS: {improvement['user_suggestion']}")
                        improvement_lines.append("")
                    
                    improvement_lines.extend([
                        "🚨 MANDATORY REQUIREMENTS:",
                        "- If user said 'you didnt give me the may subscriptions' - CREATE SEPARATE COLUMNS FOR MAY DATA",
                        "- If user said 'there is no column for may' - LABEL COLUMNS CLEARLY WITH MONTH NAMES", 
                        "- If user wants comparison data - USE SEPARATE COLUMNS OR CLEAR ROW LABELS",
                        "- If user said 'when i ask for something and also to visualise, show me both the data and the graph' - ALWAYS SHOW DATA TABLE EVEN IF GRAPH FAILS",
                        "- DO NOT USE UNION - USE SEPARATE SUBQUERIES OR CASE STATEMENTS",
                        "",
                        "🚨 CRITICAL GRAPH REQUIREMENTS:",
                        "- For single-row comparison data (April vs May) - this IS suitable for bar charts",
                        "- Transform column-based comparisons into category-value pairs for graphing",
                        "- Example: april_subscriptions=45, may_subscriptions=52 becomes Categories: [April, May], Values: [45, 52]",
                        "- ALWAYS show data table AND attempt graph generation for visualization requests",
                        "",
                        "🚨 EXAMPLE FOR MONTH COMPARISON:",
                        "SELECT ",
                        "  (SELECT COUNT(*) FROM table WHERE MONTH(date) = 4) AS april_count,",
                        "  (SELECT COUNT(*) FROM table WHERE MONTH(date) = 5) AS may_count",
                        "",
                        "🚨 FAILURE TO INCORPORATE THESE SUGGESTIONS WILL RESULT IN REPEATED USER COMPLAINTS!"
                    ])
                    
                    improvement_context = "\n".join(improvement_lines)
                    improvement_found = True
                    logger.info(f"✅ Found {len(improvements)} improvement suggestions to apply")
                else:
                    improvement_context = self._get_pattern_based_improvements(user_query)
                    
            except Exception as e:
                logger.warning(f"⚠️ Failed to fetch improvement suggestions: {e}")
                improvement_context = self._get_pattern_based_improvements(user_query)
        else:
            improvement_context = self._get_pattern_based_improvements(user_query)
        
        prompt = f"""
You are a subscription analytics assistant with graph generation capabilities. Analyze the user's query and choose the most appropriate tool.

CONVERSATION HISTORY:
{history_context}

EXTRACTED CONTEXT DATA:
{context_data}

{improvement_context}

CURRENT USER QUERY: "{user_query}"

🚨 CRITICAL TOOL SELECTION RULES - FOLLOW THESE EXACTLY:

1. **GRAPH/VISUALIZATION REQUESTS** - Use execute_dynamic_sql_with_graph ONLY when user explicitly requests visualization:
   - User asks for "chart", "graph", "plot", "visualize", "show trend", "create chart"
   - User says "compare X vs Y and visualize" or "compare X vs Y with a chart" 
   - User explicitly mentions wanting to see graphics/visual/charts
   - User says "show me a graph of" or "plot this data"

2. **REGULAR COMPARISON QUERIES** - Use execute_dynamic_sql (NO graph):
   - "compare April vs May subscriptions" (no visualization words)
   - "show me April vs May data" (no visualization words)
   - "what's the difference between" (no visualization words)
   - Any analysis request WITHOUT explicit visualization words

3. **IMPORTANT**: Only generate graphs when user specifically asks for visualization
   - Don't assume user wants graphs for comparisons
   - Don't add visualization unless explicitly requested

3. **SIMPLE SUBSCRIPTION COUNT QUERIES** - Use pre-built tools:
   - "subscriptions in last X days" → get_subscriptions_in_last_days
   - "how many subscriptions" + time period → get_subscriptions_in_last_days
   - "new subscriptions in last X days" → get_subscriptions_in_last_days

4. **SIMPLE PAYMENT SUCCESS RATE QUERIES** - Use pre-built tools:
   - "payment success rate" + time period → get_payment_success_rate_in_last_days
   - "payment performance" + time period → get_payment_success_rate_in_last_days

5. **SPECIFIC USER QUERIES** - Use pre-built tools:
   - "payment history for user X" → get_user_payment_history
   - mentions specific merchant_user_id → get_user_payment_history

6. **HEALTH/STATUS QUERIES** - Use pre-built tools:
   - "database status" → get_database_status
   - "connection" queries → get_database_status

7. **COMPLEX ANALYTICS WITHOUT GRAPHS** - Use execute_dynamic_sql:
   - Complex queries where user doesn't want visualization
   - Simple data extraction requests without "visualize" keywords
   - Follow-up questions with "try again" but no visualization request

🔥 CRITICAL SQL GENERATION RULES:

1. **ALWAYS START WITH SELECT**: Every SQL query must begin with "SELECT"

2. **FOR MONTH COMPARISONS** - Use this pattern:
   ```sql
   SELECT 
       (SELECT COUNT(DISTINCT sc.subscription_id) 
        FROM subscription_contract_v2 AS sc 
        WHERE MONTH(sc.subcription_start_date) = 4 AND YEAR(sc.subcription_start_date) = 2025) AS april_subscriptions,
       (SELECT COUNT(DISTINCT sc.subscription_id) 
        FROM subscription_contract_v2 AS sc 
        WHERE MONTH(sc.subcription_start_date) = 5 AND YEAR(sc.subcription_start_date) = 2025) AS may_subscriptions
   ```

3. **NEVER USE UNION FOR COMPARISONS** - Use separate columns or subqueries

4. **FOR GRAPH-WORTHY QUERIES** - Structure data to be graph-friendly:
   - For time series: Include date/time column and numeric values
   - For comparisons: Include category column and numeric values  
   - For rankings: ORDER BY the metric being ranked
   - Limit to reasonable number of rows (2-100) for good visualization

5. **🚨 INCORPORATE IMPROVEMENT SUGGESTIONS**: If improvement suggestions are provided above:
   - If user complained about missing May data - CREATE SEPARATE MAY COLUMNS
   - If user said "no column for may" - ENSURE MAY DATA HAS SEPARATE COLUMNS
   - If user wants comparison - SEPARATE COLUMNS FOR EACH TIME PERIOD
   - If user said "show payment rates" - ADD payment success rate calculations

6. **FOR COMPREHENSIVE ANALYTICS**: When user asks for "subscription analytics", include:
   ```sql
   SELECT 
       COUNT(DISTINCT sc.subscription_id) as total_subscriptions,
       COUNT(DISTINCT sc.merchant_user_id) as unique_users,
       COUNT(pd.subcription_payment_details_id) as total_payments,
       SUM(CASE WHEN pd.status = 'ACTIVE' THEN 1 ELSE 0 END) as successful_payments,
       ROUND((SUM(CASE WHEN pd.status = 'ACTIVE' THEN 1 ELSE 0 END) * 100.0 / COUNT(pd.subcription_payment_details_id)), 2) as payment_success_rate,
       SUM(CASE WHEN pd.status = 'ACTIVE' THEN pd.trans_amount_decimal ELSE 0 END) as total_revenue
   FROM subscription_contract_v2 AS sc 
   LEFT JOIN subscription_payment_details AS pd ON sc.subscription_id = pd.subscription_id 
   WHERE sc.subcription_start_date >= DATE_SUB(CURDATE(), INTERVAL X DAY)
   ```

{self.db_schema}

🚨 EXAMPLES OF CORRECT TOOL SELECTION:

✅ "How many new subscriptions in the last 7 days" → get_subscriptions_in_last_days (days: 7)
✅ "Payment success rate last month" → get_payment_success_rate_in_last_days (days: 30)  
✅ "Show payment history for user abc123" → get_user_payment_history (merchant_user_id: "abc123")
✅ "Database status" → get_database_status
🎯 "Chart comparing April vs May subscriptions" → execute_dynamic_sql_with_graph
🎯 "Show me a trend of payments over time" → execute_dynamic_sql_with_graph  
🎯 "Visualize top 10 users by success rate" → execute_dynamic_sql_with_graph
❌ "Compare April vs May subscriptions" (no chart mentioned) → execute_dynamic_sql
❌ "Top 10 users by success rate" (no chart mentioned) → execute_dynamic_sql

🔥 GRAPH TYPE GUIDANCE:
- Time series data (dates + values) → suggest "line" 
- Category comparisons → suggest "bar" or "horizontal_bar"
- Parts of a whole → suggest "pie"
- Two numeric relationships → suggest "scatter"
- Leave graph_type empty for auto-detection

🔥 IF IMPROVEMENT SUGGESTIONS ARE PROVIDED ABOVE, YOU MUST INCORPORATE THEM INTO YOUR SQL!

Choose the most appropriate tool and provide necessary parameters. For visualization requests, use execute_dynamic_sql_with_graph. For simple queries, use pre-built tools. For complex queries without visualization, use execute_dynamic_sql.
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
                        if fc.name in ['execute_dynamic_sql', 'execute_dynamic_sql_with_graph'] and 'sql_query' in params:
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
                            'original_query': user_query,
                            'wants_graph': fc.name == 'execute_dynamic_sql_with_graph'
                        })
            
            if not tool_calls:
                logger.warning("AI did not return a valid function call, defaulting to database status")
                return [{
                    'tool': 'get_database_status',
                    'parameters': {},
                    'original_query': user_query,
                    'wants_graph': False
                }]
            
            logger.info(f"AI selected tool(s): {[tc['tool'] for tc in tool_calls]}")
            if any(tc.get('wants_graph', False) for tc in tool_calls):
                logger.info("🎯 Graph generation requested")
            if improvement_found:
                logger.info(f"🎯 Applied improvement suggestions to tool selection and SQL generation")
            return tool_calls
            
        except Exception as e:
            logger.error(f"Error in parse_query: {e}", exc_info=True)
            # Fallback to database status
            return [{
                'tool': 'get_database_status',
                'parameters': {},
                'original_query': user_query,
                'wants_graph': False
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

    def _get_pattern_based_improvements(self, user_query: str) -> str:
        """Get pattern-based improvement suggestions as fallback."""
        improvement_hints = []
        query_lower = user_query.lower()
        
        # Pattern-based improvement suggestions
        if 'analytics' in query_lower or 'subscription' in query_lower:
            improvement_hints.append("⚠️ For subscription analytics: Users expect comprehensive data including success rates, payment rates, revenue totals, and user counts - not just simple counts")
            improvement_hints.append("⚠️ For complete analytics: JOIN subscription_contract_v2 with subscription_payment_details to get payment success rates")
        
        if 'success rate' in query_lower and ('user' in query_lower or 'merchant' in query_lower):
            improvement_hints.append("For user success rates: Always use GROUP BY merchant_user_id and include minimum payment thresholds (HAVING COUNT(*) >= 3)")
        
        if 'compare' in query_lower or 'comparison' in query_lower or ('april' in query_lower and 'may' in query_lower):
            improvement_hints.append("🚨 For comparisons: Use separate columns for each time period, NOT UNION queries")
            improvement_hints.append("🚨 For month comparisons: Use subqueries with clear column names like april_subscriptions, may_subscriptions")
        
        if 'top' in query_lower or 'best' in query_lower or 'worst' in query_lower:
            improvement_hints.append("For ranking queries: Use ORDER BY with LIMIT, and consider tie-breaking with secondary sort criteria")
        
        if 'failed' in query_lower or 'failure' in query_lower:
            improvement_hints.append("For failure analysis: Use pd.status != 'ACTIVE' or pd.status IN ('FAILED', 'FAIL') for failed payments")
        
        if 'revenue' in query_lower or 'amount' in query_lower:
            improvement_hints.append("For revenue calculations: Only sum amounts where pd.status = 'ACTIVE' for accurate totals")
        
        if 'last' in query_lower and ('day' in query_lower or 'week' in query_lower or 'month' in query_lower):
            improvement_hints.append("For date filtering: Use proper MySQL date functions like DATE_SUB(CURDATE(), INTERVAL X DAY)")
        
        # Graph-specific improvements
        if any(keyword in query_lower for keyword in ['chart', 'graph', 'plot', 'visualize', 'trend']):
            improvement_hints.append("🎯 For graph queries: Structure data with clear x-axis (categories/dates) and y-axis (numeric values)")
            improvement_hints.append("🎯 For time series: Include date columns and order by date")
            improvement_hints.append("🎯 For comparisons: Use meaningful category names and limit to 2-50 categories for readability")
            improvement_hints.append("🚨 CRITICAL: When user asks for visualization, ALWAYS show data table even if graph generation fails")
            improvement_hints.append("🚨 Single-row comparison data (April vs May) IS suitable for bar charts - don't reject it")
        
        if 'visualize' in query_lower and 'data' in query_lower:
            improvement_hints.append("🚨 USER FEEDBACK: 'when i ask for something and also to visualise, show me both the data and the graph'")
            improvement_hints.append("🚨 ALWAYS show data table AND attempt graph generation for visualization requests")
            improvement_hints.append("🚨 Single-row comparison data (April vs May) IS suitable for bar charts - don't reject it")
        
        if improvement_hints:
            return "PATTERN-BASED IMPROVEMENT SUGGESTIONS:\n" + "\n".join(f"- {hint}" for hint in improvement_hints)
        else:
            return "No specific improvement patterns identified for this query type."

# Core Client Class (enhanced with graph generation)
class UniversalClient:
    def __init__(self, config: dict):
        self.config = config
        self.nlp = GeminiNLPProcessor()
        self.session = None
        self.formatter = ResultFormatter()
        self.history = []
        self.graph_generator = GraphGenerator()

    async def __aenter__(self):
        # Enhanced SSL configuration with better certificate handling
        ssl_context = None
        
        # Try multiple SSL configurations with improved fallbacks
        ssl_configs = [
            {
                'name': 'Certifi with system certs',
                'setup': lambda: self._create_ssl_context_with_certifi()
            },
            {
                'name': 'System default SSL',
                'setup': lambda: ssl.create_default_context()
            },
            {
                'name': 'Permissive SSL (hostname verification disabled)',
                'setup': lambda: self._create_permissive_ssl_context()
            },
            {
                'name': 'Insecure SSL (verification disabled)',
                'setup': lambda: False  # Disable SSL verification entirely
            }
        ]
        
        for attempt, config in enumerate(ssl_configs):
            try:
                if config['name'] == 'Insecure SSL (verification disabled)':
                    ssl_context = False
                    logger.warning(f"⚠️ Using insecure SSL configuration (no verification)")
                    break
                else:
                    ssl_context = config['setup']()
                
                # Test the SSL context with a real connection (but shorter timeout)
                test_connector = aiohttp.TCPConnector(ssl=ssl_context, limit=1)
                test_session = aiohttp.ClientSession(connector=test_connector, timeout=aiohttp.ClientTimeout(total=5))
                
                try:
                    server_url = self.config.get('SUBSCRIPTION_API_URL', 'https://subscription-analytics.onrender.com')
                    test_url = f"{server_url}/health"
                    async with test_session.get(test_url) as test_response:
                        if test_response.status in [200, 404, 401]:  # Accept these as "connection working"
                            logger.info(f"✅ SSL configuration '{config['name']}' tested successfully (status: {test_response.status})")
                            await test_session.close()
                            break
                        else:
                            logger.warning(f"⚠️ SSL configuration '{config['name']}' got unexpected status: {test_response.status}")
                            await test_session.close()
                            continue
                except Exception as test_e:
                    await test_session.close()
                    logger.warning(f"⚠️ SSL configuration '{config['name']}' failed connection test: {test_e}")
                    continue
                
            except Exception as e:
                logger.warning(f"⚠️ SSL configuration '{config['name']}' failed setup: {e}")
                continue
        
        connector = aiohttp.TCPConnector(
            ssl=ssl_context,
            limit=10,           # Total connection limit
            limit_per_host=5,   # Per-host connection limit
            enable_cleanup_closed=True,  # Clean up closed connections
            force_close=True,   # Force close connections after each request
            ttl_dns_cache=300   # DNS cache TTL
            # Note: keepalive_timeout cannot be set when force_close=True
        )
        self.session = aiohttp.ClientSession(
            connector=connector,
            timeout=aiohttp.ClientTimeout(total=120),  # Increased timeout
            headers={'Connection': 'close'}  # Force connection close
        )
        return self
    
    def _create_ssl_context_with_certifi(self):
        """Create SSL context with certifi certificate bundle."""
        try:
            context = ssl.create_default_context(cafile=certifi.where())
            context.check_hostname = True
            context.verify_mode = ssl.CERT_REQUIRED
            return context
        except Exception as e:
            logger.warning(f"Could not create SSL context with certifi: {e}")
            return ssl.create_default_context()
    
    def _create_permissive_ssl_context(self):
        """Create permissive SSL context that doesn't verify hostnames."""
        try:
            context = ssl.create_default_context(cafile=certifi.where())
            context.check_hostname = False
            context.verify_mode = ssl.CERT_REQUIRED
            return context
        except Exception as e:
            logger.warning(f"Could not create permissive SSL context: {e}")
            # Fallback to even more permissive
            context = ssl.create_default_context()
            context.check_hostname = False
            context.verify_mode = ssl.CERT_NONE
            return context

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

    async def call_tool(self, tool_name: str, parameters: Dict = None, original_query: str = "", wants_graph: bool = False) -> QueryResult:
        """Call a tool on the API server with enhanced error handling and graph generation support"""
        
        # Handle the special case of execute_dynamic_sql_with_graph
        if tool_name == 'execute_dynamic_sql_with_graph':
            return await self._handle_sql_with_graph(parameters, original_query)
        
        headers = {
            "Authorization": f"Bearer {self.config['API_KEY_1']}",
            "Connection": "close"  # Force connection close
        }
        payload = {"tool_name": tool_name, "parameters": parameters or {}}
        server_url = self.config['SUBSCRIPTION_API_URL']
        
        # Log the request for debugging
        if tool_name == 'execute_dynamic_sql':
            logger.info(f"🔍 Executing SQL: {parameters.get('sql_query', 'N/A')}")
        elif tool_name in ['get_subscriptions_in_last_days', 'get_payment_success_rate_in_last_days']:
            logger.info(f"✅ Using pre-built tool: {tool_name} with params: {parameters}")
        
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
    
    async def _handle_sql_with_graph(self, parameters: Dict, original_query: str) -> QueryResult:
        """Handle the special execute_dynamic_sql_with_graph workflow."""
        try:
            # First, execute the SQL query
            sql_result = await self.call_tool('execute_dynamic_sql', {
                'sql_query': parameters['sql_query']
            }, original_query)
            
            if not sql_result.success:
                # If SQL failed, return the error
                return sql_result
            
            if not sql_result.data:
                # If no data, return the result with message
                sql_result.message = (sql_result.message or "") + "\n💡 No data returned - cannot generate graph"
                return sql_result
            
            # Always show the SQL data first, regardless of graph generation success
            logger.info(f"📊 SQL returned {len(sql_result.data)} rows for graph analysis")
            
            # Check if we can generate graphs
            if not self.graph_generator.can_generate_graphs():
                # Return SQL result with note about graph unavailability
                sql_result.message = (sql_result.message or "") + "\n⚠️ Graph generation unavailable (matplotlib not installed)"
                return sql_result
            
            # Try to generate graph data using the new API tool
            graph_config = {
                'data': sql_result.data
            }
            
            # Add graph type if specified
            if parameters.get('graph_type'):
                graph_config['graph_type'] = parameters['graph_type']
            
            logger.info(f"📊 Attempting graph generation for data: {sql_result.data}")
            graph_result = await self.call_tool('generate_graph_data', graph_config)
            
            if not graph_result.success:
                # Graph generation failed, but we still have SQL data
                logger.warning(f"⚠️ Graph generation failed: {graph_result.error}")
                sql_result.message = (sql_result.message or "") + f"\n⚠️ Graph generation failed: {graph_result.error}\n💡 Showing data table instead"
                return sql_result
            
            # Generate the actual graph file
            graph_filepath = None
            if graph_result.data:
                graph_filepath = self.graph_generator.generate_graph(graph_result.data, original_query)
            
            # Enhance the SQL result with graph information
            sql_result.graph_data = graph_result.data
            sql_result.graph_generated = graph_filepath is not None
            
            if graph_filepath:
                sql_result.graph_filepath = graph_filepath
                sql_result.message = (sql_result.message or "") + f"\n📊 Graph generated and saved"
                logger.info(f"✅ Graph successfully generated for query: {original_query}")
            else:
                sql_result.message = (sql_result.message or "") + "\n⚠️ Graph data generated but file creation failed\n💡 Showing data table"
            
            return sql_result
            
        except Exception as e:
            logger.error(f"❌ Error in _handle_sql_with_graph: {e}")
            # Fall back to regular SQL execution to ensure user gets data
            fallback_result = await self.call_tool('execute_dynamic_sql', {
                'sql_query': parameters['sql_query']
            }, original_query)
            
            if fallback_result.success:
                fallback_result.message = (fallback_result.message or "") + f"\n⚠️ Graph generation failed: {str(e)}\n💡 Showing data table instead"
            
            return fallback_result

    async def query(self, nl_query: str) -> Union[QueryResult, List[QueryResult]]:
        """Process a natural language query with graph generation support"""
        try:
            # FIXED: Pass self (client) to parse_query so it can fetch improvement suggestions
            parsed_calls = await self.nlp.parse_query(nl_query, self.history, client=self)
            
            if len(parsed_calls) > 1:
                # Multiple tool calls
                results = []
                for call in parsed_calls:
                    result = await self.call_tool(
                        call['tool'], 
                        call['parameters'], 
                        call['original_query'],
                        call.get('wants_graph', False)
                    )
                    results.append(result)
                return results
            else:
                # Single tool call
                call = parsed_calls[0]
                return await self.call_tool(
                    call['tool'], 
                    call['parameters'], 
                    call['original_query'],
                    call.get('wants_graph', False)
                )
                
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
    
    async def submit_feedback(self, result: QueryResult, helpful: bool, improvement_suggestion: str = None):
        """Submit feedback for a dynamic query with enhanced negative feedback handling."""
        if result.is_dynamic and result.generated_sql and result.original_query:
            try:
                # Prepare feedback parameters
                feedback_params = {
                    'original_question': result.original_query,
                    'sql_query': result.generated_sql,
                    'was_helpful': helpful
                }
                
                # Add improvement suggestion if provided
                if not helpful and improvement_suggestion:
                    feedback_params['improvement_suggestion'] = improvement_suggestion
                
                feedback_result = await self.call_tool('record_query_feedback', feedback_params)
                
                if feedback_result.success and feedback_result.message:
                    print(f"✅ {feedback_result.message}")
                    
                    # If negative feedback with improvement, show additional confirmation
                    if not helpful and improvement_suggestion and feedback_result.data and feedback_result.data.get('improvement_recorded'):
                        print("🎯 Your improvement suggestion has been saved and will influence future similar queries.")
                        print("💡 The AI will try to incorporate your suggestions in future SQL generation.")
                    
                    # If negative feedback, offer to show similar queries to help understand the issue
                    if not helpful:
                        print("🔍 The system will now remember this as an example to avoid.")
                        
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

# Standalone Interactive Mode (enhanced with graph support)
async def interactive_mode():
    """Run the interactive CLI mode with graph generation support"""
    print("✨ Subscription Analytics AI Agent with Graph Generation ✨")
    print("=" * 70)
    
    # Get configuration
    config_manager = ConfigManager()
    
    try:
        user_config = config_manager.get_config()
        
        # Configure Gemini AI
        genai.configure(api_key=user_config['GOOGLE_API_KEY'])
        
        print(f"🔗 Connected to server: {user_config['SUBSCRIPTION_API_URL']}")
        print("🧠 Enhanced with real-time improvement suggestion learning!")
        print("🔧 Fixed tool selection + SQL generation learning!")
        
        # Check graph generation capability
        if MATPLOTLIB_AVAILABLE:
            print("📊 Graph generation enabled (matplotlib available)")
            # Test graph directory creation
            test_generator = GraphGenerator()
            if test_generator.graphs_dir and test_generator.graphs_dir.exists():
                print(f"📁 Graph directory ready: {test_generator.graphs_dir}")
            else:
                print("⚠️ Warning: Graph directory creation may have issues")
        else:
            print("⚠️ Graph generation disabled (matplotlib not installed)")
            print("   Install with: pip install matplotlib")
        
        async with UniversalClient(config=user_config) as client:
            print("\n💬 Enter questions in natural language. Type 'quit' or 'exit' to leave.")
            print("\n📚 Example queries:")
            print("  • How many new subscriptions in the last 7 days? (will use pre-built tool)")
            print("  • What's the payment success rate for the last month? (will use pre-built tool)")
            print("  • Show me payment history for user abc123 (will use pre-built tool)")
            print("  • Compare April vs May subscriptions (will use dynamic SQL)")
            print("  • Give me subscription analytics for 10 days (will use dynamic SQL)")
            print("  • Chart showing payment trends over time (will generate graph)")
            print("  • Visualize top 10 users by success rate (will generate graph)")
            
            # Check if auto-open is desired
            auto_open_graphs = True  # Default to True
            try:
                auto_open_input = input("\n🎨 Auto-open graphs when generated? (y/n, default: y): ").lower().strip()
                if auto_open_input in ['n', 'no']:
                    auto_open_graphs = False
                    print("📊 Graphs will be saved but not automatically opened")
                else:
                    print("📊 Graphs will be automatically opened when generated")
            except (KeyboardInterrupt, EOFError):
                pass  # Use default
            
            print("\n" + "="*50)
            
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
                    
                    # Show graph file location if generated
                    if isinstance(result, QueryResult) and hasattr(result, 'graph_filepath'):
                        print(f"\n🎨 Graph saved to: {result.graph_filepath}")
                        
                        if auto_open_graphs:
                            # Try to auto-open the graph
                            try:
                                import subprocess
                                import os
                                
                                opened = False
                                if os.name == 'nt':  # Windows
                                    os.startfile(result.graph_filepath)
                                    opened = True
                                    print("📊 Graph opened in default image viewer")
                                elif os.name == 'posix':  # macOS/Linux
                                    if os.uname().sysname == 'Darwin':  # macOS
                                        subprocess.run(['open', result.graph_filepath], check=True, timeout=5)
                                        opened = True
                                        print("📊 Graph opened in default image viewer")
                                    else:  # Linux
                                        try:
                                            subprocess.run(['xdg-open', result.graph_filepath], check=True, timeout=5)
                                            opened = True
                                            print("📊 Graph opened in default image viewer")
                                        except (subprocess.TimeoutExpired, subprocess.CalledProcessError):
                                            # Try alternative methods for Linux
                                            for cmd in [['display'], ['eog'], ['feh'], ['gpicview']]:
                                                try:
                                                    subprocess.run(cmd + [result.graph_filepath], check=True, timeout=5)
                                                    opened = True
                                                    print(f"📊 Graph opened with {cmd[0]}")
                                                    break
                                                except:
                                                    continue
                                
                                if not opened:
                                    print("💡 Could not auto-open. Open manually:")
                                    if os.name == 'nt':
                                        print(f"   Command: start {result.graph_filepath}")
                                    else:
                                        print(f"   Command: open {result.graph_filepath}")
                                        
                            except Exception as e:
                                print(f"⚠️ Could not auto-open graph: {e}")
                                print("💡 Please open the graph file manually to view the visualization")
                                if os.name == 'nt':
                                    print(f"   Windows: start {result.graph_filepath}")
                                elif os.name == 'posix':
                                    print(f"   macOS: open {result.graph_filepath}")
                                    print(f"   Linux: xdg-open {result.graph_filepath}")
                        else:
                            print("💡 To view the graph, open:")
                            if os.name == 'nt':
                                print(f"   Windows: start {result.graph_filepath}")
                            elif os.name == 'posix':
                                print(f"   macOS: open {result.graph_filepath}")
                                print(f"   Linux: xdg-open {result.graph_filepath}")
                        
                        # Also try to show a preview in terminal if supported
                        try:
                            # Check if we're in a terminal that supports images (iTerm2, etc.)
                            if os.environ.get('TERM_PROGRAM') == 'iTerm.app':
                                print("\n🖼️ Attempting to show graph preview in terminal...")
                                try:
                                    # iTerm2 inline image protocol
                                    import base64
                                    with open(result.graph_filepath, 'rb') as f:
                                        image_data = base64.b64encode(f.read()).decode()
                                    
                                    # iTerm2 escape sequence for inline images
                                    print(f"\033]1337;File=inline=1:{image_data}\a")
                                    print("📊 Graph preview shown above (if supported by your terminal)")
                                except Exception:
                                    pass  # Silently fail if not supported
                                    
                        except Exception:
                            pass  # Silently fail if terminal detection fails
                    
                    # Update conversation history
                    client.manage_history(query, output)
                    
                    # Handle feedback for dynamic queries
                    if (isinstance(result, QueryResult) and 
                        result.is_dynamic and 
                        result.success and 
                        result.data is not None):
                        
                        print("\n" + "="*50)
                        print("📝 This answer was generated using a custom SQL query.")
                        
                        # Special feedback for graph generation
                        if result.graph_generated:
                            print("📊 A graph was also generated for this data.")
                        
                        while True:
                            feedback_input = input("Was this answer helpful and accurate? (y/n/skip): ").lower().strip()
                            if feedback_input in ['y', 'yes']:
                                await client.submit_feedback(result, True)
                                break
                            elif feedback_input in ['n', 'no']:
                                print("\n💬 Help us improve! What was wrong with this answer?")
                                print("Examples:")
                                print("  • 'The SQL should have filtered by date range'")
                                print("  • 'Missing JOIN with another table'") 
                                print("  • 'Wrong aggregation function used'")
                                print("  • 'Results should be sorted differently'")
                                print("  • 'Query timeout - needs optimization'")
                                print("  • 'Should also show payment rates and success rates'")
                                print("  • 'Missing column for May data'")
                                print("  • 'Should have separate columns for each month'")
                                if result.graph_generated:
                                    print("  • 'Wrong graph type - should be line/bar/pie chart'")
                                    print("  • 'Graph axes are confusing or incorrectly labeled'")
                                
                                while True:
                                    improvement = input("\nHow can this be improved? (or 'skip' to not provide): ").strip()
                                    if improvement.lower() in ['skip', 's', '']:
                                        await client.submit_feedback(result, False)
                                        break
                                    elif len(improvement) < 10:
                                        print("⚠️ Please provide a more detailed suggestion (at least 10 characters) or type 'skip'.")
                                        continue
                                    elif len(improvement) > 500:
                                        print("⚠️ Please keep suggestions under 500 characters.")
                                        continue
                                    else:
                                        await client.submit_feedback(result, False, improvement)
                                        print("🙏 Thank you for the detailed feedback! This helps improve the system.")
                                        break
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
        print("4. For graph features, install matplotlib: pip install matplotlib")
    
    print("\n👋 Goodbye!")

if __name__ == "__main__":
    try:
        asyncio.run(interactive_mode())
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
    except Exception as e:
        print(f"❌ Failed to start client: {e}")
        print("\n💡 The client only needs Google API key and server API key, not database credentials!")
        if not MATPLOTLIB_AVAILABLE:
            print("📊 For graph features, install matplotlib: pip install matplotlib")