#!/usr/bin/env python3
"""
Debug script to test chart detection logic
"""

import sys
import os
from pathlib import Path

# Add the client directory to the path
sys.path.insert(0, str(Path(__file__).parent / 'client'))

from universal_client import CompleteSmartNLPProcessor

def test_chart_detection():
    """Test the chart detection logic"""
    processor = CompleteSmartNLPProcessor()
    
    # Test cases
    test_cases = [
        "number of merchants with more than 5 subscriptions and number of merchants with more than 5 payments (generate a bar graph showing these values)",
        "number of merchants with more than 5 subscriptions and number of merchants with more than 5 payments (show a graph as well)",
        "number of merchants with more than 5 subscriptions and number of merchants with more than 5 payments (gnereate a grpah also)",
    ]
    
    print("🔧 Testing chart detection logic...")
    
    for i, test_query in enumerate(test_cases, 1):
        print(f"\nTest {i}: {test_query}")
        
        # Test chart analysis
        chart_analysis = processor._analyze_complete_chart_requirements(test_query, [])
        print(f"Chart analysis: {chart_analysis}")
        
        # Test if wants_visualization is True
        if chart_analysis.get('wants_visualization', False):
            print("✅ Visualization detected!")
        else:
            print("❌ No visualization detected")
            
        # Test if chart_type is set
        if chart_analysis.get('chart_type'):
            print(f"✅ Chart type detected: {chart_analysis['chart_type']}")
        else:
            print("❌ No chart type detected")

if __name__ == "__main__":
    test_chart_detection() 