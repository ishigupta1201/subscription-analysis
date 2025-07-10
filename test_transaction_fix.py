#!/usr/bin/env python3
"""
Test script to verify the transaction vs subscription detection improvements
"""

import asyncio
import sys
import os
from pathlib import Path

# Add the client directory to the path
sys.path.insert(0, str(Path(__file__).parent / 'client'))

from universal_client import CompleteSmartNLPProcessor

async def test_transaction_detection():
    """Test the improved transaction detection logic"""
    processor = CompleteSmartNLPProcessor()
    
    # Test cases
    test_cases = [
        {
            'query': 'number of transactions done on 9 june 2050',
            'expected_table': 'subscription_payment_details',
            'expected_field': 'created_date'
        },
        {
            'query': 'number of payments on 9 june 2050', 
            'expected_table': 'subscription_payment_details',
            'expected_field': 'created_date'
        },
        {
            'query': 'number of subscriptions on 9 june 2050',
            'expected_table': 'subscription_contract_v2', 
            'expected_field': 'subcription_start_date'
        }
    ]
    
    print("🔧 Testing transaction vs subscription detection...")
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\nTest {i}: {test_case['query']}")
        
        # Test the specific date query handler
        result = processor.handle_specific_date_queries(test_case['query'], [])
        
        if result:
            sql = result[0]['parameters']['sql_query']
            print(f"Generated SQL: {sql}")
            
            # Check if correct table is used
            if test_case['expected_table'] in sql:
                print(f"✅ Correct table detected: {test_case['expected_table']}")
            else:
                print(f"❌ Wrong table! Expected: {test_case['expected_table']}")
                
            # Check if correct field is used
            if test_case['expected_field'] in sql:
                print(f"✅ Correct field detected: {test_case['expected_field']}")
            else:
                print(f"❌ Wrong field! Expected: {test_case['expected_field']}")
        else:
            print("❌ No result generated")

async def test_feedback_processing():
    """Test the feedback processing for query type corrections"""
    processor = CompleteSmartNLPProcessor()
    
    # Test feedback extraction
    test_feedback = "i asked for transactions not subscriptions"
    
    print("\n🔧 Testing feedback processing...")
    print(f"Feedback: {test_feedback}")
    
    # Test the feedback extraction
    feedback_result = processor._extract_complete_recent_feedback([test_feedback])
    print(f"Extracted feedback: {feedback_result}")
    
    # Test chart detection with feedback
    query_with_feedback = f"number of subscriptions on 9 june 2050 ({test_feedback})"
    print(f"\nQuery with feedback: {query_with_feedback}")
    
    chart_analysis = processor._analyze_complete_chart_requirements(query_with_feedback, [])
    print(f"Chart analysis: {chart_analysis}")

if __name__ == "__main__":
    asyncio.run(test_transaction_detection())
    asyncio.run(test_feedback_processing()) 