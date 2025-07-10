#!/usr/bin/env python3
"""
Test script to verify the improvements to query splitting and LIMIT enforcement
"""

import asyncio
import sys
import os
from pathlib import Path

# Add the client directory to the path
sys.path.insert(0, str(Path(__file__).parent / 'client'))

from universal_client import CompleteSmartNLPProcessor, CompleteEnhancedUniversalClient
import json

async def test_query_splitting():
    """Test the improved query splitting logic"""
    processor = CompleteSmartNLPProcessor()
    
    # Test multi-part query with different chart types
    test_query = "Show me a pie chart of payment success rates; show me a bar chart of the top 5 merchants by total payment revenue"
    
    print("🔧 Testing query splitting...")
    print(f"Original query: {test_query}")
    
    # Test the parse_query method
    tool_calls = await processor.parse_query(test_query, [], None)
    
    print(f"✅ Generated {len(tool_calls)} tool calls:")
    for i, call in enumerate(tool_calls, 1):
        print(f"  {i}. Tool: {call['tool']}")
        if 'sql_query' in call['parameters']:
            sql = call['parameters']['sql_query']
            print(f"     SQL: {sql[:100]}...")
            # Check if LIMIT is present
            if 'LIMIT 5' in sql:
                print("     ✅ LIMIT 5 found in SQL")
            else:
                print("     ❌ LIMIT 5 NOT found in SQL")
        if 'graph_type' in call['parameters']:
            print(f"     Graph type: {call['parameters']['graph_type']}")
        print()

async def test_real_execution():
    """Test the real system execution with the improvements"""
    print("\n🔧 Testing real system execution...")
    
    # Load config
    config_path = Path(__file__).parent / 'client' / 'config.json'
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    async with CompleteEnhancedUniversalClient(config) as client:
        # Test the multi-part query
        test_query = "Show me a pie chart of payment success rates; show me a bar chart of the top 5 merchants by total payment revenue"
        
        print(f"Testing query: {test_query}")
        
        try:
            result = await client.query(test_query)
            
            if isinstance(result, list):
                print(f"✅ Generated {len(result)} results:")
                for i, res in enumerate(result, 1):
                    print(f"  Result {i}:")
                    if res.success:
                        print(f"    ✅ Success: {res.message}")
                        if res.generated_sql:
                            print(f"    SQL: {res.generated_sql}")
                            if 'LIMIT 5' in res.generated_sql:
                                print("    ✅ LIMIT 5 found in executed SQL")
                            else:
                                print("    ❌ LIMIT 5 NOT found in executed SQL")
                    else:
                        print(f"    ❌ Error: {res.error}")
            else:
                print(f"✅ Single result: {result.message}")
                if result.generated_sql:
                    print(f"SQL: {result.generated_sql}")
                    if 'LIMIT 5' in result.generated_sql:
                        print("✅ LIMIT 5 found in executed SQL")
                    else:
                        print("❌ LIMIT 5 NOT found in executed SQL")
                        
        except Exception as e:
            print(f"❌ Error during execution: {e}")

async def test_limit_enforcement():
    """Test the LIMIT enforcement logic"""
    processor = CompleteSmartNLPProcessor()
    
    test_cases = [
        {
            'query': 'show me a bar chart of the top 5 merchants by total payment revenue',
            'sql': 'SELECT c.merchant_user_id, COUNT(*) AS total_payments FROM subscription_payment_details p JOIN subscription_contract_v2 c ON p.subscription_id = c.subscription_id WHERE p.status = "ACTIVE" GROUP BY c.merchant_user_id ORDER BY total_payments DESC',
            'expected_limit': 5
        },
        {
            'query': 'show me the top 10 customers by subscription value',
            'sql': 'SELECT c.merchant_user_id, COALESCE(c.user_email, "Email not provided") as email, COALESCE(c.user_name, "Name not provided") as name, COALESCE(c.renewal_amount, c.max_amount_decimal, 0) as subscription_value FROM subscription_contract_v2 c ORDER BY COALESCE(c.renewal_amount, c.max_amount_decimal, 0) DESC',
            'expected_limit': 10
        }
    ]
    
    print("🔧 Testing LIMIT enforcement...")
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\nTest {i}: {test_case['query']}")
        print(f"Original SQL: {test_case['sql']}")
        
        # Apply LIMIT enforcement
        enforced_sql = processor._enforce_top_n_limit(test_case['sql'], test_case['query'])
        
        print(f"Enforced SQL: {enforced_sql}")
        
        if f"LIMIT {test_case['expected_limit']}" in enforced_sql:
            print(f"✅ LIMIT {test_case['expected_limit']} correctly enforced")
        else:
            print(f"❌ LIMIT {test_case['expected_limit']} NOT found")

async def main():
    """Run all tests"""
    print("🧪 Testing Query Splitting and LIMIT Enforcement Improvements")
    print("=" * 60)
    
    await test_query_splitting()
    print("\n" + "=" * 60)
    await test_limit_enforcement()
    print("\n" + "=" * 60)
    await test_real_execution()
    
    print("\n✅ All tests completed!")

if __name__ == "__main__":
    asyncio.run(main()) 