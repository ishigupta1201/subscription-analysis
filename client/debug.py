#!/usr/bin/env python3
"""
Debug script to check environment variable loading
Run this in both your IDE and terminal to compare
"""
import os
import sys
from pathlib import Path

print("🔍 ENVIRONMENT VARIABLE DEBUG")
print("=" * 50)

# Check current working directory
print(f"📁 Current working directory: {os.getcwd()}")
print(f"📄 Script location: {__file__ if '__file__' in globals() else 'Unknown'}")

# Check for .env files in various locations
env_locations = [
    '.env',
    '../.env', 
    os.path.join(os.getcwd(), '.env'),
    os.path.join(Path(__file__).parent if '__file__' in globals() else '.', '.env'),
    os.path.join(Path(__file__).parent.parent if '__file__' in globals() else '.', '.env'),
]

print(f"\n📋 Checking .env file locations:")
for location in env_locations:
    exists = os.path.exists(location)
    print(f"  {'✅' if exists else '❌'} {location}")
    if exists:
        try:
            with open(location, 'r') as f:
                content = f.read()
                has_api_key = 'API_KEY' in content
                has_subscription_url = 'SUBSCRIPTION_API_URL' in content
                has_gemini = 'GEMINI_API_KEY' in content
                print(f"    📊 Contains API_KEY: {'✅' if has_api_key else '❌'}")
                print(f"    📊 Contains SUBSCRIPTION_API_URL: {'✅' if has_subscription_url else '❌'}")
                print(f"    📊 Contains GEMINI_API_KEY: {'✅' if has_gemini else '❌'}")
        except Exception as e:
            print(f"    ❌ Error reading file: {e}")

# Try loading with python-dotenv
print(f"\n🔄 Testing python-dotenv loading:")
try:
    from dotenv import load_dotenv
    
    for location in ['.env', '../.env']:
        if os.path.exists(location):
            result = load_dotenv(location)
            print(f"  📁 {location}: {'✅ Loaded' if result else '❌ Failed'}")
except ImportError:
    print("  ❌ python-dotenv not installed")

# Check environment variables
print(f"\n🔑 Environment Variables:")
env_vars = [
    'API_KEY_1',
    'SUBSCRIPTION_API_KEY', 
    'SUBSCRIPTION_API_URL',
    'GEMINI_API_KEY'
]

for var in env_vars:
    value = os.getenv(var)
    if value:
        print(f"  ✅ {var}: {value[:20]}{'...' if len(value) > 20 else ''}")
    else:
        print(f"  ❌ {var}: Not set")

# Check Python path
print(f"\n🐍 Python Environment:")
print(f"  Python executable: {sys.executable}")
print(f"  Python path: {sys.path[:3]}...")  # First 3 entries

# Check if we're in a virtual environment
venv = os.getenv('VIRTUAL_ENV')
conda_env = os.getenv('CONDA_DEFAULT_ENV')
print(f"  Virtual environment: {venv if venv else 'None'}")
print(f"  Conda environment: {conda_env if conda_env else 'None'}")

print("\n" + "=" * 50)
print("💡 TROUBLESHOOTING TIPS:")
print("1. Make sure .env file is in the correct directory")
print("2. Check that your IDE is using the same Python interpreter")
print("3. Verify your IDE loads environment variables from .env")
print("4. Try restarting your IDE after setting up environment")