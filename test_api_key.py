#!/usr/bin/env python3
"""
Test script to verify Hugging Face API key
"""

import os
import requests

def test_api_key():
    
    print("🔑 Testing Hugging Face API Key...")
    print("=" * 50)
    
    # Test 1: Check if API key is valid
    print("1. Testing API key validity...")
    headers = {"Authorization": f"Bearer {api_key}"}
    
    # Try to access user info
    try:
        response = requests.get("https://huggingface.co/api/whoami", headers=headers)
        if response.status_code == 200:
            user_info = response.json()
            print(f"✅ API key is valid! User: {user_info.get('name', 'Unknown')}")
        else:
            print(f"❌ API key validation failed: {response.status_code}")
            return
    except Exception as e:
        print(f"❌ Error testing API key: {e}")
        return
    
    # Test 2: Try a simple model
    print("\n2. Testing model access...")
    models_to_test = [
        "runwayml/stable-diffusion-v1-5",
        "prompthero/openjourney",
        "CompVis/stable-diffusion-v1-4"
    ]
    
    for model in models_to_test:
        try:
            print(f"   Testing {model}...")
            api_url = f"https://api-inference.huggingface.co/models/{model}"
            payload = {"inputs": "test"}
            response = requests.post(api_url, headers=headers, json=payload, timeout=30)
            
            if response.status_code == 200:
                print(f"   ✅ {model} - Working!")
                break
            elif response.status_code == 503:
                print(f"   ⏳ {model} - Loading (this is normal)")
                break
            elif response.status_code == 404:
                print(f"   ❌ {model} - Not found")
            else:
                print(f"   ⚠️ {model} - Status: {response.status_code}")
        except Exception as e:
            print(f"   ❌ {model} - Error: {e}")
    
    print("\n" + "=" * 50)
    print("API key test completed!")

if __name__ == "__main__":
    test_api_key() 