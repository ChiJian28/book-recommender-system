#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script to test the API endpoints
"""

import requests
import json

def test_api():
    """Test the API endpoints"""
    base_url = "http://localhost:8080"
    
    print("🔍 Testing API endpoints...")
    
    # 1. Test system info API
    print("\n1. Testing System Info API...")
    try:
        response = requests.get(f"{base_url}/api/system_info")
        print(f"Status Code: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(f"Response: {json.dumps(data, indent=2, ensure_ascii=False)}")
        else:
            print(f"Error: {response.text}")
    except Exception as e:
        print(f"❌ Request failed: {str(e)}")
    
    # 2. Test popular books API
    print("\n2. Testing Popular Books API...")
    try:
        response = requests.get(f"{base_url}/api/popular_books")
        print(f"Status Code: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(f"Response: {json.dumps(data, indent=2, ensure_ascii=False)}")
        else:
            print(f"Error: {response.text}")
    except Exception as e:
        print(f"❌ Request failed: {str(e)}")
    
    # 3. Test personalized recommendations API
    print("\n3. Testing Personalized Recommendations API...")
    user_id = 30944
    try:
        response = requests.get(f"{base_url}/api/personalized_recommendations/{user_id}")
        print(f"Status Code: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(f"Response: {json.dumps(data, indent=2, ensure_ascii=False)}")
        else:
            print(f"Error: {response.text}")
    except Exception as e:
        print(f"❌ Request failed: {str(e)}")
    
    # 4. Test content-based recommendations API
    print("\n4. Testing Content-Based Recommendations API...")
    book_id = 4  # To Kill a Mockingbird
    try:
        response = requests.get(f"{base_url}/api/content_recommendations/{book_id}")
        print(f"Status Code: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(f"Response: {json.dumps(data, indent=2, ensure_ascii=False)}")
        else:
            print(f"Error: {response.text}")
    except Exception as e:
        print(f"❌ Request failed: {str(e)}")

if __name__ == "__main__":
    test_api()
