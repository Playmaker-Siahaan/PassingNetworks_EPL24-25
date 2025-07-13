
#!/usr/bin/env python3
"""
AI Setup Validation Script
Validates API keys and AI functionality
"""

import os
import requests
import json
from datetime import datetime

def validate_openai_api(api_key):
    """Validate OpenAI API key"""
    if not api_key:
        return False, "API key not provided"
    
    try:
        headers = {
            'Authorization': f'Bearer {api_key}',
            'Content-Type': 'application/json'
        }
        
        # Simple test request
        data = {
            'model': 'gpt-3.5-turbo',
            'messages': [{'role': 'user', 'content': 'Hello'}],
            'max_tokens': 5
        }
        
        response = requests.post(
            'https://api.openai.com/v1/chat/completions',
            headers=headers,
            json=data,
            timeout=10
        )
        
        if response.status_code == 200:
            return True, "OpenAI API validated successfully"
        else:
            return False, f"API error: {response.status_code}"
            
    except Exception as e:
        return False, f"Connection error: {str(e)}"

def validate_stability_ai(api_key):
    """Validate Stability AI API key"""
    if not api_key:
        return False, "API key not provided"
    
    try:
        headers = {
            'Authorization': f'Bearer {api_key}',
            'Content-Type': 'application/json'
        }
        
        # Test with account info endpoint
        response = requests.get(
            'https://api.stability.ai/v1/user/account',
            headers=headers,
            timeout=10
        )
        
        if response.status_code == 200:
            return True, "Stability AI API validated successfully"
        else:
            return False, f"API error: {response.status_code}"
            
    except Exception as e:
        return False, f"Connection error: {str(e)}"

def validate_huggingface_token(token):
    """Validate Hugging Face token"""
    if not token:
        return False, "Token not provided"
    
    try:
        headers = {
            'Authorization': f'Bearer {token}',
            'Content-Type': 'application/json'
        }
        
        # Test with whoami endpoint
        response = requests.get(
            'https://huggingface.co/api/whoami',
            headers=headers,
            timeout=10
        )
        
        if response.status_code == 200:
            return True, "Hugging Face token validated successfully"
        else:
            return False, f"API error: {response.status_code}"
            
    except Exception as e:
        return False, f"Connection error: {str(e)}"

def main():
    """Main validation function"""
    print("🤖 AI Setup Validation Script")
    print("=" * 50)
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Get API keys from environment
    openai_key = os.getenv('OPENAI_API_KEY')
    stability_key = os.getenv('STABILITY_API_KEY')
    huggingface_token = os.getenv('HUGGINGFACE_TOKEN')
    
    # Validation results
    results = []
    
    # Validate OpenAI
    print("🔍 Validating OpenAI API...")
    valid, message = validate_openai_api(openai_key)
    results.append(("OpenAI", valid, message))
    print(f"   {'✅' if valid else '❌'} {message}")
    
    # Validate Stability AI
    print("\n🔍 Validating Stability AI API...")
    valid, message = validate_stability_ai(stability_key)
    results.append(("Stability AI", valid, message))
    print(f"   {'✅' if valid else '❌'} {message}")
    
    # Validate Hugging Face
    print("\n🔍 Validating Hugging Face Token...")
    valid, message = validate_huggingface_token(huggingface_token)
    results.append(("Hugging Face", valid, message))
    print(f"   {'✅' if valid else '❌'} {message}")
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 VALIDATION SUMMARY")
    print("=" * 50)
    
    valid_count = sum(1 for _, valid, _ in results if valid)
    total_count = len(results)
    
    for service, valid, message in results:
        status = "✅ VALID" if valid else "❌ INVALID"
        print(f"{service:15} | {status}")
    
    print(f"\nOverall: {valid_count}/{total_count} APIs validated")
    
    if valid_count == total_count:
        print("🎉 All AI features are ready to use!")
    elif valid_count > 0:
        print("⚠️  Some AI features available, others will use fallback mode")
    else:
        print("ℹ️  No AI APIs configured, using fallback mode")
    
    print("\n💡 Next Steps:")
    if valid_count < total_count:
        print("• Add missing API keys to Replit Secrets")
        print("• Restart the application")
        print("• Run this script again to verify")
    else:
        print("• Start using AI features in the main application")
        print("• Monitor API usage in respective dashboards")

if __name__ == "__main__":
    main()
