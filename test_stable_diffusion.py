#!/usr/bin/env python3
"""
Test script to verify Stable Diffusion API
"""

from ai import generate_image_stable_diffusion, STABLE_DIFFUSION_API_KEY

def test_stable_diffusion():
    print("🎨 Testing Stable Diffusion API...")
    print("=" * 50)
    
    print(f"API Key: {STABLE_DIFFUSION_API_KEY[:10]}...")
    
    # Test the API
    result = generate_image_stable_diffusion("a beautiful sunset", STABLE_DIFFUSION_API_KEY)
    
    if isinstance(result, str) and result.startswith("Error"):
        print(f"❌ Error: {result}")
    else:
        print("✅ Stable Diffusion API is working!")
        print(f"Generated image data size: {len(result)} bytes")
    
    print("\n" + "=" * 50)
    print("Test completed!")

if __name__ == "__main__":
    test_stable_diffusion() 