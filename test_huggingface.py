#!/usr/bin/env python3
"""
Test script for Hugging Face image generation
"""

from ai import generate_image_huggingface, generate_and_save_image

def test_huggingface():
    print("🧪 Testing Hugging Face Image Generation...")
    print("=" * 50)
    
    # Test 1: Direct function call
    print("1. Testing direct Hugging Face function...")
    result = generate_image_huggingface("a beautiful sunset", "stable-diffusion")
    if isinstance(result, str) and result.startswith("Error"):
        print(f"❌ Error: {result}")
    else:
        print("✅ Direct function call successful!")
    
    # Test 2: Using the wrapper function
    print("\n2. Testing wrapper function...")
    filename, status = generate_and_save_image(
        "a cute cat", 
        use_stable_diffusion=False, 
        use_huggingface=True, 
        hf_model="stable-diffusion"
    )
    print(f"Status: {status}")
    if filename:
        print(f"✅ Image saved to: {filename}")
    else:
        print("❌ Failed to generate image")
    
    # Test 3: Try different model
    print("\n3. Testing different model (openjourney)...")
    filename2, status2 = generate_and_save_image(
        "a futuristic city", 
        use_stable_diffusion=False, 
        use_huggingface=True, 
        hf_model="openjourney"
    )
    print(f"Status: {status2}")
    if filename2:
        print(f"✅ Image saved to: {filename2}")
    else:
        print("❌ Failed to generate image")
    
    print("\n" + "=" * 50)
    print("Test completed!")

if __name__ == "__main__":
    test_huggingface() 