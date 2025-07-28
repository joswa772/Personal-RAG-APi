#!/usr/bin/env python3
"""
Test script to check current state of image generation
"""

from ai import generate_image_wrapper, generate_pdf_image_wrapper

def test_current_state():
    print("🔍 Testing Current Image Generation State...")
    print("=" * 50)
    
    # Test 1: Regular image generation
    print("1. Testing regular image generation...")
    filename, status = generate_image_wrapper(
        "a beautiful sunset", 
        use_sd=False, 
        use_hf=True, 
        hf_model="stable-diffusion"
    )
    print(f"Status: {status}")
    if filename:
        print(f"✅ Image saved to: {filename}")
    else:
        print("❌ Failed to generate image")
    
    # Test 2: PDF-based image generation (simulate with some text)
    print("\n2. Testing PDF-based image generation...")
    from ai import CLEANED_TEXT
    # Set some dummy text to simulate PDF content
    import ai
    ai.CLEANED_TEXT = "This is a test PDF about artificial intelligence and machine learning."
    
    filename2, status2 = generate_pdf_image_wrapper(
        "generate an image of AI", 
        use_sd=False, 
        use_hf=True, 
        hf_model="stable-diffusion"
    )
    print(f"Status: {status2}")
    if filename2:
        print(f"✅ Image saved to: {filename2}")
    else:
        print("❌ Failed to generate image")
    
    print("\n" + "=" * 50)
    print("Current state test completed!")

if __name__ == "__main__":
    test_current_state() 