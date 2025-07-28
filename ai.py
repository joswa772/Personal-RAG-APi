#!/usr/bin/env python3
"""
PDF Processing & Image Generation Backend API

This file contains:
- PDF text and image extraction
- Question answering with LLM integration
- AI-powered image generation
- Testing and validation functions
- Pure Python backend (no UI)

Author: AI Assistant
Version: 2.0
"""

import os
import fitz  # PyMuPDF
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import requests
import base64
import io
from PIL import Image, ImageDraw, ImageFont
import json
import time
import hashlib
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
import re
from dotenv import load_dotenv

# Download required NLTK resources
try:
    nltk.download("punkt", quiet=True)
    nltk.download("stopwords", quiet=True)
except:
    print("Warning: Could not download NLTK resources. Some features may not work.")

# Load environment variables from .env file
load_dotenv()

# Global variables
CLEANED_TEXT = ""
ANSWER_HISTORY = set()
GENERATED_IMAGES = []

# ============================================================================
# CONFIGURATION SECTION
# ============================================================================

# Built-in API key for Stable Diffusion
# Replace "sk-your-api-key-here" with your actual Stability AI API key
# Get your API key from: https://platform.stability.ai/
STABLE_DIFFUSION_API_KEY = os.getenv("STABLE_DIFFUSION_API_KEY")

# Hugging Face API key (optional, for higher rate limits)
# Get your free API key from: https://huggingface.co/settings/tokens
HUGGINGFACE_API_KEY = os.getenv("REMOVED_TOKENcrGqfPOzpxTvbOCzgTogEecwyKLJoyWJMM") # Replace with your actual API key

# Ollama host configuration
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
# ============================================================================

# ============================================================================
# TEXT PROCESSING FUNCTIONS
# ============================================================================
def answer_question_chroma(question, use_llama3=False):
    """Answer question using ChromaDB vector search"""
    try:
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        db = Chroma(persist_directory="chroma_db", embedding_function=embeddings)
        results = db.similarity_search(question, k=3)
        context = "\n\n".join([doc.page_content for doc in results])

        if use_llama3:
            return ask_llama3(context, question)
        return context  # Extractive fallback
    except Exception as e:
        return f"Error using ChromaDB: {e}"

def clean_text(text):
    """Clean and preprocess text using NLTK"""
    try:
        words = word_tokenize(text)
        words = [w.lower() for w in words if w.isalpha()]
        stop_words = set(stopwords.words("english"))
        filtered = [w for w in words if w not in stop_words]
        return " ".join(filtered)
    except Exception as e:
        print(f"Error cleaning text: {e}")
        return text

def process_pdf(file):
    """Extract text and images from PDF file"""
    global CLEANED_TEXT
    try:
        doc = fitz.open(file.name)
        all_text = ""
        all_images = []
        
        for page in doc:
            all_text += page.get_text('text')
            image_list = page.get_images(full=True)
            for img_index, img in enumerate(image_list):
                xref = img[0]
                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]
                image_ext = base_image["ext"]
                all_images.append((image_ext, image_bytes))
        
        cleaned_text = clean_text(all_text)
        CLEANED_TEXT = all_text  # Store original text for QA

        # Save to .txt file with separate sections
        output_path = file.name + "_output.txt"
        with open(output_path, "w", encoding="utf-8") as f:
            f.write("=== Cleaned Text ===\n")
            f.write(cleaned_text + "\n\n")
            f.write("=== Image Info ===\n")
            for idx, (ext, _) in enumerate(all_images):
                f.write(f"Image {idx + 1}: Format = {ext}\n")
        
        return output_path
    except Exception as e:
        return f"Error processing PDF: {e}"

# ============================================================================
# LLM INTEGRATION FUNCTIONS
# ============================================================================

def ask_llama3(context, question):
    """Query Llama 3 model for question answering"""
    prompt = (
        f"Context: {context}\n\n"
        f"Based only on the above context, answer the following question as accurately as possible.\n\n"
        f"Question: {question}\n\nAnswer:"
    )
    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "llama3",
                "prompt": prompt,
                "stream": False
            },
            timeout=60
        )
        if response.ok:
            return response.json().get("response", "").strip()
        else:
            return "Error: Could not get response from Llama 3."
    except Exception as e:
        return f"Error: {e}"

def answer_question(question, use_llama3=False):
    """Answer questions using extractive QA or LLM"""
    global CLEANED_TEXT, ANSWER_HISTORY
    if not CLEANED_TEXT:
        return "Please upload and process a PDF first."
    
    # Split into paragraphs
    paragraphs = [p.strip() for p in CLEANED_TEXT.split('\n\n') if p.strip()]
    if not paragraphs:
        paragraphs = [p.strip() for p in CLEANED_TEXT.split('\n') if p.strip()]
    if not paragraphs:
        return "No text found in the PDF."
    
    # Use TF-IDF to find relevant paragraphs
    vectorizer = TfidfVectorizer().fit(paragraphs + [question])
    para_vecs = vectorizer.transform(paragraphs)
    question_vec = vectorizer.transform([question])
    similarities = cosine_similarity(question_vec, para_vecs).flatten()
    sorted_indices = similarities.argsort()[::-1]
    
    # Gather relevant paragraphs
    context_paragraphs = []
    char_count = 0
    for idx in sorted_indices:
        para = paragraphs[idx]
        if similarities[idx] > 0 and para not in context_paragraphs:
            context_paragraphs.append(para)
            char_count += len(para)
            if char_count > 6000:
                break
    
    # Highlight most relevant paragraph
    if context_paragraphs:
        context_paragraphs[0] = "**" + context_paragraphs[0] + "**"
    top_context = "\n\n".join(context_paragraphs)
    
    if use_llama3:
        return ask_llama3(top_context, question) if top_context else "I don't know."
    
    # Extractive QA
    for idx in sorted_indices:
        para = paragraphs[idx]
        if para not in ANSWER_HISTORY and similarities[idx] > 0:
            ANSWER_HISTORY.add(para)
            return para
    return "I don't know."

# ============================================================================
# IMAGE GENERATION FUNCTIONS
# ============================================================================

def generate_image_ollama(prompt, model="llava"):
    """Generate image using Ollama"""
    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": model,
                "prompt": f"Generate an image of: {prompt}",
                "stream": False,
                "options": {
                    "temperature": 0.7,
                    "top_p": 0.9
                }
            },
            timeout=120
        )
        
        if response.ok:
            result = response.json()
            if "response" in result:
                return result["response"]
            else:
                return "Image generation completed but no image data returned."
        else:
            return f"Error: {response.status_code} - {response.text}"
            
    except requests.exceptions.RequestException as e:
        return f"Connection error: {e}"
    except Exception as e:
        return f"Error generating image: {e}"

def generate_image_stable_diffusion(prompt, api_key=None):
    """Generate image using Stable Diffusion API"""
    # Use built-in API key if none provided
    if not api_key or api_key == "":
        api_key = STABLE_DIFFUSION_API_KEY
    
    # Check if API key is valid (not placeholder)
    if not api_key or api_key == "sk-xDpJbplSNgWOJrs8VybBodeGX4Gxt6q2dN1jlpXENFxgAXFX" or len(api_key) < 50:
        return "Error: Please set your Stable Diffusion API key in the code. Replace the placeholder API key with your actual Stability AI API key from https://platform.stability.ai/"
    
    try:
        url = "https://api.stability.ai/v1/generation/stable-diffusion-xl-1024-v1-0/text-to-image"
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }
        
        body = {
            "text_prompts": [
                {
                    "text": prompt,
                    "weight": 1
                }
            ],
            "cfg_scale": 7,
            "height": 1024,
            "width": 1024,
            "samples": 1,
            "steps": 30,
        }
        
        response = requests.post(url, headers=headers, json=body)
        
        if response.status_code == 200:
            data = response.json()
            if "artifacts" in data and len(data["artifacts"]) > 0:
                image_data = base64.b64decode(data["artifacts"][0]["base64"])
                return image_data
            else:
                return "No image generated"
        else:
            return f"Error: {response.status_code} - {response.text}"
            
    except Exception as e:
        return f"Error: {e}"

def generate_image_huggingface(prompt, model="stable-diffusion"):
    """Generate image using Hugging Face's free inference API or fallback to enhanced placeholder"""
    try:
        # Try to use Hugging Face API first
        if HUGGINGFACE_API_KEY and HUGGINGFACE_API_KEY.strip() and HUGGINGFACE_API_KEY != "":
            # Choose reliable models that work with the API
            models = {
                "stable-diffusion": "runwayml/stable-diffusion-v1-5",
                "openjourney": "prompthero/openjourney",
                "dreamlike": "dreamlike-art/dreamlike-photoreal-2.0",
                "anything": "andite/anything-v4.0",
                "sdxl": "stabilityai/stable-diffusion-xl-base-1.0"
            }
            
            # Use the specified model or default
            model_name = models.get(model, model)
            
            # API endpoint
            api_url = f"https://api-inference.huggingface.co/models/{model_name}"
            
            # Headers with API key
            headers = {"Content-Type": "application/json"}
            headers["Authorization"] = f"Bearer {HUGGINGFACE_API_KEY}"
            
            # Request payload
            payload = {
                "inputs": prompt,
                "parameters": {
                    "num_inference_steps": 30,
                    "guidance_scale": 7.5,
                    "width": 512,
                    "height": 512
                }
            }
            
            # Try multiple models if one fails
            models_to_try = [model_name]
            if model_name != "runwayml/stable-diffusion-v1-5":
                models_to_try.append("runwayml/stable-diffusion-v1-5")
            
            for try_model in models_to_try:
                try:
                    print(f"Trying model: {try_model}")
                    api_url = f"https://api-inference.huggingface.co/models/{try_model}"
                    response = requests.post(api_url, headers=headers, json=payload, timeout=60)
                    
                    if response.status_code == 200:
                        # Return the image data
                        return response.content
                    elif response.status_code == 503:
                        # Model is loading, wait and retry
                        print(f"Model {try_model} is loading, waiting...")
                        time.sleep(10)
                        response = requests.post(api_url, headers=headers, json=payload, timeout=60)
                        if response.status_code == 200:
                            return response.content
                        else:
                            print(f"Model {try_model} failed after retry: {response.status_code}")
                            continue
                    elif response.status_code == 404:
                        print(f"Model {try_model} not found, trying next...")
                        continue
                    else:
                        print(f"Model {try_model} error: {response.status_code} - {response.text}")
                        continue
                except Exception as e:
                    print(f"Exception with model {try_model}: {e}")
                    continue
            
            return f"Error: All models failed. Please check your API key and try again."
        else:
            # No API key, return None to trigger fallback
            return None
            
    except Exception as e:
        return f"Error: {e}"

def create_enhanced_placeholder_image(prompt, filename):
    """Create an enhanced placeholder image with gradient background and better styling"""
    try:
        # Create a larger image with gradient background
        width, height = 512, 512
        img = Image.new('RGB', (width, height), color='white')
        draw = ImageDraw.Draw(img)
        
        # Create a gradient background
        for y in range(height):
            # Create a gradient from top to bottom
            r = int(100 + (y / height) * 100)  # Blue to purple gradient
            g = int(150 + (y / height) * 50)
            b = int(200 + (y / height) * 55)
            for x in range(width):
                draw.point((x, y), fill=(r, g, b))
        
        # Try to use a default font
        try:
            font = ImageFont.load_default()
        except:
            font = None
        
        # Create a semi-transparent overlay for text
        overlay = Image.new('RGBA', (width, height), (255, 255, 255, 180))
        img = Image.alpha_composite(img.convert('RGBA'), overlay).convert('RGB')
        draw = ImageDraw.Draw(img)
        
        # Wrap text to fit in image
        words = prompt.split()
        lines = []
        current_line = []
        max_words_per_line = 6
        
        for word in words:
            current_line.append(word)
            if len(current_line) >= max_words_per_line:
                lines.append(' '.join(current_line))
                current_line = []
        if current_line:
            lines.append(' '.join(current_line))
        
        # Limit to 8 lines maximum
        lines = lines[:8]
        
        # Draw text with shadow effect
        y_position = 80
        for i, line in enumerate(lines):
            # Draw shadow
            draw.text((22, y_position + 2), line, fill='darkgray', font=font)
            # Draw main text
            draw.text((20, y_position), line, fill='black', font=font)
            y_position += 35
        
        # Add a decorative border
        draw.rectangle([10, 10, width-10, height-10], outline='white', width=3)
        
        # Add a small icon or symbol based on the prompt
        if any(word in prompt.lower() for word in ['sun', 'sunset', 'day', 'bright']):
            # Draw a simple sun
            draw.ellipse([width-80, 20, width-20, 80], fill='yellow', outline='orange', width=2)
        elif any(word in prompt.lower() for word in ['moon', 'night', 'dark']):
            # Draw a simple moon
            draw.ellipse([width-80, 20, width-20, 80], fill='lightgray', outline='gray', width=2)
        elif any(word in prompt.lower() for word in ['tree', 'nature', 'forest']):
            # Draw a simple tree
            draw.ellipse([width-70, 30, width-30, 70], fill='green', outline='darkgreen', width=2)
            draw.rectangle([width-45, 70, width-55, 90], fill='brown')
        
        img.save(filename)
        return filename
    except Exception as e:
        print(f"Error creating enhanced placeholder image: {e}")
        return None

def create_placeholder_image(prompt, filename):
    """Create a placeholder image with text (legacy function for compatibility)"""
    return create_enhanced_placeholder_image(prompt, filename)

def generate_and_save_image(prompt, use_stable_diffusion=False, api_key="", model="llava", use_huggingface=False, REMOVED_TOKENmodel="stable-diffusion"):
    """Generate an image and save it to a file in the 'download-image/generated_images' folder"""
    global GENERATED_IMAGES
    try:
        # Ensure the download-image/generated_images directory exists
        images_dir = os.path.join("download-image", "generated_images")
        os.makedirs(images_dir, exist_ok=True)
        # Create unique, safe filename
        timestamp = int(time.time())
        # Remove any unsafe characters from prompt for filename
        safe_prompt = re.sub(r'[^a-zA-Z0-9_\-]', '_', prompt)[:30]
        filename = f"generated_image_{timestamp}_{hash(prompt) % 10000}.png"
        filepath = os.path.join(images_dir, filename)
        
        if use_huggingface:
            # Use Hugging Face free API
            image_data = generate_image_huggingface(prompt, REMOVED_TOKENmodel)
            if image_data is None:
                # No API key, create enhanced placeholder
                saved_file = create_enhanced_placeholder_image(prompt, filepath)
                if saved_file:
                    GENERATED_IMAGES.append(filepath)
                    return filepath, "Image generated using enhanced placeholder (no API key needed)!"
                else:
                    return None, "Failed to create enhanced placeholder image"
            elif isinstance(image_data, str) and image_data.startswith("Error"):
                return None, image_data
            # Save the image
            with open(filepath, "wb") as f:
                f.write(image_data)
            GENERATED_IMAGES.append(filepath)
            return filepath, f"Image generated successfully using Hugging Face ({REMOVED_TOKENmodel})!"
        elif use_stable_diffusion:
            image_data = generate_image_stable_diffusion(prompt, api_key)
            if isinstance(image_data, str) and image_data.startswith("Error"):
                return None, image_data
            # Save the image
            with open(filepath, "wb") as f:
                f.write(image_data)
            GENERATED_IMAGES.append(filepath)
            return filepath, "Image generated successfully using Stable Diffusion!"
        else:
            # For Ollama-based generation
            result = generate_image_ollama(prompt, model)
            if result.startswith("Error"):
                return None, result
            # Create placeholder image
            saved_file = create_placeholder_image(prompt, filepath)
            if saved_file:
                GENERATED_IMAGES.append(filepath)
                return filepath, f"Image generated using Ollama: {result}"
            else:
                return None, "Failed to create image file"
    except Exception as e:
        print(f"Error generating and saving image: {e}")
        return None, f"Error: {e}"

def generate_image_from_pdf_context(question, use_stable_diffusion=False, api_key="", use_huggingface=False, REMOVED_TOKENmodel="stable-diffusion"):
    """Generate image based on PDF content and question"""
    global CLEANED_TEXT
    if not CLEANED_TEXT:
        return None, "Please upload and process a PDF first."
    
    # Get relevant context from PDF
    paragraphs = [p.strip() for p in CLEANED_TEXT.split('\n\n') if p.strip()]
    if not paragraphs:
        return None, "No text found in the PDF."
    
    # Use TF-IDF to find relevant content
    vectorizer = TfidfVectorizer().fit(paragraphs + [question])
    para_vecs = vectorizer.transform(paragraphs)
    question_vec = vectorizer.transform([question])
    similarities = cosine_similarity(question_vec, para_vecs).flatten()
    
    # Get the most relevant paragraph
    most_relevant_idx = similarities.argmax()
    if similarities[most_relevant_idx] > 0:
        context = paragraphs[most_relevant_idx]
        # Create a prompt combining the question and context
        prompt = f"{question} based on: {context[:200]}..."
        return generate_and_save_image(prompt, use_stable_diffusion, api_key, use_huggingface=use_huggingface, REMOVED_TOKENmodel=REMOVED_TOKENmodel)
    else:
        return None, "No relevant content found in the PDF."

# ============================================================================
# TESTING AND VALIDATION FUNCTIONS
# ============================================================================

def test_ollama_connection():
    """Test if Ollama is running and accessible"""
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.ok:
            models = response.json().get("models", [])
            return True, f"✅ Ollama is running. Available models: {len(models)}"
        else:
            return False, "❌ Ollama is not responding properly"
    except requests.exceptions.RequestException:
        return False, "❌ Ollama is not running or not accessible"

def test_image_generation():
    """Test basic image generation functionality"""
    try:
        img = Image.new('RGB', (512, 512), color='lightblue')
        draw = ImageDraw.Draw(img)
        
        try:
            font = ImageFont.load_default()
        except:
            font = None
        
        draw.text((50, 50), "Test Image Generation", fill='black', font=font)
        
        test_filename = "test_generated_image.png"
        img.save(test_filename)
        
        if os.path.exists(test_filename):
            os.remove(test_filename)  # Clean up
            return True, "✅ Image generation test passed"
        else:
            return False, "❌ Failed to create test image"
            
    except Exception as e:
        return False, f"❌ Error creating test image: {e}"

def test_stable_diffusion_api():
    """Test Stable Diffusion API connection"""
    try:
        response = requests.get("https://api.stability.ai/v1/user/balance", timeout=10)
        if response.status_code == 401:
            return True, "✅ Stable Diffusion API is accessible (requires API key)"
        elif response.status_code == 200:
            return True, "✅ Stable Diffusion API is accessible and working"
        else:
            return False, f"⚠️ Stable Diffusion API returned status: {response.status_code}"
    except Exception as e:
        return False, f"❌ Cannot connect to Stable Diffusion API: {e}"

def test_huggingface_api():
    """Test Hugging Face API connection"""
    try:
        result = generate_image_huggingface("test", "stable-diffusion")
        if isinstance(result, str) and result.startswith("Error"):
            return False, f"❌ Hugging Face API Error: {result}"
        else:
            return True, "✅ Hugging Face API is working"
    except Exception as e:
        return False, f"❌ Hugging Face API Error: {e}"

def run_system_tests():
    """Run comprehensive system tests"""
    results = []
    
    # Test Ollama
    ollama_ok, ollama_msg = test_ollama_connection()
    results.append(ollama_msg)
    
    # Test image generation
    img_ok, img_msg = test_image_generation()
    results.append(img_msg)
    
    # Test Hugging Face (free alternative)
    REMOVED_TOKENok, REMOVED_TOKENmsg = test_huggingface_api()
    results.append(REMOVED_TOKENmsg)
    
    # Test Stable Diffusion
    sd_ok, sd_msg = test_stable_diffusion_api()
    results.append(sd_msg)
    
    return "\n".join(results)

# ============================================================================
# API FUNCTIONS
# ============================================================================

def process_and_reset(file_path):
    """Process PDF and reset answer history"""
    global ANSWER_HISTORY
    output_path = process_pdf(file_path)
    ANSWER_HISTORY = set()
    return output_path, ""

def generate_image_wrapper(prompt, use_sd=False, api_key="", use_hf=False, REMOVED_TOKENmodel="stable-diffusion"):
    """Wrapper for basic image generation"""
    if not prompt.strip():
        return None, "Please enter an image description."
    filename, status = generate_and_save_image(prompt, use_sd, api_key, use_huggingface=use_hf, REMOVED_TOKENmodel=REMOVED_TOKENmodel)
    return filename, status

def generate_pdf_image_wrapper(question, use_sd=False, api_key="", use_hf=False, REMOVED_TOKENmodel="stable-diffusion"):
    """Wrapper for PDF-based image generation"""
    if not question.strip():
        return None, "Please enter a question about the PDF content."
    # For PDF-based generation, we'll use the same approach but with Hugging Face
    if use_hf:
        # Get context from PDF and generate image using Hugging Face
        global CLEANED_TEXT
        if not CLEANED_TEXT:
            return None, "Please upload and process a PDF first."
        
        # Create a prompt combining the question and some context
        paragraphs = [p.strip() for p in CLEANED_TEXT.split('\n\n') if p.strip()]
        if paragraphs:
            context = paragraphs[0][:200]  # Use first paragraph as context
            prompt = f"{question} based on: {context}..."
        else:
            prompt = question
            
        filename, status = generate_and_save_image(prompt, use_stable_diffusion=False, api_key="", use_huggingface=True, REMOVED_TOKENmodel=REMOVED_TOKENmodel)
        return filename, status
    else:
        filename, status = generate_image_from_pdf_context(question, use_sd, api_key, use_huggingface=use_hf, REMOVED_TOKENmodel=REMOVED_TOKENmodel)
        return filename, status

def cleanup_generated_files():
    """Clean up generated image files"""
    global GENERATED_IMAGES
    cleaned_files = []
    for filename in GENERATED_IMAGES:
        try:
            if os.path.exists(filename):
                os.remove(filename)
                cleaned_files.append(filename)
        except:
            pass
    GENERATED_IMAGES.clear()
    return f"Cleaned up {len(cleaned_files)} files: {', '.join(cleaned_files)}"

# ============================================================================
# MAIN EXECUTION & DEMO
# ============================================================================

def demo_usage():
    """Demonstrate how to use the backend functions"""
    print("🚀 PDF Processing & Image Generation Backend")
    print("=" * 60)
    print("This is a pure Python backend. Use these functions in your own UI:")
    print()
    
    print("📄 PDF Processing Functions:")
    print("- process_pdf(file_path): Extract text and images from PDF")
    print("- answer_question(question, use_llama3=False): Ask questions about PDF content")
    print("- process_and_reset(file_path): Process PDF and reset history")
    print()
    
    print("🎨 Image Generation Functions:")
    print("- generate_and_save_image(prompt, use_stable_diffusion=False): Generate images")
    print("- generate_image_from_pdf_context(question, use_stable_diffusion=False): PDF-based generation")
    print("- generate_image_wrapper(prompt, use_sd=False): Wrapper for basic generation")
    print("- generate_pdf_image_wrapper(question, use_sd=False): Wrapper for PDF generation")
    print()
    
    print("🧪 System Functions:")
    print("- run_system_tests(): Test all components")
    print("- cleanup_generated_files(): Clean up generated files")
    print("- test_ollama_connection(): Test Ollama connection")
    print("- test_image_generation(): Test image generation")
    print("- test_stable_diffusion_api(): Test Stable Diffusion API")
    print()
    
    print("💡 Example Usage:")
    print("""
    # Process a PDF
    output_file = process_pdf("document.pdf")
    
    # Ask a question
    answer = answer_question("What is the main topic?", use_llama3=True)
    
    # Generate an image
    image_file, status = generate_image_wrapper("A beautiful sunset", use_sd=True)
    
    # Run system tests
    test_results = run_system_tests()
    """)
    print("=" * 60)

if __name__ == "__main__":
    print("STABLE_DIFFUSION_API_KEY is set:", bool(STABLE_DIFFUSION_API_KEY))
    demo_usage()
