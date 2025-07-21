#!/usr/bin/env python3
"""
FastAPI Web Interface for PDF Processing & Image Generation

This file provides:
- REST API endpoints for all backend functions
- Modern web UI with HTML/CSS/JavaScript
- File upload handling
- Real-time image generation
- Interactive question answering

Author: AI Assistant
Version: 1.0
"""

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import os
import tempfile
import shutil
from pathlib import Path
import base64
from typing import Optional

# Import our backend functions
from ai import (
    process_pdf, answer_question, generate_image_wrapper, 
    generate_pdf_image_wrapper
)

# Initialize FastAPI app
app = FastAPI(
    title="Personal RAG API",
    description="A modern web interface for PDF processing and AI image generation",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create uploads directory
UPLOADS_DIR = Path("uploads")
UPLOADS_DIR.mkdir(exist_ok=True)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# ============================================================================
# HTML TEMPLATES
# ============================================================================

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>RAG API</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            color: #333;
        }

        .container {
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
        }

        .header {
            text-align: center;
            margin-bottom: 30px;
            color: white;
        }

        .header h1 {
            font-size: 2.5rem;
            margin-bottom: 10px;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        }

        .header p {
            font-size: 1.1rem;
            opacity: 0.9;
        }

        .main-content {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 30px;
            margin-bottom: 30px;
        }

        .card {
            background: white;
            border-radius: 15px;
            padding: 25px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.2);
            transition: transform 0.3s ease;
        }

        .card:hover {
            transform: translateY(-5px);
        }

        .card h2 {
            color: #667eea;
            margin-bottom: 20px;
            font-size: 1.5rem;
            border-bottom: 2px solid #f0f0f0;
            padding-bottom: 10px;
        }

        .form-group {
            margin-bottom: 20px;
        }

        .form-group label {
            display: block;
            margin-bottom: 8px;
            font-weight: 600;
            color: #555;
        }

        .form-group input, .form-group textarea, .form-group select {
            width: 100%;
            padding: 12px;
            border: 2px solid #e1e5e9;
            border-radius: 8px;
            font-size: 14px;
            transition: border-color 0.3s ease;
        }

        .form-group input:focus, .form-group textarea:focus, .form-group select:focus {
            outline: none;
            border-color: #667eea;
        }

        .btn {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 12px 25px;
            border: none;
            border-radius: 8px;
            cursor: pointer;
            font-size: 14px;
            font-weight: 600;
            transition: all 0.3s ease;
            margin-right: 10px;
            margin-bottom: 10px;
        }

        .btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(0,0,0,0.2);
        }

        .btn-secondary {
            background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        }

        .btn-success {
            background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        }

        .result-area {
            background: #f8f9fa;
            border: 2px solid #e1e5e9;
            border-radius: 8px;
            padding: 15px;
            margin-top: 15px;
            min-height: 100px;
            white-space: pre-wrap;
            font-family: 'Courier New', monospace;
            font-size: 13px;
        }

        .image-result {
            max-width: 100%;
            border-radius: 8px;
            margin-top: 15px;
            box-shadow: 0 5px 15px rgba(0,0,0,0.1);
        }

        .loading {
            display: none;
            text-align: center;
            padding: 20px;
        }

        .spinner {
            border: 4px solid #f3f3f3;
            border-top: 4px solid #667eea;
            border-radius: 50%;
            width: 40px;
            height: 40px;
            animation: spin 1s linear infinite;
            margin: 0 auto 10px;
        }

        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }

        .status {
            padding: 10px;
            border-radius: 5px;
            margin: 10px 0;
            font-weight: 600;
        }

        .status.success {
            background: #d4edda;
            color: #155724;
            border: 1px solid #c3e6cb;
        }

        .status.error {
            background: #f8d7da;
            color: #721c24;
            border: 1px solid #f5c6cb;
        }

        .status.info {
            background: #d1ecf1;
            color: #0c5460;
            border: 1px solid #bee5eb;
        }

        .full-width {
            grid-column: 1 / -1;
        }

        @media (max-width: 768px) {
            .main-content {
                grid-template-columns: 1fr;
            }
            
            .header h1 {
                font-size: 2rem;
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Personal RAG API</h1>
            <p>Upload PDFs, ask questions, and generate AI images</p>
        </div>

        <div class="main-content">
            <!-- PDF Processing Card -->
            <div class="card">
                <h2>📄 PDF Processing</h2>
                <form id="pdfForm" enctype="multipart/form-data">
                    <div class="form-group">
                        <label for="pdfFile">Select PDF File:</label>
                        <input type="file" id="pdfFile" name="file" accept=".pdf" required>
                    </div>
                    <button type="submit" class="btn">Process PDF</button>
                </form>
                <div id="pdfResult" class="result-area"></div>
            </div>

            <!-- Question Answering Card -->
            <div class="card">
                <h2>Ask Something ❓</h2>
                <form id="questionForm">
                    <div class="form-group">
                        <label for="question">Ask a question about the PDF:</label>
                        <textarea id="question" name="question" rows="3" placeholder="Enter your question here..." required></textarea>
                    </div>
                    <div class="form-group">
                        <label for="useLlama3">Use Llama 3 (if available):</label>
                        <select id="useLlama3" name="use_llama3">
                            <option value="false">No (TF-IDF)</option>
                            <option value="true">Yes (Llama 3)</option>
                        </select>
                    </div>
                    <button type="submit" class="btn">Ask Question</button>
                </form>
                <div id="questionResult" class="result-area"></div>
            </div>

            <!-- Image Generation Card -->
            <div class="card">
                <h2>🎨 Image Generation</h2>
                <form id="imageForm">
                    <div class="form-group">
                        <label for="imagePrompt">Image Description:</label>
                        <textarea id="imagePrompt" name="prompt" rows="3" placeholder="Describe the image you want to generate..." required></textarea>
                    </div>
                    <div class="form-group">
                        <label for="useStableDiffusion">Use Stable Diffusion:</label>
                        <select id="useStableDiffusion" name="use_stable_diffusion">
                            <option value="false">No (Ollama)</option>
                            <option value="true">Yes (Stable Diffusion)</option>
                        </select>
                    </div>
                    <button type="submit" class="btn">Generate Image</button>
                </form>
                <div id="imageResult" class="result-area"></div>
                <img id="generatedImage" class="image-result" style="display: none;">
            </div>

            <!-- PDF-based Image Generation Card -->
            <div class="card">
                <h2>📄🎨 PDF-based Image Generation</h2>
                <form id="pdfImageForm">
                    <div class="form-group">
                        <label for="pdfImageQuestion">Question about PDF content:</label>
                        <textarea id="pdfImageQuestion" name="question" rows="3" placeholder="Ask about content to generate related image..." required></textarea>
                    </div>
                    <div class="form-group">
                        <label for="pdfImageUseSD">Use Stable Diffusion:</label>
                        <select id="pdfImageUseSD" name="use_stable_diffusion">
                            <option value="false">No (Ollama)</option>
                            <option value="true">Yes (Stable Diffusion)</option>
                        </select>
                    </div>
                    <button type="submit" class="btn">Generate PDF-based Image</button>
                </form>
                <div id="pdfImageResult" class="result-area"></div>
                <img id="pdfGeneratedImage" class="image-result" style="display: none;">
            </div>
        </div>

        <!-- Loading Spinner -->
        <div id="loading" class="loading">
            <div class="spinner"></div>
            <p>Processing...</p>
        </div>
    </div>

    <script>
        // Utility functions
        function showLoading() {
            document.getElementById('loading').style.display = 'block';
        }

        function hideLoading() {
            document.getElementById('loading').style.display = 'none';
        }

        function showStatus(elementId, message, type = 'info') {
            const element = document.getElementById(elementId);
            element.innerHTML = `<div class="status ${type}">${message}</div>`;
        }

        // PDF Processing
        document.getElementById('pdfForm').addEventListener('submit', async (e) => {
            e.preventDefault();
            showLoading();
            
            const formData = new FormData();
            const fileInput = document.getElementById('pdfFile');
            formData.append('file', fileInput.files[0]);

            try {
                const response = await fetch('/process-pdf', {
                    method: 'POST',
                    body: formData
                });
                
                const result = await response.json();
                
                if (response.ok) {
                    showStatus('pdfResult', `✅ PDF processed successfully!\\nOutput file: ${result.output_file}`, 'success');
                } else {
                    showStatus('pdfResult', `❌ Error: ${result.detail}`, 'error');
                }
            } catch (error) {
                showStatus('pdfResult', `❌ Error: ${error.message}`, 'error');
            } finally {
                hideLoading();
            }
        });

        // Question Answering
        document.getElementById('questionForm').addEventListener('submit', async (e) => {
            e.preventDefault();
            showLoading();
            
            const formData = new FormData();
            formData.append('question', document.getElementById('question').value);
            formData.append('use_llama3', document.getElementById('useLlama3').value);

            try {
                const response = await fetch('/ask-question', {
                    method: 'POST',
                    body: formData
                });
                
                const result = await response.json();
                
                if (response.ok) {
                    showStatus('questionResult', `✅ Answer:\\n${result.answer}`, 'success');
                } else {
                    showStatus('questionResult', `❌ Error: ${result.detail}`, 'error');
                }
            } catch (error) {
                showStatus('questionResult', `❌ Error: ${error.message}`, 'error');
            } finally {
                hideLoading();
            }
        });

        // Image Generation
        document.getElementById('imageForm').addEventListener('submit', async (e) => {
            e.preventDefault();
            showLoading();
            
            const formData = new FormData();
            formData.append('prompt', document.getElementById('imagePrompt').value);
            formData.append('use_stable_diffusion', document.getElementById('useStableDiffusion').value);

            try {
                const response = await fetch('/generate-image', {
                    method: 'POST',
                    body: formData
                });
                
                const result = await response.json();
                
                if (response.ok) {
                    showStatus('imageResult', `✅ ${result.status}`, 'success');
                    if (result.image_file) {
                        const img = document.getElementById('generatedImage');
                        img.src = `/download-image/${result.image_file}`;
                        img.style.display = 'block';
                    }
                } else {
                    showStatus('imageResult', `❌ Error: ${result.detail}`, 'error');
                }
            } catch (error) {
                showStatus('imageResult', `❌ Error: ${error.message}`, 'error');
            } finally {
                hideLoading();
            }
        });

        // PDF-based Image Generation
        document.getElementById('pdfImageForm').addEventListener('submit', async (e) => {
            e.preventDefault();
            showLoading();
            
            const formData = new FormData();
            formData.append('question', document.getElementById('pdfImageQuestion').value);
            formData.append('use_stable_diffusion', document.getElementById('pdfImageUseSD').value);

            try {
                const response = await fetch('/generate-pdf-image', {
                    method: 'POST',
                    body: formData
                });
                
                const result = await response.json();
                
                if (response.ok) {
                    showStatus('pdfImageResult', `✅ ${result.status}`, 'success');
                    if (result.image_file) {
                        const img = document.getElementById('pdfGeneratedImage');
                        img.src = `/download-image/${result.image_file}`;
                        img.style.display = 'block';
                    }
                } else {
                    showStatus('pdfImageResult', `❌ Error: ${result.detail}`, 'error');
                }
            } catch (error) {
                showStatus('pdfImageResult', `❌ Error: ${error.message}`, 'error');
            } finally {
                hideLoading();
            }
        });
    </script>
</body>
</html>
"""

# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.get("/", response_class=HTMLResponse)
async def read_root():
    """Serve the main web interface"""
    return HTMLResponse(content=HTML_TEMPLATE)

@app.post("/process-pdf")
async def process_pdf_endpoint(file: UploadFile = File(...)):
    """Process uploaded PDF file"""
    try:
        # Save uploaded file temporarily
        temp_file_path = UPLOADS_DIR / f"temp_{file.filename}"
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Create a file-like object for our backend
        class FileWrapper:
            def __init__(self, path):
                self.name = str(path)
        
        file_obj = FileWrapper(temp_file_path)
        
        # Process the PDF
        result = process_pdf(file_obj)
        
        # Clean up temp file
        if temp_file_path.exists():
            temp_file_path.unlink()
        
        return {"output_file": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ask-question")
async def ask_question_endpoint(
    question: str = Form(...),
    use_llama3: bool = Form(False)
):
    """Ask a question about the processed PDF"""
    try:
        answer = answer_question(question, use_llama3)
        return {"answer": answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/generate-image")
async def generate_image_endpoint(
    prompt: str = Form(...),
    use_stable_diffusion: bool = Form(False)
):
    """Generate an image from a prompt"""
    try:
        image_file, status = generate_image_wrapper(prompt, use_stable_diffusion)
        return {"image_file": image_file, "status": status}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/generate-pdf-image")
async def generate_pdf_image_endpoint(
    question: str = Form(...),
    use_stable_diffusion: bool = Form(False)
):
    """Generate an image based on PDF content and question"""
    try:
        image_file, status = generate_pdf_image_wrapper(question, use_stable_diffusion)
        return {"image_file": image_file, "status": status}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/download-image/{filename}")
async def download_image(filename: str):
    """Download a generated image file"""
    file_path = Path(filename)
    if file_path.exists():
        return FileResponse(file_path, media_type="image/png")
    else:
        raise HTTPException(status_code=404, detail="Image file not found")

# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("🚀 Starting FastAPI server...")
    print("📱 Web UI will be available at: http://localhost:8000")
    print("📚 API documentation at: http://localhost:8000/docs")
    print("🔧 Press Ctrl+C to stop the server")
    print("=" * 60)
    
    uvicorn.run(
        "fastapi_app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    ) 