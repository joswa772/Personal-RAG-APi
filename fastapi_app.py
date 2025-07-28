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
import shutil
from pathlib import Path
import base64
from typing import Optional
from ai import query_llama
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
    <title>Conversational RAG Interface</title>
    <link rel="stylesheet" href="/static/style.css">
</head>
<body>
    <div class="chat-container">
        <div class="chat-header">Conversational RAG Interface</div>
        <div class="chat-box" id="chatBox"></div>
        <form class="chat-input-form" id="chatForm" action="#">
            <input type="file" id="pdfFile" accept=".pdf" onchange="handleFileSelect(this.files)">
            <button type="button" id="fileUploadButton" onclick="document.getElementById('pdfFile').click();">📄</button>
            <input type="text" id="userInput" placeholder="Ask a question or describe an image...">
            <button type="submit" class="btn">Send</button>
        </form>
    </div>
    <script src="/static/app.js" defer></script>
</body>
</html>
"""

# ============================================================================
# API ENDPOINTS
# ============================================================================
@app.post("/ask")
def ask_question(data: dict):
    prompt = data["prompt"]
    result = query_llama(prompt)
    return result

@app.post("/ask-chroma")
async def ask_chroma_endpoint(
    question: str = Form(...),
    use_llama3: bool = Form(False)
):
    try:
        from ai import answer_question_chroma
        answer = answer_question_chroma(question, use_llama3)
        return {"answer": answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

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
        # Only return the base filename
        from pathlib import Path
        if image_file:
            image_file = Path(image_file).name
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
        # Only return the base filename
        from pathlib import Path
        if image_file:
            image_file = Path(image_file).name
        return {"image_file": image_file, "status": status}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/generated_image/{filename}")
async def download_image(filename: str):
    from pathlib import Path
    filename = Path(filename).name
    file_path = Path("download-image/generated_images") / filename
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
