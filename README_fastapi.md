# 🚀 PDF Processing & Image Generation - FastAPI Web Interface

A modern web application built with FastAPI that provides a beautiful UI for PDF processing and AI-powered image generation.

## ✨ Features

- **📄 PDF Processing**: Upload and extract text/images from PDF files
- **❓ Question Answering**: Ask questions about PDF content using TF-IDF or Llama 3
- **🎨 Image Generation**: Generate AI images using Ollama or Stable Diffusion
- **📄🎨 PDF-based Image Generation**: Create images based on PDF content
- **🧪 System Testing**: Built-in system diagnostics
- **🎨 Modern UI**: Beautiful, responsive web interface
- **🔌 REST API**: Full API endpoints for integration

## 🛠️ Installation

### 1. Install Dependencies

```bash
pip install -r requirements_fastapi.txt
```

### 2. Setup Requirements

Make sure you have:
- **Ollama** running locally (for Llama 3 and image generation)
- **Stable Diffusion API key** (optional, for better image generation)

### 3. Configure API Key

Edit `ai.py` and replace the placeholder API key:
```python
STABLE_DIFFUSION_API_KEY = "your-actual-api-key-here"
```

## 🚀 Running the Application

### Start the Server

```bash
python fastapi_app.py
```

### Access the Application

- **🌐 Web UI**: http://localhost:8000
- **📚 API Docs**: http://localhost:8000/docs
- **🔧 Interactive API**: http://localhost:8000/redoc

## 📱 Using the Web Interface

### 1. PDF Processing
1. Click "Choose File" and select a PDF
2. Click "Process PDF"
3. View the extracted text and image information

### 2. Question Answering
1. Process a PDF first
2. Enter your question in the text area
3. Choose between TF-IDF or Llama 3
4. Click "Ask Question" to get an answer

### 3. Image Generation
1. Enter an image description
2. Choose between Ollama or Stable Diffusion
3. Click "Generate Image" to create AI artwork

### 4. PDF-based Image Generation
1. Process a PDF first
2. Ask a question about the PDF content
3. The system will generate an image related to your question

### 5. System Tests
- Click "Run System Tests" to check all components
- Click "Cleanup Generated Files" to remove temporary files

## 🔌 API Endpoints

### PDF Processing
- `POST /process-pdf` - Upload and process PDF file

### Question Answering
- `POST /ask-question` - Ask questions about PDF content

### Image Generation
- `POST /generate-image` - Generate images from prompts
- `POST /generate-pdf-image` - Generate images based on PDF content
- `GET /download-image/{filename}` - Download generated images

### System Management
- `GET /system-tests` - Run system diagnostics
- `POST /cleanup-files` - Clean up generated files

## 🎨 UI Features

- **Responsive Design**: Works on desktop, tablet, and mobile
- **Modern Styling**: Beautiful gradient backgrounds and card-based layout
- **Real-time Feedback**: Loading indicators and status messages
- **Image Preview**: Generated images display directly in the interface
- **Error Handling**: Clear error messages and validation

## 🔧 Configuration

### Environment Variables
You can set these environment variables:

```bash
export STABLE_DIFFUSION_API_KEY="your-api-key"
export OLLAMA_HOST="http://localhost:11434"
```

### Customization
- Modify the HTML template in `fastapi_app.py` to customize the UI
- Adjust CSS styles for different themes
- Add new API endpoints as needed

## 🐛 Troubleshooting

### Common Issues

1. **Ollama Connection Error**
   - Make sure Ollama is running: `ollama serve`
   - Check if models are downloaded: `ollama list`

2. **Stable Diffusion API Error**
   - Verify your API key is correct
   - Check your account balance at Stability AI

3. **PDF Processing Error**
   - Ensure the PDF file is not corrupted
   - Check file permissions

4. **Port Already in Use**
   - Change the port in `fastapi_app.py`:
   ```python
   uvicorn.run("fastapi_app:app", host="0.0.0.0", port=8001)
   ```

### Debug Mode

Run with debug logging:
```bash
uvicorn fastapi_app:app --reload --log-level debug
```

## 📁 Project Structure

```
├── fastapi_app.py          # Main FastAPI application
├── ai.py                   # Backend functions
├── requirements_fastapi.txt # FastAPI dependencies
├── README_fastapi.md       # This file
├── uploads/                # Temporary upload directory
└── static/                 # Static files (if needed)
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📄 License

This project is open source and available under the MIT License.

## 🆘 Support

If you encounter any issues:
1. Check the troubleshooting section
2. Review the API documentation at `/docs`
3. Check the system tests for component status
4. Open an issue with detailed error information

---

**Happy PDF processing and image generation! 🎉** 