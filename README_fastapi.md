# 🚀 PDF Processing & Image Generation - FastAPI Web Interfac
.\venv-fastapi\Scripts\activate
pip install -r requirement.txt
python ingest_pdfs.py
python fastapi_app.py
deactivate
cloudflared tunnel --url http://localhost:11434


