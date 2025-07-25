@echo off
cd /d "C:\Users\joswa\OneDrive\Desktop\Task\Personal RAG Api"
call .\venv-fastapi\Scripts\activate
pip install --upgrade pip
pip install -r requirement.txt
python ingest_pdfs.py
python fastapi_app.py
pause