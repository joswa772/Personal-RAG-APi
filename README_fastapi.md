cd /d "C:\Users\joswa\OneDrive\Desktop\Task\Personal RAG Api"
.\venv-fastapi\Scripts\activate
python.exe -m pip install --upgrade pip
pip install -r requirement.txt
python ingest_pdfs.py
ollama run llama3
python fastapi_app.py

