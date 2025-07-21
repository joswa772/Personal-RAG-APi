# Use official Python base image
FROM python:3.10-slim

# Set the working directory inside the container
WORKDIR /code

# Copy all project files into the container
COPY . .

# Upgrade pip and install Python dependencies
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirement.txt

# Create a directory for uploaded files (optional, if your app saves files)
RUN mkdir -p /code/uploads

# Hugging Face Spaces expects the app to run on port 7860
EXPOSE 7860

# Start the FastAPI app using uvicorn
CMD ["uvicorn", "fastapi_app:app", "--host", "0.0.0.0", "--port", "7860"]
