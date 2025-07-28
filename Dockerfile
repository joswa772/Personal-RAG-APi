# Use official Python base image
FROM python:3.10-slim

# Set the working directory inside the container
WORKDIR /code

# Copy only requirement.txt first for better Docker layer caching
COPY requirement.txt .

# Upgrade pip and install Python dependencies
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirement.txt --root-user-action=ignore

# Now copy the rest of the project files
COPY . .

# Create directories for uploads and generated images
RUN mkdir -p /code/uploads /code/download-image/generated_images

# Expose the port FastAPI will run on
EXPOSE 8000


RUN python -m nltk.downloader punkt


# Start the FastAPI app using uvicorn
CMD ["uvicorn", "fastapi_app:app", "--host", "0.0.0.0", "--port", "8000"]
