# Use official Python base image
FROM python:3.10-slim

# Set the working directory inside the container
WORKDIR /code

# Copy all project files into the container
COPY . .

# Upgrade pip and install Python dependencies
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirement.txt

# Create directories for uploads and generated images
RUN mkdir -p /code/uploads /code/download-image/generated_images

# Expose the port FastAPI will run on
EXPOSE 8000

# Start the FastAPI app using uvicorn
CMD ["uvicorn", "fastapi_app:app", "--host", "0.0.0.0", "--port", "8000"] 