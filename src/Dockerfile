FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install dependencies
COPY requirements.txt . 
RUN pip install --no-cache-dir -r requirements.txt

# Copy all source code
COPY . .

# Expose port for Render (Render sets $PORT automatically)
EXPOSE 10000  

# Start Gunicorn with Uvicorn worker for FastAPI or Flask
CMD ["gunicorn", "--bind", "0.0.0.0:10000", "app:app", "--worker-class", "uvicorn.workers.UvicornWorker"]
