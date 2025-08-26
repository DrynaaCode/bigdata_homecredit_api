FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y git-lfs && git lfs install && rm -rf /var/lib/apt/lists/*

# Set workdir
WORKDIR /app

# Copy project files
COPY . /app

# Install Python dependencies
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Expose port (change if needed)
EXPOSE 8000

# Command to run the API (adapt if needed)
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
