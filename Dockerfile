FROM python:3.11-slim

WORKDIR /app

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p pipeline_outputs analysis_results logs

# Expose port
EXPOSE 8000

# Start command
CMD ["uvicorn", "main_4asset_backend_job_based:app", "--host", "0.0.0.0", "--port", "8000"]
