# Base Image
FROM python:3.9-slim

# Working Directory
WORKDIR /app

# Install Dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy Source Code
COPY . .

# Generate Data and Train Models (Ensure system is ready on startup)
# In production, we might pull pre-trained models, but for this self-contained demo, we build them.
RUN python scripts/generate_datasets.py && \
    python src/models/train.py

# Expose Streamlit Port
EXPOSE 8501

# Run Application
ENTRYPOINT ["streamlit", "run", "src/dashboard/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
