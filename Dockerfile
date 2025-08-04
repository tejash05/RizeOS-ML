# Use official Python image
FROM python:3.10-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Set working directory
WORKDIR /app

# Copy files
COPY . /app

# Install dependencies
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Download NLTK data (if needed)
RUN python -m nltk.downloader stopwords wordnet omw-1.4

# Expose the port
EXPOSE 6000

# Run the app
CMD ["python", "ml_api.py"]
