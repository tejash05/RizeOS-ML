# ✅ Base image
FROM python:3.10-slim

# ✅ Set env variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# ✅ Set working directory
WORKDIR /app

# ✅ Copy app code
COPY . /app

# ✅ Install system tools and Python dependencies
RUN apt-get update && apt-get install -y wget \
  && pip install --upgrade pip \
  && pip install --no-cache-dir -r requirements.txt

# ✅ Pre-download NLTK corpora (now that nltk is installed)
RUN python -m nltk.downloader stopwords wordnet omw-1.4 \
  && apt-get purge -y wget \
  && apt-get autoremove -y \
  && rm -rf /var/lib/apt/lists/*

# ✅ Expose port 8080 for Cloud Run
EXPOSE 8080

# ✅ Start Flask app
CMD ["python", "ml_api.py"]
