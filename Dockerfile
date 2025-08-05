# ✅ Use slim Python image
FROM python:3.10-slim

# ✅ Environment variables for clean output
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# ✅ Set working directory
WORKDIR /app

# ✅ Copy all source code
COPY . /app

# ✅ Pre-download NLTK data (avoid runtime download)
RUN apt-get update && apt-get install -y wget \
  && python -m nltk.downloader stopwords wordnet omw-1.4 \
  && apt-get purge -y wget \
  && apt-get autoremove -y \
  && rm -rf /var/lib/apt/lists/*

# ✅ Install Python dependencies
RUN pip install --upgrade pip \
  && pip install --no-cache-dir -r requirements.txt

# ✅ Expose port Cloud Run expects
EXPOSE 8080

# ✅ Run Flask app
CMD ["python", "ml_api.py"]
