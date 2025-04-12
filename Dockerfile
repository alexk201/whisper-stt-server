FROM python:3.9-slim

# Install system dependencies
RUN apt-get update && apt-get install -y ffmpeg git && apt-get clean

# Install Python dependencies
RUN pip install --no-cache-dir numpy uvicorn fastapi ffmpeg-python git+https://github.com/openai/whisper.git 

RUN python -c "import whisper; whisper.load_model('turbo')"

# Copy your app
WORKDIR /app
COPY whisper_server.py whisper_server.py

# Expose the port
EXPOSE 8000

# Run the server
CMD ["uvicorn", "whisper_server:app", "--host", "0.0.0.0", "--port", "8000"]
