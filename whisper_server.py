from fastapi import FastAPI, HTTPException, Request
import numpy as np
import whisper
import ffmpeg
import os

app = FastAPI(
    title="Whisper Speech-to-Text API",
    description="Upload audio files and get back transcribed text using OpenAI Whisper.",
    version="1.0.0"
)

# Load model once
# Default to "turbo" if not specified
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "turbo")
model = whisper.load_model(WHISPER_MODEL)

# Max upload size in MB (env configurable)
MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", 25))  # default = 25 MB


def load_audio(file_bytes: bytes, sr: int = 16_000) -> np.ndarray:
    """
    Use file's bytes and transform to mono waveform, resampling as necessary
    Parameters
    ----------
    file: bytes
        The bytes of the audio file
    sr: int
        The sample rate to resample the audio if necessary
    Returns
    -------
    A NumPy array containing the audio waveform, in float32 dtype.
    """
    try:
        # This launches a subprocess to decode audio while down-mixing and resampling as necessary.
        # Requires the ffmpeg CLI and `ffmpeg-python` package to be installed.
        out, _ = (
            ffmpeg.input('pipe:', threads=0)
            .output("pipe:", format="s16le", acodec="pcm_s16le", ac=1, ar=sr)
            .run_async(pipe_stdin=True, pipe_stdout=True)
        ).communicate(input=file_bytes)

    except ffmpeg.Error as e:
        raise RuntimeError(f"Failed to load audio: {e.stderr.decode()}") from e

    return np.frombuffer(out, np.int16).flatten().astype(np.float32) / 32768.0


@app.post("/transcribe")
async def transcribe(
    request: Request
):
    # Check file size
    content_length = request.headers.get("content-length")
    if content_length and int(content_length) > MAX_UPLOAD_MB * 1024 * 1024:
        raise HTTPException(
            status_code=413, detail=f"File too large. Max allowed is {MAX_UPLOAD_MB} MB.")

     # Read the file into memory as a file-like object
    audio_data = await request.body()

    # Load the audio using the helper function
    audio_np = load_audio(audio_data)

    # Directly use Whisper to process the NumPy array
    result = model.transcribe(audio_np)

    return {
        "transcription": result["text"].strip(),
        "language": result["language"]
    }
