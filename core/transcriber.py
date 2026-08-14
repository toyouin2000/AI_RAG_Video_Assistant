import whisper
import os
import subprocess
import requests

# Sarvam's sync STT-translate API rejects audio longer than 30s.
# We slice each chunk into 25s pieces (with a 5s safety margin) before sending.
SARVAM_PIECE_SECONDS = 25

WHISPER_MODEL = os.getenv("WHISPER_MODEL", "small")

SARVAM_API_KEY = os.getenv("SARVAM_API_KEY")
SARVAM_STT_TRANSLATE_URL = "https://api.sarvam.ai/speech-to-text-translate"
SARVAM_MODEL = os.getenv("SARVAM_STT_MODEL", "saaras:v2.5")

_model = None


def load_model():
    global _model

    if _model is None:
        print(f"Loading Whisper model: {WHISPER_MODEL} ...")
        _model = whisper.load_model(WHISPER_MODEL)
        print("Whisper model loaded.")

    return _model


def transcribe_chunk_whisper(chunk_path: str) -> str:
    model = load_model()

    result = model.transcribe(
        chunk_path,
        task="transcribe"
    )

    return result["text"]


def _send_to_sarvam(piece_path: str) -> str:
    """Send one ≤30s WAV file to Sarvam and return the English transcript."""

    if not SARVAM_API_KEY:
        raise RuntimeError(
            "SARVAM_API_KEY is not set in environment / .env"
        )

    headers = {
        "api-subscription-key": SARVAM_API_KEY
    }

    with open(piece_path, "rb") as f:
        files = {
            "file": (
                os.path.basename(piece_path),
                f,
                "audio/wav"
            )
        }

        data = {
            "model": SARVAM_MODEL,
            "with_diarization": "false"
        }

        response = requests.post(
            SARVAM_STT_TRANSLATE_URL,
            headers=headers,
            files=files,
            data=data,
            timeout=120,
        )

    if not response.ok:
        print(f"\n❌ Sarvam returned {response.status_code}")
        print(f"Response body: {response.text}\n")
        response.raise_for_status()

    return response.json().get("transcript", "")


def get_audio_duration(audio_path: str) -> float:
    """Get audio duration in seconds using FFprobe."""

    command = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        audio_path,
    ]

    result = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    if result.returncode != 0:
        raise RuntimeError(
            f"FFprobe failed:\n{result.stderr}"
        )

    try:
        return float(result.stdout.strip())
    except ValueError:
        raise RuntimeError(
            f"Could not determine audio duration for {audio_path}"
        )


def create_audio_piece(
    input_path: str,
    output_path: str,
    start_seconds: float,
    duration_seconds: int
) -> str:
    """Extract a piece of audio using FFmpeg."""

    command = [
        "ffmpeg",
        "-y",
        "-ss",
        str(start_seconds),
        "-i",
        input_path,
        "-t",
        str(duration_seconds),
        "-ar",
        "16000",
        "-ac",
        "1",
        "-c:a",
        "pcm_s16le",
        output_path,
    ]

    result = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    if result.returncode != 0:
        raise RuntimeError(
            f"FFmpeg audio slicing failed:\n{result.stderr}"
        )

    return output_path


def transcribe_chunk_sarvam(chunk_path: str) -> str:
    """
    Sarvam sync API only accepts ≤30s audio.

    We split each chunk into 25-second pieces,
    send each separately, and join the transcripts.
    """

    if not SARVAM_API_KEY:
        raise RuntimeError(
            "SARVAM_API_KEY is not set in environment / .env"
        )

    duration = get_audio_duration(chunk_path)

    total_pieces = int(
        (duration + SARVAM_PIECE_SECONDS - 1)
        // SARVAM_PIECE_SECONDS
    )

    full_text = ""

    for i in range(total_pieces):

        start = i * SARVAM_PIECE_SECONDS

        piece_path = (
            f"{chunk_path}_sv_{i}.wav"
        )

        create_audio_piece(
            input_path=chunk_path,
            output_path=piece_path,
            start_seconds=start,
            duration_seconds=SARVAM_PIECE_SECONDS,
        )

        try:
            print(
                f"  → Sarvam piece "
                f"{i + 1}/{total_pieces} ..."
            )

            transcript = _send_to_sarvam(
                piece_path
            )

            full_text += transcript + " "

        finally:
            if os.path.exists(piece_path):
                os.remove(piece_path)

    return full_text.strip()


def transcribe_chunk(
    chunk_path: str,
    language: str = "english"
) -> str:
    """
    Route one chunk to Whisper or Sarvam depending
    on language choice.

    - english → Whisper (local model)
    - hinglish → Sarvam (translates to English while transcribing)
    """

    if language.lower() == "hinglish":
        return transcribe_chunk_sarvam(chunk_path)

    return transcribe_chunk_whisper(chunk_path)


def transcribe_all(
    chunks: list,
    language: str = "english"
) -> str:

    full_transcript = ""

    engine = (
        "Sarvam AI"
        if language.lower() == "hinglish"
        else "Whisper"
    )

    print(
        f"Using {engine} for transcription."
    )

    for i, chunk in enumerate(chunks):

        print(
            f"Transcribing chunk "
            f"{i + 1}/{len(chunks)}..."
        )

        text = transcribe_chunk(
            chunk,
            language=language
        )

        full_transcript += text + " "

    print("Transcription complete.")

    return full_transcript.strip()
