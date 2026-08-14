import os
import subprocess
import yt_dlp

DOWNLOAD_DIR = "downloades"
os.makedirs(DOWNLOAD_DIR, exist_ok=True)


def run_ffmpeg(input_path: str, output_path: str) -> str:
    """Convert audio/video to 16kHz mono WAV using FFmpeg."""

    command = [
        "ffmpeg",
        "-y",
        "-i",
        input_path,
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
            f"FFmpeg conversion failed:\n{result.stderr}"
        )

    return output_path


def download_youtube_audio(url: str) -> str:
    """Download YouTube audio and convert it to WAV."""


    import shutil

    print("FFmpeg:", shutil.which("ffmpeg"))
    print("FFprobe:", shutil.which("ffprobe"))
    print("Deno:", shutil.which("deno"))
    
    output_template = os.path.join(
        DOWNLOAD_DIR,
        "%(title)s.%(ext)s"
    )

    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": output_template,
        "remote_components": ["ejs:github"],
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "wav",
                "preferredquality": "192",
            }
        ],
        "quiet": True,
        "noplaylist": True,
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)

        downloaded_file = ydl.prepare_filename(info)

        base_path = os.path.splitext(downloaded_file)[0]
        wav_path = base_path + ".wav"

    if not os.path.exists(wav_path):
        raise FileNotFoundError(
            f"YouTube audio download failed: {wav_path}"
        )

    # Ensure Whisper receives 16kHz mono WAV
    converted_path = os.path.splitext(wav_path)[0] + "_16k.wav"

    return run_ffmpeg(wav_path, converted_path)


def convert_to_wav(input_path: str) -> str:
    """Convert any audio/video file to 16kHz mono WAV."""

    if not os.path.exists(input_path):
        raise FileNotFoundError(
            f"Input file not found: {input_path}"
        )

    output_path = (
        os.path.splitext(input_path)[0]
        + "_converted.wav"
    )

    return run_ffmpeg(input_path, output_path)


def get_audio_duration(wav_path: str) -> float:
    """Get audio duration in seconds using FFprobe."""

    command = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        wav_path,
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

    return float(result.stdout.strip())


def chunk_audio(
    wav_path: str,
    chunk_minutes: int = 10
) -> list:
    """Split WAV audio into fixed-length chunks using FFmpeg."""

    if not os.path.exists(wav_path):
        raise FileNotFoundError(
            f"WAV file not found: {wav_path}"
        )

    chunk_seconds = chunk_minutes * 60

    duration = get_audio_duration(wav_path)

    chunks = []

    start = 0
    index = 0

    while start < duration:

        chunk_path = (
            f"{os.path.splitext(wav_path)[0]}"
            f"_chunk_{index}.wav"
        )

        command = [
            "ffmpeg",
            "-y",
            "-i",
            wav_path,
            "-ss",
            str(start),
            "-t",
            str(chunk_seconds),
            "-ar",
            "16000",
            "-ac",
            "1",
            "-c:a",
            "pcm_s16le",
            chunk_path,
        ]

        result = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        if result.returncode != 0:
            raise RuntimeError(
                f"FFmpeg chunking failed:\n{result.stderr}"
            )

        chunks.append(chunk_path)

        start += chunk_seconds
        index += 1

    return chunks


def process_input(source: str) -> list:
    """Download/convert input and split it into audio chunks."""

    if source.startswith("http://") or source.startswith("https://"):
        print("Detected YouTube URL. Downloading audio...")
        wav_path = download_youtube_audio(source)

    else:
        print("Detected local file. Converting to WAV...")
        wav_path = convert_to_wav(source)

    print("Chunking audio...")

    chunks = chunk_audio(wav_path)

    print(
        f"Audio ready — {len(chunks)} chunk(s) created."
    )

    return chunks
