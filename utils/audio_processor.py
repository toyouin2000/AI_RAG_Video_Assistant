import os
import shutil
import subprocess

import yt_dlp


DOWNLOAD_DIR = "downloades"
os.makedirs(DOWNLOAD_DIR, exist_ok=True)


def get_deno_path() -> str:
    """
    Locate the Deno binary installed by the Python `deno` package.
    """

    # First try PATH
    deno_path = shutil.which("deno")

    if deno_path:
        return deno_path

    # Fall back to the Python Deno package
    try:
        import deno

        deno_path = deno.find_deno_bin()

        if deno_path and os.path.exists(deno_path):
            return deno_path

    except Exception as e:
        print(f"Could not locate Deno through Python package: {e}")

    raise RuntimeError(
        "Deno was not found. "
        "Make sure `deno==2.9.3` is installed."
    )


def run_ffmpeg(
    input_path: str,
    output_path: str,
    sample_rate: int = 16000,
    channels: int = 1,
) -> str:
    """
    Convert audio/video to WAV using the system FFmpeg binary.
    """

    ffmpeg_path = shutil.which("ffmpeg")

    if not ffmpeg_path:
        raise RuntimeError(
            "FFmpeg is not installed or not available in PATH."
        )

    command = [
        ffmpeg_path,
        "-y",
        "-i",
        input_path,
        "-vn",
        "-ac",
        str(channels),
        "-ar",
        str(sample_rate),
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
            "FFmpeg conversion failed:\n"
            + result.stderr
        )

    return output_path


def download_youtube_audio(url: str) -> str:
    """
    Download YouTube audio and convert it to WAV.

    Uses Deno + yt-dlp EJS support for YouTube's
    current JavaScript challenge requirements.
    """

    ffmpeg_path = shutil.which("ffmpeg")
    ffprobe_path = shutil.which("ffprobe")
    deno_path = get_deno_path()

    print("FFmpeg:", ffmpeg_path)
    print("FFprobe:", ffprobe_path)
    print("Deno:", deno_path)

    if not ffmpeg_path:
        raise RuntimeError("FFmpeg is not installed.")

    if not ffprobe_path:
        raise RuntimeError("FFprobe is not installed.")

    if not deno_path:
        raise RuntimeError("Deno is not available.")

    # Verify Deno
    result = subprocess.run(
        [deno_path, "--version"],
        capture_output=True,
        text=True,
        timeout=15,
    )

    if result.returncode != 0:
        raise RuntimeError(
            f"Deno execution failed:\n{result.stderr}"
        )

    print(result.stdout)

    output_template = os.path.join(
        DOWNLOAD_DIR,
        "%(title)s.%(ext)s",
    )

    ydl_opts = {
        # Prefer audio-only formats but allow fallback
        "format": (
            "bestaudio[ext=m4a]/"
            "bestaudio[ext=webm]/"
            "bestaudio/"
            "best"
        ),

        "outtmpl": output_template,

        # YouTube EJS
        "remote_components": [
            "ejs:github"
        ],

        # Explicit Deno
        "js_runtimes": {
            "deno": {
                "path": deno_path,
            }
        },

        # Try multiple times
        "retries": 5,
        "fragment_retries": 5,
        "file_access_retries": 5,

        # Don't use playlist
        "noplaylist": True,

        # Don't reuse stale downloaded files
        "overwrites": True,

        # FFmpeg conversion
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "wav",
                "preferredquality": "192",
            }
        ],

        "quiet": False,
        "no_warnings": False,
    }

    try:

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:

            info = ydl.extract_info(
                url,
                download=True,
            )

            downloaded_path = (
                ydl.prepare_filename(info)
            )

    except Exception as e:

        raise RuntimeError(
            f"YouTube download failed:\n{e}"
        ) from e

    base_path = os.path.splitext(
        downloaded_path
    )[0]

    wav_path = base_path + ".wav"

    if not os.path.exists(wav_path):

        possible_files = [
            base_path + ".webm",
            base_path + ".m4a",
            base_path + ".mp4",
            downloaded_path,
        ]

        source_file = None

        for candidate in possible_files:

            if os.path.exists(candidate):

                source_file = candidate
                break

        if source_file is None:

            raise FileNotFoundError(
                "YouTube audio was downloaded "
                "but the resulting file was not found."
            )

        run_ffmpeg(
            source_file,
            wav_path,
            sample_rate=16000,
            channels=1,
        )

    return wav_path
def convert_to_wav(input_path: str) -> str:
    """
    Convert any audio/video file to 16kHz mono WAV.
    """

    if not os.path.exists(input_path):
        raise FileNotFoundError(
            f"Input file not found: {input_path}"
        )

    output_path = (
        os.path.splitext(input_path)[0]
        + "_converted.wav"
    )

    return run_ffmpeg(
        input_path=input_path,
        output_path=output_path,
        sample_rate=16000,
        channels=1,
    )


def get_audio_duration(wav_path: str) -> float:
    """
    Get WAV duration using ffprobe.
    """

    ffprobe_path = shutil.which("ffprobe")

    if not ffprobe_path:
        raise RuntimeError(
            "FFprobe is not installed."
        )

    command = [
        ffprobe_path,
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
            "FFprobe failed:\n"
            + result.stderr
        )

    return float(result.stdout.strip())


def chunk_audio(
    wav_path: str,
    chunk_minutes: int = 10,
) -> list:
    """
    Split WAV into chunks using FFmpeg.

    Default:
    10-minute chunks.
    """

    if not os.path.exists(wav_path):
        raise FileNotFoundError(
            f"WAV file not found: {wav_path}"
        )

    duration = get_audio_duration(wav_path)

    chunk_seconds = chunk_minutes * 60

    chunks = []

    start = 0
    index = 0

    while start < duration:

        chunk_path = (
            f"{wav_path}_chunk_{index}.wav"
        )

        remaining = duration - start

        current_duration = min(
            chunk_seconds,
            remaining,
        )

        run_ffmpeg_segment(
            input_path=wav_path,
            output_path=chunk_path,
            start_seconds=start,
            duration_seconds=current_duration,
        )

        chunks.append(chunk_path)

        start += chunk_seconds
        index += 1

    return chunks


def run_ffmpeg_segment(
    input_path: str,
    output_path: str,
    start_seconds: float,
    duration_seconds: float,
) -> str:
    """
    Extract an audio segment using FFmpeg.
    """

    ffmpeg_path = shutil.which("ffmpeg")

    if not ffmpeg_path:
        raise RuntimeError(
            "FFmpeg is not installed."
        )

    command = [
        ffmpeg_path,
        "-y",
        "-ss",
        str(start_seconds),
        "-i",
        input_path,
        "-t",
        str(duration_seconds),
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
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
            "FFmpeg chunking failed:\n"
            + result.stderr
        )

    return output_path


def process_input(source: str) -> list:
    """
    Process either a YouTube URL or a local audio/video file.

    Returns:
        list[str]: WAV chunk paths
    """

    if not source:
        raise ValueError(
            "Input source cannot be empty."
        )

    if source.startswith(
        "http://"
    ) or source.startswith(
        "https://"
    ):

        print(
            "Detected YouTube URL. "
            "Downloading audio..."
        )

        wav_path = download_youtube_audio(
            source
        )

    else:

        print(
            "Detected local file. "
            "Converting to WAV..."
        )

        wav_path = convert_to_wav(
            source
        )

    print("Chunking audio...")

    chunks = chunk_audio(
        wav_path
    )

    print(
        f"Audio ready — "
        f"{len(chunks)} chunk(s) created."
    )

    return chunks
