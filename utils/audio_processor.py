import os
import shutil
import subprocess
from pathlib import Path

import yt_dlp
from pydub import AudioSegment


DOWNLOAD_DIR = Path("downloads")
DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)

BGUTIL_HOME = Path.home() / "bgutil-ytdlp-pot-provider"


def find_executable(name: str):
    return shutil.which(name)


def get_deno():
    """
    Deno is installed through the Python environment.
    Do not install it through apt/packages.txt.
    """

    deno = shutil.which("deno")

    if deno:
        return deno

    candidates = [
        "/home/adminuser/venv/bin/deno",
        "/home/oai/share/deno",
        "/usr/local/bin/deno",
        "/usr/bin/deno",
    ]

    for path in candidates:
        if os.path.exists(path):
            return path

    return None


def setup_environment():

    ffmpeg = find_executable("ffmpeg")

    if not ffmpeg:
        raise RuntimeError(
            "FFmpeg is not installed. "
            "Add ffmpeg to packages.txt."
        )

    print(f"FFmpeg: {ffmpeg}")

    ffprobe = find_executable("ffprobe")

    if ffprobe:
        print(f"FFprobe: {ffprobe}")

    deno = get_deno()

    if deno:

        print(f"Deno: {deno}")

        try:
            result = subprocess.run(
                [deno, "--version"],
                capture_output=True,
                text=True,
                timeout=10,
            )

            print(result.stdout)

        except Exception as e:
            print(f"Deno check failed: {e}")

    else:
        print(
            "WARNING: Deno was not found."
        )

    print(
        "yt-dlp version:",
        yt_dlp.version.__version__,
    )

    return deno


def setup_bgutil_script(deno_path):
    """
    Install the BgUtils POT generation script if it isn't present.

    The official bgutil provider supports a script mode where yt-dlp
    invokes the generator when a token is required.
    """

    server_dir = (
        BGUTIL_HOME / "server"
    )

    if server_dir.exists():
        return server_dir

    print(
        "Installing BgUtils PO-token provider..."
    )

    try:

        subprocess.run(
            [
                "git",
                "clone",
                "--depth",
                "1",
                "https://github.com/"
                "Brainicism/"
                "bgutil-ytdlp-pot-provider.git",
                str(BGUTIL_HOME),
            ],
            check=True,
            timeout=180,
        )

    except Exception as e:

        print(
            "Could not install BgUtils:",
            e,
        )

        return None

    # --------------------------------------------------------
    # The provider's script method requires its server
    # dependencies to be installed.
    # --------------------------------------------------------

    if deno_path:

        try:

            subprocess.run(
                [
                    deno_path,
                    "install",
                    "--allow-scripts=npm:canvas",
                    "--frozen",
                ],
                cwd=str(server_dir),
                check=True,
                timeout=300,
            )

            print(
                "BgUtils Deno dependencies installed."
            )

        except Exception as e:

            print(
                "BgUtils dependency setup failed:",
                e,
            )

    return server_dir


def get_ytdlp_options(
    deno_path,
    bgutil_server_dir=None,
):

    options = {

        # Let yt-dlp select the best available audio.
        "format": "bestaudio/best",

        "outtmpl": str(
            DOWNLOAD_DIR /
            "%(title)s.%(ext)s"
        ),

        "noplaylist": True,

        "retries": 5,
        "fragment_retries": 5,
        "extractor_retries": 3,

        "continuedl": True,

        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "wav",
                "preferredquality": "192",
            }
        ],

        "quiet": False,
        "no_warnings": False,

        # Needed for YouTube JS challenges.
        "remote_components": [
            "ejs:github"
        ],

        "verbose": True,
    }

    # --------------------------------------------------------
    # Deno
    # --------------------------------------------------------

    if deno_path:

        options["js_runtimes"] = {
            "deno": {
                "path": deno_path
            }
        }

    # --------------------------------------------------------
    # BgUtils script provider
    # --------------------------------------------------------

    if bgutil_server_dir:

        options["extractor_args"] = {
            "youtubepot-bgutilscript": {
                "server_home": str(
                    bgutil_server_dir
                )
            }
        }

    return options


def download_youtube_audio(
    url: str
) -> str:

    print(
        "\n========================================"
    )

    print(
        "YouTube download started"
    )

    print(
        "========================================"
    )

    deno_path = setup_environment()

    bgutil_dir = setup_bgutil_script(
        deno_path
    )

    ydl_opts = get_ytdlp_options(
        deno_path,
        bgutil_dir,
    )

    try:

        with yt_dlp.YoutubeDL(
            ydl_opts
        ) as ydl:

            info = ydl.extract_info(
                url,
                download=True,
            )

            prepared_path = Path(
                ydl.prepare_filename(info)
            )

    except yt_dlp.utils.DownloadError as e:

        message = str(e)

        print(
            "\n========== YOUTUBE ERROR =========="
        )

        print(message)

        print(
            "===================================\n"
        )

        if "403" in message:

            raise RuntimeError(
                "YouTube returned HTTP 403.\n\n"
                "The video was successfully identified, "
                "but YouTube rejected the media request.\n\n"
                "Check the verbose logs for:\n"
                "[pot] PO Token Providers\n\n"
                "If BgUtils is loaded and 403 remains, "
                "the Streamlit Cloud IP is likely being "
                "blocked by YouTube."
            ) from e

        raise RuntimeError(
            f"YouTube download failed:\n{message}"
        ) from e

    # --------------------------------------------------------
    # Locate WAV
    # --------------------------------------------------------

    candidates = [
        prepared_path.with_suffix(".wav"),
        Path(
            str(prepared_path)
            .replace(".webm", ".wav")
        ),
        Path(
            str(prepared_path)
            .replace(".m4a", ".wav")
        ),
        Path(
            str(prepared_path)
            .replace(".mp4", ".wav")
        ),
    ]

    for path in candidates:

        if path.exists():

            print(
                f"Downloaded WAV: {path}"
            )

            return str(path)

    # Fallback
    wav_files = list(
        DOWNLOAD_DIR.glob("*.wav")
    )

    if wav_files:

        latest = max(
            wav_files,
            key=lambda p: p.stat().st_mtime,
        )

        return str(latest)

    raise RuntimeError(
        "yt-dlp completed but no WAV file "
        "was produced."
    )


def convert_to_wav(
    input_path: str
) -> str:

    input_file = Path(input_path)

    if not input_file.exists():

        raise FileNotFoundError(
            f"File not found: {input_path}"
        )

    output_path = (
        input_file.parent
        / f"{input_file.stem}_converted.wav"
    )

    audio = AudioSegment.from_file(
        str(input_file)
    )

    audio = (
        audio
        .set_channels(1)
        .set_frame_rate(16000)
    )

    audio.export(
        str(output_path),
        format="wav",
    )

    return str(output_path)


def chunk_audio(
    wav_path: str,
    chunk_minutes: int = 10,
) -> list[str]:

    audio = AudioSegment.from_wav(
        wav_path
    )

    chunk_ms = (
        chunk_minutes *
        60 *
        1000
    )

    chunks = []

    for i, start in enumerate(
        range(
            0,
            len(audio),
            chunk_ms,
        )
    ):

        chunk = audio[
            start:start + chunk_ms
        ]

        chunk_path = (
            f"{wav_path}"
            f"_chunk_{i}.wav"
        )

        chunk.export(
            chunk_path,
            format="wav",
        )

        chunks.append(chunk_path)

    return chunks


def process_input(
    source: str
) -> list[str]:

    if not source:
        raise ValueError(
            "No input source provided."
        )

    source = source.strip()

    if (
        source.startswith("http://")
        or source.startswith("https://")
    ):

        wav_path = download_youtube_audio(
            source
        )

    else:

        print(
            "Detected local file."
        )

        wav_path = convert_to_wav(
            source
        )

    print(
        "Chunking audio..."
    )

    chunks = chunk_audio(
        wav_path
    )

    print(
        f"Audio ready — "
        f"{len(chunks)} chunk(s) created."
    )

    return chunks
