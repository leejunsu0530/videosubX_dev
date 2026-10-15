import os
from shutil import which
from pathlib import Path

ffmpeg_path = which("ffmpeg")
if ffmpeg_path is None:
    raise RuntimeError("ffmpeg is not installed or not found in PATH.")
os.add_dll_directory(str(Path(ffmpeg_path).parent))
