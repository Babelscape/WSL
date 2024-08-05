from pathlib import Path

from wsl.inference.annotator import WSL

VERSION = {}  # type: ignore
with open(Path(__file__).parent / "version.py", "r") as version_file:
    exec(version_file.read(), VERSION)

__version__ = VERSION["VERSION"]
