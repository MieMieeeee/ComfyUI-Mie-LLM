"""General helpers shared by the core.

Deliberately free of numpy, opencv and torch: the product layer installs
this package to talk to a model, and it should not have to pull a deep
learning stack for that. The image helpers live in :mod:`mie_llm.media`,
which imports the heavy stack lazily.
"""

from __future__ import annotations

import datetime
import hashlib
import json
import math
from pathlib import Path

#: The mark the core puts in front of its log lines. The same sheep the
#: MieNodes pack uses, so a log line reads the same wherever it came from.
LOG_MARK = "🐑"

def mie_log(message):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    the_message = f"[{timestamp}] {LOG_MARK}: {message}"
    print(the_message)
    return the_message


def compute_hash(file_path, hash_algorithm):
    if hash_algorithm == "None":
        return None
    hash_func = getattr(hashlib, hash_algorithm)()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_func.update(chunk)
    return hash_func.hexdigest()


def convert_size(size_bytes):
    if size_bytes == 0:
        return "0B"
    size_name = ("B", "KB", "MB", "GB", "TB", "PB", "EB", "ZB", "YB")
    i = int(math.floor(math.log(size_bytes, 1024)))
    p = math.pow(1024, i)
    s = round(size_bytes / p, 2)
    return f"{s} {size_name[i]}"


def is_image_file(file_path):
    """
    Check if a file is a valid image using Pillow.

    Parameters:
    - file_path (str): Path to the file.

    Returns:
    - bool: True if the file is a valid image, False otherwise.
    """
    try:
        with Image.open(file_path) as img:
            return img.format is not None  # Returns True if the image format is valid
    except (IOError, FileNotFoundError):
        return False


def load_plugin_config(filename="mie_llm_keys.json"):
    p = Path(__file__).resolve().parents[1] / filename
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}


def resolve_token(api_token, default_key=None, config_file="mie_llm_keys.json", config_key=None, prefer_local=True):
    cfg = load_plugin_config(config_file or "mie_llm_keys.json")
    k = config_key or default_key
    cfg_token = (cfg.get(k) or "")
    api_token = (api_token or "")
    if prefer_local:
        return (cfg_token or api_token)
    return (api_token or cfg_token)


