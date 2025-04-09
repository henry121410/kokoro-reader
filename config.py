# config.py - Configuration constants for Kokoro Reader

import logging
from pynput import keyboard
import os

# --- Shared State Flags (Use with caution!) ---
IS_SIMULATING_KEYS = (
    False  # Global flag to indicate if keys are being simulated programmatically
)

# --- Core Settings ---
DEBUG_MODE = True
APP_NAME = "Kokoro Reader"

# --- Default Settings ---
DEFAULT_LANG_CODE = "z"  # Default language ('z' for Chinese)
# !! 请根据你的实际路径修改下面的默认声音文件路径 !!
# DEFAULT_VOICE = "D:/kokoro-reader/Kokoro-82M/voice/zf_001.pt"  # Old absolute path
DEFAULT_VOICE = "./voice/zf_001.pt"  # Use relative path from project root
DEFAULT_SPEED = 1.0  # Default playback speed

# --- Available Options ---
# Structure: { lang_code: { "name": lang_display_name, "voices": { voice_id: voice_path_or_desc } } }
# 将来如果支持更多声音或语言，在这里扩展
AVAILABLE_VOICES = {
    "z": {
        "name": "Mandarin Chinese",
        "voices": {
            # 使用声音文件的基本名称作为显示名称的键可能更好管理
            # 或者直接用路径作为键，如果菜单能处理
            # 这里我们用回文件名基础作为键，路径作为值
            "zf_001": DEFAULT_VOICE,  # DEFAULT_VOICE 已经包含了完整路径
            # 如果你有其他中文声音文件，像这样添加
            # "zm_010": "D:/kokoro-reader/Kokoro-82M/voice/zm_010.pt",
        },
    },
    # "e": { # Example for English if you add voices later
    #     "name": "American English",
    #     "voices": {
    #         "af_maple": "path/to/af_maple.pt",
    #     }
    # }
}

AVAILABLE_SPEEDS = {
    # 实际速度值 : 显示文本
    0.8: "0.8x",
    0.9: "0.9x",
    1.0: "1.0x (Default)",
    1.1: "1.1x",
    1.2: "1.2x",
    1.3: "1.3x",
    1.4: "1.4x",
    1.5: "1.5x",
}

# --- Hotkey Configuration ---
# 使用 Ctrl + Backtick (`~) 键
HOTKEY_MODIFIERS = {keyboard.Key.ctrl}
HOTKEY_CHAR = r"\`"  # Target character is backtick (use raw string)
HOTKEY_VK = 192  # VK Code for Backtick/Tilde key

# --- Audio Settings ---
SAMPLE_RATE = 24000  # Audio sample rate in Hz (Kokoro default)

# --- Text Processing ---
MAX_CHUNK_CHARS = 180  # Maximum characters per chunk before forcing a split

# --- Messages ---
COPY_FAILED_MESSAGE = "Copy failed. Repeating last spoken text or check selection."

ICON_FILENAME = "icon.png"  # Or the actual name of your icon file
