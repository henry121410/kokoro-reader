# clipboard_handler.py - Handles clipboard operations and Ctrl+C simulation

import time
import pyperclip
from pynput import keyboard
import traceback  # Import traceback for potential error logging

# Initialize the keyboard controller once when the module is imported
try:
    key_controller = keyboard.Controller()
except Exception as e:
    print(f"FATAL: Failed to initialize keyboard controller: {e}")
    key_controller = None  # Set to None to indicate failure


def get_selected_text():
    """
    Attempts to copy selected text by simulating Ctrl+C and reading the clipboard.
    Includes retry logic and increased delays.

    Returns:
        str: The newly copied text if successful and different from original.
        None: If the copy fails after retries or if the keyboard controller failed init.
    """
    if key_controller is None:
        print("Error: Keyboard controller not initialized. Cannot simulate Ctrl+C.")
        return None

    newly_copied_text = None
    try:
        original_clipboard_content = pyperclip.paste()
    except Exception as e:
        # Handle potential errors during initial paste (e.g., clipboard format issues)
        print(f"Warning: Could not read initial clipboard content: {e}")
        original_clipboard_content = ""  # Assume empty if read fails

    try:
        # --- Added Delay Before Simulation --- #
        time.sleep(0.05)  # Small delay before starting simulation
        # ------------------------------------- #

        # --- Attempt 1 (with slightly increased internal delays) ---
        print("   Simulating Ctrl+C (Attempt 1 with delays)...")
        key_controller.press(keyboard.Key.ctrl)
        time.sleep(0.05)  # Delay after pressing Ctrl
        key_controller.press("c")
        time.sleep(0.05)  # Delay after pressing c
        key_controller.release("c")
        time.sleep(0.01)  # <<< Reduced delay between releasing 'c' and releasing Ctrl
        key_controller.release(keyboard.Key.ctrl)
        time.sleep(0.15)  # Main wait for clipboard update

        current_clipboard = pyperclip.paste()

        if current_clipboard and current_clipboard != original_clipboard_content:
            print("   Got text from clipboard on Attempt 1.")
            newly_copied_text = current_clipboard
        else:
            # --- Retry Logic (Add delay before re-read too) ---
            print("   Clipboard unchanged on Attempt 1. Retrying...")
            time.sleep(0.25)  # Increased retry wait from 0.20
            # Re-read clipboard after delay
            current_clipboard = pyperclip.paste()

            if current_clipboard and current_clipboard != original_clipboard_content:
                print("   Got text on Retry.")
                newly_copied_text = current_clipboard
            else:
                print("   Clipboard content still unchanged or empty after retry.")
                # newly_copied_text remains None

    except Exception as e:
        print(f"Error during clipboard simulation or reading: {type(e).__name__} - {e}")
        traceback.print_exc()  # Log full traceback for debugging
        newly_copied_text = None  # Ensure None on error

    return newly_copied_text
