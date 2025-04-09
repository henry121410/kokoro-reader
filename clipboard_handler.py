# clipboard_handler.py - Handles clipboard operations and Ctrl+C simulation

import time
import pyperclip
from pynput import keyboard
import traceback

# import logging # Revert logging
# import config # Remove config import for this version

# Initialize the keyboard controller once when the module is imported
try:
    key_controller = keyboard.Controller()
except Exception as e:
    print(f"FATAL: Failed to initialize keyboard controller: {e}")
    key_controller = None


def get_selected_text():
    """
    Attempts to copy selected text by simulating Ctrl+C and reading the clipboard.
    Restored simpler logic without comparison or complex retry needing config.

    Returns:
        str: The newly copied text if successful.
        None: If the copy fails or if the keyboard controller failed init.
    """
    if key_controller is None:
        print(
            "Error: Keyboard controller not initialized. Cannot simulate Ctrl+C."
        )  # Use print
        return None

    newly_copied_text = None

    try:
        # Clear clipboard before attempting copy (keep this part)
        try:
            pyperclip.copy("")
            time.sleep(0.05)
        except Exception as clear_e:
            print(f"Warning: Failed to clear clipboard: {clear_e}")  # Use print

        # Delay Before Simulation
        time.sleep(0.05)

        # Simulate Ctrl+C (Original delays or similar)
        print("   Simulating Ctrl+C...")  # Use print
        key_controller.press(keyboard.Key.ctrl)
        time.sleep(0.05)
        key_controller.press("c")
        time.sleep(0.05)
        key_controller.release("c")
        time.sleep(0.01)
        key_controller.release(keyboard.Key.ctrl)
        print("   Ctrl+C simulation complete, waiting...")  # Use print

        # --- Simplified Read Logic --- #
        # Wait a fixed reasonable time and read once.
        # The previous successful state used 0.15s + 0.25s retry, let's try 0.2s total wait.
        time.sleep(0.20)  # Adjusted single wait time

        try:
            current_clipboard = pyperclip.paste()
            print(
                f"   Clipboard content after wait: '{str(current_clipboard)[:50]}...'"
            )  # Use print
            # If clipboard is not empty after clearing and copying, assume success
            if current_clipboard:
                print("   Got text from clipboard.")  # Use print
                newly_copied_text = current_clipboard
            else:
                print("   Clipboard remained empty after copy attempt.")  # Use print
        except Exception as read_e:
            print(f"   Error reading clipboard after copy: {read_e}")  # Use print
            # traceback.print_exc() # Optional traceback
        # --- End Simplified Read Logic --- #

    except Exception as e:
        print(f"Error during clipboard simulation/reading: {e}")  # Use print
        traceback.print_exc()
        newly_copied_text = None

    return newly_copied_text
