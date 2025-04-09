# hotkey_listener.py - Handles global hotkey listening using pynput

import threading
from pynput import keyboard
import config  # For hotkey definitions

# --- Module-level Variables ---
current_keys = set()
listener_thread = None
_stop_event = threading.Event()  # Internal stop signal for the listener thread
_hotkey_callback = None  # To store the function to call when hotkey is pressed


def _on_press(key):
    """Internal callback for key press events."""
    global current_keys, _hotkey_callback
    if _stop_event.is_set():  # Stop processing if listener is stopping
        return False

    try:
        # Normalize modifier keys
        normalized_key = key
        if key in (keyboard.Key.ctrl_l, keyboard.Key.ctrl_r):
            normalized_key = keyboard.Key.ctrl
        elif key in (keyboard.Key.shift_l, keyboard.Key.shift_r):
            normalized_key = keyboard.Key.shift
        elif key in (keyboard.Key.alt_l, keyboard.Key.alt_r, keyboard.Key.alt_gr):
            normalized_key = keyboard.Key.alt
        elif key in (keyboard.Key.cmd_l, keyboard.Key.cmd_r):
            normalized_key = keyboard.Key.cmd
        current_keys.add(normalized_key)
        # print(f"DEBUG: Key pressed: {key}, Current keys: {current_keys}") # Keep commented

        # Hotkey Check (using definitions from config.py)
        modifiers_held = config.HOTKEY_MODIFIERS.issubset(current_keys)
        key_char = getattr(key, "char", None)
        key_vk = getattr(key, "vk", None)
        is_target_key = key_char == config.HOTKEY_CHAR or key_vk == config.HOTKEY_VK
        other_modifiers_pressed = any(
            m in current_keys
            for m in (keyboard.Key.shift, keyboard.Key.alt, keyboard.Key.cmd)
            if m not in config.HOTKEY_MODIFIERS
        )

        if modifiers_held and is_target_key and not other_modifiers_pressed:
            if _hotkey_callback:
                print(
                    f"Hotkey triggered: {config.HOTKEY_MODIFIERS} + Char('{config.HOTKEY_CHAR}')/VK({config.HOTKEY_VK})"
                )
                # --- Execute the callback function directly in a new thread ---
                # No lock checking needed here anymore
                callback_thread = threading.Thread(target=_hotkey_callback, daemon=True)
                callback_thread.start()
                # ---------------------------------------------------------------
            else:
                print("Warning: Callback not set for hotkey listener.")

    except Exception as e:
        print(f"Error in hotkey listener _on_press: {e}")
        # Consider stopping the listener on error?
        # return False


def _on_release(key):
    """Internal callback for key release events."""
    global current_keys
    if _stop_event.is_set():
        return False

    try:
        # Normalize modifier keys
        normalized_key = key
        if key in (keyboard.Key.ctrl_l, keyboard.Key.ctrl_r):
            normalized_key = keyboard.Key.ctrl
        elif key in (keyboard.Key.shift_l, keyboard.Key.shift_r):
            normalized_key = keyboard.Key.shift
        elif key in (keyboard.Key.alt_l, keyboard.Key.alt_r, keyboard.Key.alt_gr):
            normalized_key = keyboard.Key.alt
        elif key in (keyboard.Key.cmd_l, keyboard.Key.cmd_r):
            normalized_key = keyboard.Key.cmd
        current_keys.discard(normalized_key)
        # print(f"DEBUG: Key released: {key}, Current keys: {current_keys}") # Keep commented
    except Exception as e:
        print(f"Error in hotkey listener _on_release: {e}")


def start_listener(callback_function):
    """
    Starts the global hotkey listener in a background thread.

    Args:
        callback_function: The function to call when the hotkey is detected.
    """
    global listener_thread, _hotkey_callback
    if listener_thread and listener_thread.is_alive():
        print("Listener already running.")
        return

    _hotkey_callback = callback_function
    _stop_event.clear()
    current_keys.clear()  # Clear keys on start

    print(
        f"Starting hotkey listener ({config.HOTKEY_MODIFIERS} + Char('{config.HOTKEY_CHAR}')/VK({config.HOTKEY_VK}))..."
    )

    # Define the listener target function to run in a thread
    def listener_run():
        try:
            with keyboard.Listener(
                on_press=_on_press, on_release=_on_release
            ) as k_listener:
                # Block until stop event is set or listener stops itself
                _stop_event.wait()
                k_listener.stop()  # Explicitly stop if event was set
        except Exception as e:
            print(f"FATAL: Keyboard listener failed: {e}")
        finally:
            print("Hotkey listener thread stopped.")

    listener_thread = threading.Thread(
        target=listener_run, daemon=True, name="HotkeyListenerThread"
    )
    listener_thread.start()

    modifier_names = " + ".join(str(k).split(".")[-1] for k in config.HOTKEY_MODIFIERS)
    print(
        f"Listener started. Press {modifier_names} + '{config.HOTKEY_CHAR}' key after selecting text."
    )


def stop_listener():
    """Signals the listener thread to stop."""
    global listener_thread
    if listener_thread and listener_thread.is_alive():
        print("Stopping hotkey listener...")
        _stop_event.set()
        # Optional: Join the thread if you need to wait for it to fully stop
        # listener_thread.join(timeout=1.0) # Wait max 1 second
    else:
        print("Listener not running.")
