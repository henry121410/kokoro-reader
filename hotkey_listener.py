# hotkey_listener.py - Handles global hotkey listening using pynput

import threading
from pynput import keyboard
import config  # For hotkey definitions
import time
import logging

# --- Module-level Variables ---
current_keys = set()
listener_thread = None
_stop_event = threading.Event()  # Internal stop signal for the listener thread
_hotkey_callback = None  # To store the function to call when hotkey is pressed

# Global set to keep track of currently pressed modifier keys
pressed_modifiers = set()
# Global flag to indicate if the target key (VK) is currently pressed
is_target_key_pressed = False
# Lock for thread safety when accessing shared state
state_lock = threading.Lock()
# Flag to prevent immediate re-triggering
recently_triggered = False
cooldown_period = 0.5  # Cooldown in seconds


def _on_press(key):
    """Internal callback for key press events.
    Allows propagation for all keys now.
    """
    global current_keys, _hotkey_callback
    if _stop_event.is_set():
        return False  # Stop processing if listener is stopping

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

        # Hotkey Check
        modifiers_held = config.HOTKEY_MODIFIERS.issubset(current_keys)
        # key_char = getattr(key, "char", None) # No longer needed for checking
        key_vk = getattr(key, "vk", None)
        # Check ONLY based on VK code now
        is_target_key = key_vk is not None and key_vk == config.HOTKEY_VK
        other_modifiers_pressed = any(
            m in current_keys
            for m in (
                keyboard.Key.shift,
                keyboard.Key.alt,
                keyboard.Key.cmd,
            )  # Note: Check if this logic is still desired
            if m not in config.HOTKEY_MODIFIERS
        )

        # Use logging instead of print for internal messages if possible, or ensure prints are clearly marked as DEBUG/internal
        if modifiers_held and is_target_key and not other_modifiers_pressed:
            if _hotkey_callback:
                # Construct user-friendly hotkey representation for logging/printing
                try:
                    modifier_names = " + ".join(
                        str(k).split(".")[-1] for k in config.HOTKEY_MODIFIERS
                    )
                    try:
                        target_key_repr = f"'{keyboard.KeyCode(vk=config.HOTKEY_VK).char}' (VK={config.HOTKEY_VK})"
                    except AttributeError:
                        target_key_repr = f"VK={config.HOTKEY_VK}"
                    hotkey_repr = f"{modifier_names} + {target_key_repr}"
                except Exception:  # Fallback
                    hotkey_repr = (
                        f"Modifiers: {config.HOTKEY_MODIFIERS}, VK: {config.HOTKEY_VK}"
                    )

                logging.info(f"Hotkey triggered: {hotkey_repr}")  # Use logging
                # Execute the callback function directly in a new thread
                callback_thread = threading.Thread(target=_hotkey_callback, daemon=True)
                callback_thread.start()
                # --- REMOVED event suppression ---
                # print("   -> Suppressing original target key event.")
                # return False # <<< REMOVED: Allow propagation
                # --------------------------------- #
            else:
                logging.warning("Callback not set for hotkey listener.")  # Use logging
        # If it wasn't the target hotkey combination, allow propagation

    except Exception as e:
        logging.error(f"Error in hotkey listener _on_press: {e}")  # Use logging

    # Allow propagation by default
    return None


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

    return None  # Ensure release events propagate


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

    # Log the hotkey combination being listened for
    try:
        # Construct a user-friendly representation of the modifiers
        modifier_names = " + ".join(
            str(key).split(".")[-1] for key in config.HOTKEY_MODIFIERS
        )
        # Try to get a character representation for the VK code if possible, otherwise just show VK
        try:
            target_key_repr = f"'{keyboard.KeyCode(vk=config.HOTKEY_VK).char}' (VK={config.HOTKEY_VK})"
        except AttributeError:
            # Handle cases where VK might not map directly to a printable char or if .char is None
            target_key_repr = f"VK={config.HOTKEY_VK}"

        logging.info(
            f"Starting hotkey listener ({modifier_names} + {target_key_repr})..."
        )

    except Exception as e:
        logging.error(f"Error constructing hotkey log message: {e}")
        logging.info(
            f"Starting hotkey listener (Modifiers: {config.HOTKEY_MODIFIERS}, VK: {config.HOTKEY_VK})..."
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

    # Construct the user-friendly message for the console
    try:
        modifier_names = " + ".join(
            str(k).split(".")[-1].capitalize() for k in config.HOTKEY_MODIFIERS
        )  # Capitalize modifier names
        try:
            # Attempt to get char from VK, fallback to VK code
            target_key_obj = keyboard.KeyCode(vk=config.HOTKEY_VK)
            target_key_char = getattr(target_key_obj, "char", None)
            # Use a more descriptive name if char is None or ambiguous
            if target_key_char == "`":
                target_key_name = "Backtick"
            elif target_key_char:
                target_key_name = f"'{target_key_char}'"
            else:
                # Fallback if no char representation
                target_key_name = f"Key with VK {config.HOTKEY_VK}"
            target_key_repr = (
                f"{target_key_name} [VK={config.HOTKEY_VK}]"  # Use brackets
            )
        except Exception:
            # Broader fallback if KeyCode creation fails or anything else
            target_key_repr = f"[VK={config.HOTKEY_VK}]"
        print(
            f"Listener started. Press {modifier_names} + {target_key_repr} after selecting text."
        )
    except Exception as e:
        # Fallback print in case of error during message construction
        logging.error(f"Error constructing console start message: {e}")
        print(
            f"Listener started. Check logs for hotkey details (Modifiers: {config.HOTKEY_MODIFIERS}, VK: {config.HOTKEY_VK}). Press Esc to exit."
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


def _listener_thread(callback):
    """The actual listener loop running in its own thread."""
    with keyboard.Listener(
        on_press=_on_press_factory(callback), on_release=_on_release
    ) as listener:
        listener.join()


def _on_press_factory(callback):
    """Factory to create the on_press handler with the callback."""
    global recently_triggered

    def _on_press(key):
        """Handles key press events."""
        global pressed_modifiers, is_target_key_pressed, recently_triggered
        log_str = f"Key pressed: {key}"

        with state_lock:
            triggered = False
            is_modifier = key in config.HOTKEY_MODIFIERS
            is_target_vk = False
            try:
                # Check if the pressed key's VK code matches the target VK code
                if hasattr(key, "vk") and key.vk == config.HOTKEY_VK:
                    is_target_vk = True
                    is_target_key_pressed = True
                    log_str += f" (Target VK Match: {key.vk})"
            except AttributeError:
                # Some special keys (like modifiers) might not have vk
                pass

            if is_modifier:
                pressed_modifiers.add(key)
                log_str += " (Modifier)"

            # Check if all required modifiers are pressed *and* the target key is pressed
            # Also check the cooldown flag
            if (
                config.HOTKEY_MODIFIERS.issubset(pressed_modifiers)
                and is_target_key_pressed
                and not recently_triggered
            ):
                logging.info("Hotkey combination detected!")
                recently_triggered = True  # Set flag immediately
                triggered = True  # Mark that we triggered the callback
                # Reset target key state *after* checking, so release isn't needed to retrigger within combo
                # is_target_key_pressed = False # Don't reset here, let release handle it
                # Schedule cooldown reset
                threading.Timer(cooldown_period, _reset_cooldown).start()

            logging.debug(
                log_str
                + f" | Modifiers: {pressed_modifiers} | TargetPressed: {is_target_key_pressed} | RecentlyTriggered: {recently_triggered}"
            )

            if triggered:
                # Call the main callback (e.g., process_selected_text)
                callback()
                # Suppress the key event if the hotkey was triggered to prevent interference
                # This prevents 'q' from being typed if Ctrl+Alt+Q is the hotkey
                logging.debug("Hotkey triggered, suppressing key event.")
                return False  # Returning False suppresses the event

        # Allow the event to propagate if it wasn't the hotkey trigger
        return True

    return _on_press


def _on_release(key):
    """Handles key release events."""
    global pressed_modifiers, is_target_key_pressed
    log_str = f"Key released: {key}"
    with state_lock:
        # Check if the released key is the target VK
        try:
            if hasattr(key, "vk") and key.vk == config.HOTKEY_VK:
                is_target_key_pressed = False
                log_str += " (Target VK)"
        except AttributeError:
            pass

        # Remove modifier if it's released
        if key in pressed_modifiers:
            pressed_modifiers.discard(key)
            log_str += " (Modifier)"

        logging.debug(
            log_str
            + f" | Modifiers: {pressed_modifiers} | TargetPressed: {is_target_key_pressed}"
        )


def _reset_cooldown():
    """Resets the recently_triggered flag after the cooldown period."""
    global recently_triggered
    with state_lock:
        recently_triggered = False
        logging.debug("Hotkey cooldown finished.")


# Example Usage (if run directly, though typically imported):
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    def my_callback():
        print("--- Hotkey Activated! Processing... ---")
        time.sleep(1)  # Simulate work
        print("--- Processing Done! ---")

    print(
        "Starting hotkey listener test. Press Ctrl+Alt+Q to trigger. Press Esc to exit."
    )
    listener_thread = start_listener(my_callback)

    # Keep the main thread alive, or use listener.join() in a real app
    # For testing, let's just wait for Escape key
    def on_esc(key):
        if key == keyboard.Key.esc:
            print("Escape pressed, stopping listener...")
            return False  # Stop listener

    with keyboard.Listener(on_press=on_esc) as esc_listener:
        esc_listener.join()

    print("Listener stopped.")
