# tray_app.py - Handles the system tray icon and menu using pystray

import pystray
from PIL import Image, ImageDraw
import os
import threading
import time

# Assuming config.py exists
import config

# --- Module-level Variables ---
_tray_icon = None
_update_pending = False

# --- Callback Functions (need external handlers) ---
# These will be set by start_tray_app
_get_current_voice = None
_get_current_speed = None
_get_last_spoken_text = None
_set_voice_handler = (
    None  # Function to call when voice changes (e.g., re-initialize TTS)
)
_set_speed_handler = None  # Function to call when speed changes
_replay_handler = None  # Function to call for replay last
_exit_handler = None  # Function to call to properly exit the app


# --- Menu Creation ---
def _create_menu():
    """Creates the dynamic system tray menu."""
    global _get_current_voice, _get_current_speed, _get_last_spoken_text
    global _set_voice_handler, _set_speed_handler, _replay_handler, _exit_handler

    menu_items = []

    # Language Display (Simplified)
    current_lang_code = config.DEFAULT_LANG_CODE  # Assuming only one lang for now
    current_lang_name = "Unknown Lang"
    if current_lang_code in config.AVAILABLE_VOICES:
        current_lang_name = config.AVAILABLE_VOICES[current_lang_code].get(
            "name", current_lang_code
        )
    menu_items.append(
        pystray.MenuItem(f"Language: {current_lang_name}", None, enabled=False)
    )
    menu_items.append(pystray.Menu.SEPARATOR)

    # Voice Submenu
    voice_menu_items = []
    if _get_current_voice and current_lang_code in config.AVAILABLE_VOICES:
        lang_data = config.AVAILABLE_VOICES[current_lang_code]
        if "voices" in lang_data and isinstance(lang_data["voices"], dict):
            if lang_data["voices"]:
                for voice_id, voice_path in lang_data["voices"].items():
                    # Define the action for this menu item
                    def voice_action_factory(path=voice_path):
                        def action(icon, item):
                            if _set_voice_handler:
                                _set_voice_handler(
                                    path
                                )  # Call the handler passed from main

                        return action

                    voice_menu_items.append(
                        pystray.MenuItem(
                            voice_id,
                            voice_action_factory(),  # Pass the generated action
                            checked=lambda item, path=voice_path: _get_current_voice()
                            == path,
                            radio=True,
                        )
                    )
            else:
                voice_menu_items.append(
                    pystray.MenuItem("(No voices defined)", None, enabled=False)
                )
        else:
            voice_menu_items.append(
                pystray.MenuItem("(Voice config error)", None, enabled=False)
            )
    else:
        voice_menu_items.append(
            pystray.MenuItem("(Lang not configured)", None, enabled=False)
        )

    if voice_menu_items:
        menu_items.append(pystray.MenuItem("Voice", pystray.Menu(*voice_menu_items)))
    else:
        menu_items.append(pystray.MenuItem("Voice: Error", None, enabled=False))

    # Speed Submenu
    speed_menu_items = []
    if _get_current_speed and _set_speed_handler:
        for speed_val, speed_text in config.AVAILABLE_SPEEDS.items():
            # Define the action for this menu item
            def speed_action_factory(val=speed_val):
                def action(icon, item):
                    if _set_speed_handler:
                        _set_speed_handler(val)  # Call the handler passed from main

                return action

            speed_menu_items.append(
                pystray.MenuItem(
                    speed_text,
                    speed_action_factory(),
                    checked=lambda item, val=speed_val: _get_current_speed() == val,
                    radio=True,
                )
            )
    if speed_menu_items:
        menu_items.append(pystray.MenuItem("Speed", pystray.Menu(*speed_menu_items)))
    else:
        menu_items.append(pystray.MenuItem("Speed: Error", None, enabled=False))

    # Replay Last
    if _get_last_spoken_text and _replay_handler:
        last_text = _get_last_spoken_text()
        replay_menu_text = (
            f"Replay Last Completed ({len(last_text)} chars)"
            if last_text
            else "Replay Last (Nothing Completed)"
        )

        def replay_action_func(icon, item):
            if _replay_handler:
                _replay_handler()  # Call the handler passed from main

        menu_items.append(pystray.Menu.SEPARATOR)
        menu_items.append(
            pystray.MenuItem(
                replay_menu_text, replay_action_func, enabled=bool(last_text)
            )
        )

    # Exit
    if _exit_handler:
        menu_items.append(pystray.Menu.SEPARATOR)
        menu_items.append(
            pystray.MenuItem("Exit", _exit_handler)
        )  # Call handler from main

    return pystray.Menu(*menu_items)


def _update_menu_thread():
    """Runs in a background thread to update the menu when requested."""
    global _tray_icon, _update_pending
    while True:
        time.sleep(0.2)  # Check periodically
        if _update_pending and _tray_icon and _tray_icon.visible:
            try:
                _tray_icon.menu = _create_menu()
                _tray_icon.update_menu()  # Important to refresh the visible menu
                _update_pending = False
                # print("DEBUG: Tray menu updated.") # Optional
            except Exception as e:
                print(f"Error updating tray menu: {e}")
                _update_pending = False  # Avoid loop spamming errors


def schedule_menu_update():
    """Schedules a menu update to be performed by the background thread."""
    global _update_pending
    _update_pending = True


# --- Tray Setup and Control ---
def start_tray_app(
    get_current_voice_func,
    get_current_speed_func,
    get_last_spoken_text_func,
    set_voice_handler_func,
    set_speed_handler_func,
    replay_handler_func,
    exit_handler_func,
):
    """
    Creates and runs the system tray icon application.

    Args:
        get_current_voice_func: Function that returns the current voice path.
        get_current_speed_func: Function that returns the current speed float.
        get_last_spoken_text_func: Function that returns the last spoken text.
        set_voice_handler_func: Function to call when a voice is selected (takes path).
        set_speed_handler_func: Function to call when a speed is selected (takes speed value).
        replay_handler_func: Function to call when Replay Last is selected.
        exit_handler_func: Function to call when Exit is selected.
    """
    global _tray_icon, _get_current_voice, _get_current_speed, _get_last_spoken_text
    global _set_voice_handler, _set_speed_handler, _replay_handler, _exit_handler

    # Store the passed handler functions
    _get_current_voice = get_current_voice_func
    _get_current_speed = get_current_speed_func
    _get_last_spoken_text = get_last_spoken_text_func
    _set_voice_handler = set_voice_handler_func
    _set_speed_handler = set_speed_handler_func
    _replay_handler = replay_handler_func
    _exit_handler = exit_handler_func

    # Attempt to load icon using config
    icon_path = config.ICON_FILENAME
    image = None
    try:
        print(f"Attempting to load icon from: {icon_path}")
        if os.path.exists(icon_path):
            image = Image.open(icon_path)
            print(f"Successfully loaded icon from {icon_path}")
        else:
            print(f"Icon file not found at path: {icon_path}. Using default icon.")
    except Exception as e:
        print(
            f"Error processing icon path '{icon_path}': {type(e).__name__} - {e}. Using default icon."
        )
        image = None

    # Create default icon if loading failed (image is None)
    if image is None:
        # Create a simple default image
        width = 64
        height = 64
        color1 = "navy"
        color2 = "skyblue"
        image = Image.new("RGB", (width, height), color1)
        dc = ImageDraw.Draw(image)
        dc.rectangle(
            (width // 4, height // 4, width * 3 // 4, height * 3 // 4), fill=color2
        )
        dc.text((10, 10), "K R", fill=color2)  # Add some text

    # Create the initial menu
    menu = _create_menu()

    # Create the icon instance
    _tray_icon = pystray.Icon("kokoro_reader", image, "Kokoro Reader", menu)

    # Start the background thread for menu updates
    update_thread = threading.Thread(
        target=_update_menu_thread, daemon=True, name="TrayUpdateThread"
    )
    update_thread.start()

    print("System tray icon setup complete. Running...")
    # Run the icon application (this blocks the main thread)
    _tray_icon.run()
    print("System tray icon stopped.")  # This prints after icon.stop() is called


def stop_tray_app():
    """Stops the system tray icon application."""
    global _tray_icon
    if _tray_icon:
        print("Stopping system tray icon...")
        _tray_icon.stop()
