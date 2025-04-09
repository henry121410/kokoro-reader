#!/usr/bin/env python
import time
import threading
import sys  # Keep sys for potential exit
import traceback
import torch
import gc
import argparse
import numpy as np
import sounddevice as sd
import pyperclip
from pynput import keyboard
import pystray  # type: ignore
from PIL import Image, ImageDraw  # For default icon
import os  # <<< Added import os

# Assuming kokoro is installed and accessible
from kokoro import KPipeline, KModel

import config  # Import the new config file
import clipboard_handler  # <<< Import the new module
import hotkey_listener  # <<< Import the new module
import audio_player  # <<< Import the new module

# --- Global Variables ---
# current_keys = set() # <<< Removed global current_keys
processing_lock = threading.Lock()  # Keep the lock here, pass it to listener
kokoro_pipeline = None
# keyboard_listener = None # <<< Listener object managed within hotkey_listener module now
tray_icon = None
# Initialize global state from config defaults
current_lang_code = config.DEFAULT_LANG_CODE
current_voice = config.DEFAULT_VOICE
current_speed = config.DEFAULT_SPEED


# --- Removed Constant Definitions ---
# DEFAULT_LANG_CODE, DEFAULT_VOICE, DEFAULT_SPEED
# AVAILABLE_VOICES, AVAILABLE_SPEEDS
# HOTKEY_MODIFIERS, HOTKEY_CHAR, HOTKEY_VK
# SAMPLE_RATE
# MAX_CHUNK_CHARS
# COPY_FAILED_MESSAGE
# --- Were moved to config.py ---


# --- Initialization (Modified to use repo_id and add CUDA checks) ---
def initialize_tts(lang_code=config.DEFAULT_LANG_CODE, voice_name=config.DEFAULT_VOICE):
    """Initialize or re-initialize the Kokoro TTS pipeline with more verbose CUDA checks."""
    global kokoro_pipeline
    print(
        f"Initializing Kokoro v1.1-zh pipeline for lang: {lang_code}, voice: {voice_name}..."
    )
    try:
        # --- Explicit CUDA Check --- #
        cuda_available = torch.cuda.is_available()
        print(f"   torch.cuda.is_available(): {cuda_available}")
        if cuda_available:
            try:
                gpu_name = torch.cuda.get_device_name(0)
                print(f"   CUDA Device Found: {gpu_name}")
            except Exception as e:
                print(f"   Warning: Could not get CUDA device name: {e}")
                cuda_available = False  # Treat as unavailable if device query fails
        # --------------------------- #

        device = "cuda" if cuda_available else "cpu"
        print(f"   Attempting to initialize pipeline on device: {device}")

        kokoro_pipeline = KPipeline(
            repo_id="hexgrad/Kokoro-82M-v1.1-zh", lang_code=lang_code, device=device
        )

        # --- Verify Actual Model Device (Best Effort) --- #
        # This depends on KPipeline internals, might not be straightforward
        # Let's check a parameter's device after loading voice as a proxy
        # -------------------------------------------------- #

        # --- Simplified Voice Validation --- #
        print(f"   Attempting to load voice: {voice_name}")
        voice_pack = kokoro_pipeline.load_single_voice(voice_name)
        print(
            f"   Voice {voice_name.split('/')[-1]} seems loaded (no immediate error)."
        )
        # --------------------------------- #

        # --- Check Model Parameter Device (Post Voice Load - More Robust) --- #
        model_verified_on_device = None
        model_attribute_name = None
        model_instance = None

        print("   Attempting to verify actual model device...")
        try:
            for attr_name in dir(kokoro_pipeline):
                if attr_name.startswith(
                    "_"
                ):  # Skip private/internal attributes typically
                    continue
                try:
                    attr_value = getattr(kokoro_pipeline, attr_name)
                    if isinstance(attr_value, torch.nn.Module):
                        # Found a potential model attribute
                        print(
                            f"      Found potential torch.nn.Module in attribute: '{attr_name}'"
                        )
                        # Check its parameters
                        try:
                            first_param = next(attr_value.parameters())
                            actual_device = first_param.device
                            print(
                                f"      Parameter device for '{attr_name}': {actual_device}"
                            )
                            model_verified_on_device = str(actual_device)
                            model_attribute_name = attr_name
                            model_instance = attr_value  # Store the model instance
                            break  # Assume the first one found is representative
                        except StopIteration:
                            print(f"      Module '{attr_name}' has no parameters.")
                        except Exception as param_e:
                            print(
                                f"      Error checking parameters for module '{attr_name}': {param_e}"
                            )
                except Exception as getattr_e:
                    # Some attributes might not be easily gettable
                    # print(f"      Could not getattr '{attr_name}': {getattr_e}")
                    pass

            if model_verified_on_device:
                print(
                    f"   Verification successful: Model ('{model_attribute_name}') detected on device: {model_verified_on_device}"
                )
                # --- Modify the mismatch check --- #
                # Consider 'cuda:0', 'cuda:1' etc. as matching a request for 'cuda'
                is_on_requested_device_type = (model_verified_on_device == device) or (
                    device == "cuda" and model_verified_on_device.startswith("cuda:")
                )
                # --------------------------------- #

                if not is_on_requested_device_type:
                    print(
                        f"   *** WARNING: Model device ({model_verified_on_device}) mismatch with requested device type ({device})! ***"
                    )
                    # --- Attempt to Force Move (only if mismatch AND requested cuda) --- #
                    if device == "cuda" and model_instance is not None:
                        print(
                            f"   >>> Attempting to forcefully move model '{model_attribute_name}' to {device}..."
                        )
                        try:
                            model_instance.to(
                                device
                            )  # PyTorch usually handles cuda:0 automatically if 'cuda' is specified
                            # Re-verify after move
                            first_param = next(model_instance.parameters())
                            new_device = str(first_param.device)
                            print(
                                f"   >>> Verification after move: Model '{model_attribute_name}' now on device: {new_device}"
                            )
                            if new_device.startswith(device):
                                model_verified_on_device = new_device  # Update status
                                print(f"   >>> Force move successful.")
                            else:
                                print(
                                    f"   >>> *** WARNING: Force move failed! Model still on {new_device}. ***"
                                )
                        except Exception as move_e:
                            print(
                                f"   >>> *** ERROR: Failed to force move model to {device}: {move_e} ***"
                            )
                    # ---------------------------------------------------------- #
            else:
                print(
                    "   Verification failed: Could not find a suitable torch.nn.Module with parameters to verify device."
                )

        except Exception as e:
            print(f"   Verification Error during attribute scan: {e}")
        # -------------------------------------------------------------- #

        if not kokoro_pipeline:
            print(
                "   Error: Kokoro pipeline object is invalid after voice load attempt."
            )
            return False

        print(
            f"   Performing initial run for {voice_name.split('/')[-1]}... (on {device})"
        )
        init_text = "你好" if lang_code == "z" else "Hello"
        if callable(kokoro_pipeline):
            _ = list(kokoro_pipeline(init_text, voice=voice_name))
        else:
            print("   Error: kokoro_pipeline is not callable for initial run.")
            return False
        print(
            f"Kokoro v1.1-zh pipeline ready for lang: {lang_code}, voice: {voice_name.split('/')[-1]} (on {device})."
        )
        return True
    except FileNotFoundError:
        print(f"   Error: Voice file not found at path: {voice_name}")
        kokoro_pipeline = None
        return False
    except Exception as e:
        print(
            f"Error initializing Kokoro v1.1-zh for {lang_code}/{voice_name}: {type(e).__name__} - {e}"
        )
        kokoro_pipeline = None
        return False


# --- Hotkey Processing (Modified to use audio_player) ---
def process_selected_text():
    """Handles the hotkey trigger, clipboard ops, and initiates TTS via audio_player."""
    if not processing_lock.acquire(blocking=False):
        print("Already processing. Ignoring concurrent hotkey trigger.")
        return

    print("\n--- New Hotkey Trigger Detected ---")
    audio_player.stop_all_playback()
    time.sleep(0.2)

    newly_copied_text = None
    try:
        print("   Attempting to get selected text via clipboard handler...")
        newly_copied_text = clipboard_handler.get_selected_text()
    except Exception as e:
        print(f"Error calling clipboard handler: {type(e).__name__} - {e}")
        traceback.print_exc()

    try:
        # --- Clear stop flag before potentially starting new playback --- #
        audio_player.clear_stop_flag()  # <<< Added call here
        # ------------------------------------------------------------- #

        if newly_copied_text:
            text_to_process = newly_copied_text
            print(
                f"   Processing newly copied text (Length: {len(text_to_process)}): {text_to_process[:60]}..."
            )
            seq_thread = threading.Thread(
                target=audio_player.speak_sequentially,
                args=(text_to_process, kokoro_pipeline, current_voice, current_speed),
                daemon=True,
            )
            seq_thread.start()
        else:
            last_text = audio_player.get_last_spoken_text()
            if last_text:
                print(f"   Repeating last completed segment: {last_text[:60]}...")
                replay_thread = threading.Thread(
                    target=audio_player.replay_last,
                    args=(kokoro_pipeline, current_voice, current_speed),
                    daemon=True,
                )
                replay_thread.start()
            else:
                text_to_process = config.COPY_FAILED_MESSAGE
                print("   Nothing spoken previously, speaking failure message.")
                error_thread = threading.Thread(
                    target=audio_player.speak_text,
                    args=(
                        text_to_process,
                        kokoro_pipeline,
                        current_voice,
                        current_speed,
                    ),
                    daemon=True,
                )
                error_thread.start()

    except Exception as e:
        print(f"Error in post-clipboard processing: {type(e).__name__} - {e}")
        traceback.print_exc()
    finally:
        processing_lock.release()
        collected_count = gc.collect()
        print("--- Hotkey processing finished ---")


# --- System Tray Functions ---
def set_voice(icon, item):
    """Callback function when a voice is selected from the menu."""
    global current_voice, current_lang_code, kokoro_pipeline  # Need lang code too

    selected_voice_id = item.text  # Get the voice ID (e.g., "zf_001") from item text
    new_voice_path = None

    # Find the path corresponding to the selected voice ID in the config
    if (
        current_lang_code in config.AVAILABLE_VOICES
        and "voices" in config.AVAILABLE_VOICES[current_lang_code]
        and selected_voice_id in config.AVAILABLE_VOICES[current_lang_code]["voices"]
    ):
        new_voice_path = config.AVAILABLE_VOICES[current_lang_code]["voices"][
            selected_voice_id
        ]
    else:
        print(
            f"Error: Could not find path for selected voice ID '{selected_voice_id}' in config."
        )
        return

    # Check if voice actually changed
    if current_voice != new_voice_path:
        print(
            f"Switching voice to [{current_lang_code}] {selected_voice_id} ({new_voice_path})"
        )

        # Re-initialize TTS in a background thread
        def reinit_task():
            if initialize_tts(
                current_lang_code, new_voice_path
            ):  # Use current_lang_code
                global current_voice  # Only update voice path on success
                current_voice = new_voice_path
                update_menu()
            else:
                print("Failed to switch voice. TTS might be unavailable.")
                # Optionally revert current_voice or show error state in menu

        threading.Thread(target=reinit_task, daemon=True).start()
    # else: # No change, do nothing
    #     pass


def set_speed(icon, item):
    """Callback function when a speed is selected."""
    global current_speed

    selected_speed_text = item.text  # Get the text like "1.0x (Default)"
    new_speed_value = None

    # Find the float value corresponding to the selected text in config
    for speed_val, speed_text in config.AVAILABLE_SPEEDS.items():
        if speed_text == selected_speed_text:
            new_speed_value = speed_val
            break

    if new_speed_value is None:
        print(
            f"Error: Could not find speed value for selected text '{selected_speed_text}' in config."
        )
        return

    # Check if speed actually changed
    # Use a small tolerance for float comparison
    if abs(current_speed - new_speed_value) > 0.01:
        print(f"Setting speed to {new_speed_value}x")
        current_speed = new_speed_value
        # No need to re-initialize TTS for speed changes
        update_menu()
    # else: # No change
    #     pass


def exit_action(icon, item):
    print("Exiting application...")
    hotkey_listener.stop_listener()
    audio_player.stop_all_playback()
    if tray_icon:
        tray_icon.stop()


# --- Test Function --- (Modified to handle interruption)
def test_length_limit(
    lang_code,
    voice_path,
    start_len=50,
    end_len=500,
    step=50,
    base_text="hello world",
    test_speed=1.0,
):
    """Runs a test using specific language/voice, handles interruption."""
    global stop_playback_flag  # Include flag
    # Need to re-initialize or ensure the pipeline matches the requested lang_code!
    # This is complex because the global pipeline might be for a different language.
    # For simplicity, let's ASSUME the global pipeline is already initialized for the requested lang_code.
    # A better implementation might temporarily re-initialize the pipeline here.
    if not kokoro_pipeline or kokoro_pipeline.lang_code != lang_code:
        print(
            f"ERROR: TTS Pipeline not ready or not initialized for the requested test language '{lang_code}'."
        )
        print(
            f"       Please switch voice to the desired language ('{lang_code}') first, then run the test."
        )
        # Or alternatively, re-initialize here temporarily (more complex state management)
        # print(f"Attempting temporary initialization for {lang_code}...")
        # temp_pipeline = KPipeline(...) # Needs careful handling
        return

    if not processing_lock.acquire(blocking=False):
        print(
            "ERROR: Another process (TTS or test) is running, cannot start length test."
        )
        return

    print("\n=== Starting TTS Length Limit Test ===")
    print(
        f"Testing Lang: '{lang_code}', Voice: {voice_path.split('/')[-1]}, Base Text: '{base_text}'"
    )
    print(
        f"Parameters: Start={start_len}, End={end_len}, Step={step}, Speed: {test_speed}"
    )
    print(
        "NOTE: This test calls the TTS engine directly, bypassing sentence splitting."
    )
    print("Watch for audio duration plateaus or errors indicating the limit.\n")

    results = []
    previous_duration_per_char = None
    limit_found_at = -1
    stop_playback_flag.clear()  # Ensure flag is clear at test start

    try:
        for length in range(start_len, end_len + 1, step):
            # --- Check stop flag at start of each length test ---
            if stop_playback_flag.is_set():
                print(f"--- Test interrupted by flag before length {length} ---")
                status = "Interrupted"
                break  # Exit the loop

            # Construct test text
            test_text = (base_text * (length // len(base_text) + 1))[:length]
            print(f"--- Testing length: {length} chars ---")
            start_time = time.time()
            generated_audio_segments = []
            status = "Success"
            error_msg = ""
            audio_duration = 0.0
            yielded_results_count = 0  # Track generator output

            try:
                # Directly call the pipeline using the specific voice path and speed
                generator = kokoro_pipeline(
                    test_text, voice=voice_path, speed=test_speed
                )
                print("   Generator created. Iterating...")
                for i, result in enumerate(generator):
                    yielded_results_count += 1
                    print(
                        f"      Generator yielded item {i} (type: {type(result)})"
                    )  # Log yield

                    # --- Corrected Audio Extraction --- #
                    audio_segment = None
                    if (
                        isinstance(result, KPipeline.Result)
                        and hasattr(result, "output")
                        and isinstance(result.output, KModel.Output)
                        and hasattr(result.output, "audio")
                    ):
                        # Access audio tensor from result.output.audio
                        audio_tensor = result.output.audio
                        if audio_tensor is not None and isinstance(
                            audio_tensor, torch.Tensor
                        ):
                            # Convert tensor to numpy array for concatenation/playback
                            # Ensure it's on CPU before converting to numpy
                            audio_segment = audio_tensor.detach().cpu().numpy()
                            print(
                                f"         Got audio segment (shape: {audio_segment.shape})"
                            )
                            generated_audio_segments.append(audio_segment)
                        else:
                            print(
                                "         Result contained None or invalid audio tensor."
                            )
                    else:
                        print(
                            f"         Warning: Unexpected result format or missing audio data: {result}"
                        )
                    # ---------------------------------- #

                print(
                    f"   Generator finished. Total items yielded: {yielded_results_count}"
                )

                if not generated_audio_segments:
                    # Check if generator yielded anything at all
                    if yielded_results_count == 0:
                        status = "Failed (No Yield)"
                        print("   Error: Generator yielded nothing.")
                    else:
                        status = "Failed (No Audio)"
                        print(
                            "   Error: No valid audio data extracted despite generator yielding."
                        )
                else:
                    full_audio = np.concatenate(generated_audio_segments)
                    audio_duration = len(full_audio) / config.SAMPLE_RATE

                    # --- ADDED PLAYBACK --- #
                    try:
                        print(
                            f"      Playing generated audio ({audio_duration:.3f}s)..."
                        )
                        # Ensure sounddevice is available and working
                        if sd is not None:
                            sd.play(full_audio, config.SAMPLE_RATE)
                            # Wait loop with stop check
                            playback_start_time = time.time()
                            while sd.get_stream().active:
                                if stop_playback_flag.is_set():
                                    print(
                                        "      Playback interrupted by flag during test."
                                    )
                                    sd.stop()
                                    status = "Interrupted"
                                    break  # Exit wait loop
                                time.sleep(0.05)
                            if (
                                status == "Interrupted"
                            ):  # Break outer loop if interrupted during playback
                                break
                            # Check flag again after natural finish
                            if stop_playback_flag.is_set():
                                print(
                                    "      Playback stopped by flag just after finishing."
                                )
                                status = "Interrupted"
                                break

                            if (
                                status != "Interrupted"
                            ):  # Only print finish if not interrupted
                                print("      Playback finished.")
                    except sd.PortAudioError as pae:
                        print(f"      Error during playback: {pae}")
                        status = "Failed (Playback Error)"
                        sd.stop()
                    except Exception as play_e:
                        print(f"      Unexpected error during playback: {play_e}")
                        status = "Failed (Playback Error)"
                        sd.stop()
                    # --------------------- #

            except RuntimeError as rte:
                status = "Failed (RuntimeError)"
                error_msg = str(rte)
                print(f"   RuntimeError encountered: {error_msg}")
            except Exception as e:
                status = f"Failed ({type(e).__name__})"
                error_msg = str(e)
                print(f"   Error encountered: {type(e).__name__} - {error_msg}")

            end_time = time.time()
            elapsed = end_time - start_time
            print(f"   Input Chars: {length}")
            print(f"   Output Duration: {audio_duration:.3f}s")
            print(f"   Time Elapsed: {elapsed:.3f}s")
            print(f"   Status: {status}")

            # --- Truncation Detection Logic --- #
            truncation_detected = False
            if (
                status == "Success" or status == "Truncated?"
            ) and audio_duration > 0.01:
                current_duration_per_char = audio_duration / length
                if previous_duration_per_char is not None:
                    if current_duration_per_char < previous_duration_per_char * 0.85:
                        print(
                            "   *** Truncation likely detected (duration/char decreased significantly) ***"
                        )
                        truncation_detected = True
                        status = "Truncated?"
                previous_duration_per_char = current_duration_per_char
            elif status.startswith("Failed"):
                previous_duration_per_char = None
            # ---------------------------------- #

            results.append(
                {
                    "length": length,
                    "duration": audio_duration,
                    "status": status,
                    "elapsed": elapsed,
                    "error": error_msg,
                }
            )

            if limit_found_at < 0 and (
                not status.startswith("Success") or truncation_detected
            ):
                limit_found_at = length

            time.sleep(0.1)  # Shorter sleep between lengths unless debugging

    finally:
        stop_playback_flag.clear()  # Ensure flag is clear on exit
        processing_lock.release()
        print("\n=== TTS Length Limit Test Finished ===")
        suggested_limit = "Not definitively found within range."
        if limit_found_at > 0:
            last_working_length = limit_found_at - step
            suggested_limit = f"~ {last_working_length} chars (issue detected at {limit_found_at} chars)"
        print(f"\nSuggested Approximate Limit: {suggested_limit}")
        print(
            f"(Results based on lang='{lang_code}', voice='{voice_path.split('/')[-1]}', base_text='{base_text}')"
        )
        print("====================================\n")


def create_menu():
    """Creates the dynamic system tray menu."""
    menu_items = []

    # --- Language Selection (Simplified - Shows current, no switch option) ---
    # We can add language switching back later if needed.
    current_lang_name = "Unknown Lang"
    if current_lang_code in config.AVAILABLE_VOICES:
        current_lang_name = config.AVAILABLE_VOICES[current_lang_code].get(
            "name", current_lang_code
        )
    menu_items.append(
        pystray.MenuItem(f"Language: {current_lang_name}", None, enabled=False)
    )
    menu_items.append(pystray.Menu.SEPARATOR)

    # --- Voice Submenu (Corrected for new config structure AND removed value arg) ---
    voice_menu_items = []
    if current_lang_code in config.AVAILABLE_VOICES:
        lang_data = config.AVAILABLE_VOICES[current_lang_code]
        if "voices" in lang_data and isinstance(lang_data["voices"], dict):
            if lang_data["voices"]:
                for voice_id, voice_path in lang_data["voices"].items():
                    voice_menu_items.append(
                        pystray.MenuItem(
                            voice_id,  # Text is the voice ID
                            set_voice,  # Callback function
                            checked=lambda item, path=voice_path: current_voice == path,
                            radio=True,
                            # value=voice_path # <<< Removed unsupported 'value' argument
                        )
                    )
            else:
                # voices dict exists but is empty
                voice_menu_items.append(
                    pystray.MenuItem("(No voices defined)", None, enabled=False)
                )
        else:
            # "voices" key is missing or not a dictionary
            voice_menu_items.append(
                pystray.MenuItem("(Voice config error)", None, enabled=False)
            )
    else:
        # current_lang_code not found in AVAILABLE_VOICES
        voice_menu_items.append(
            pystray.MenuItem("(Lang not configured)", None, enabled=False)
        )

    # Only add the submenu if there are items (prevents empty menu error maybe?)
    if voice_menu_items:
        menu_items.append(pystray.MenuItem("Voice", pystray.Menu(*voice_menu_items)))
    else:
        # Fallback if something went wrong
        menu_items.append(pystray.MenuItem("Voice: Error", None, enabled=False))
    # ---------------------------------------------------------- #

    # --- Speed Submenu (Corrected - removed value arg) ---
    speed_menu_items = []
    for speed_val, speed_text in config.AVAILABLE_SPEEDS.items():
        speed_menu_items.append(
            pystray.MenuItem(
                speed_text,  # Text is the display string (e.g., "1.0x (Default)")
                set_speed,  # Callback uses item.text to find value
                checked=lambda item, val=speed_val: current_speed == val,
                radio=True,
                # value=speed_val, # <<< Removed unsupported 'value' argument
            )
        )
    if speed_menu_items:
        menu_items.append(pystray.MenuItem("Speed", pystray.Menu(*speed_menu_items)))
    else:
        menu_items.append(pystray.MenuItem("Speed: Error", None, enabled=False))
    # ---------------------------------------------------- #

    # --- Replay Last (Modified to use audio_player) --- #
    last_text = audio_player.get_last_spoken_text()
    replay_menu_text = (
        f"Replay Last Completed ({len(last_text)} chars)"
        if last_text
        else "Replay Last (Nothing Completed)"
    )

    def replay_action_func(icon, item):
        audio_player.replay_last(kokoro_pipeline, current_voice, current_speed)

    menu_items.append(pystray.Menu.SEPARATOR)
    menu_items.append(
        pystray.MenuItem(replay_menu_text, replay_action_func, enabled=bool(last_text))
    )
    # -------------------------------------------------- #

    # --- Exit --- #
    menu_items.append(pystray.Menu.SEPARATOR)
    menu_items.append(pystray.MenuItem("Exit", exit_action))
    # ------------ #

    return pystray.Menu(*menu_items)


def update_menu():
    """Updates the tray icon's menu dynamically."""
    if tray_icon:
        tray_icon.menu = create_menu()


def setup_tray():
    """Sets up and runs the system tray icon."""
    global tray_icon
    try:
        # Attempt to load an icon file (replace with your actual icon)
        # Needs a 64x64 png usually works well
        icon_path = config.ICON_FILENAME
        if os.path.exists(icon_path):
            image = Image.open(icon_path)
            print(f"Loaded icon from {icon_path}")
        else:
            print(f"Icon file '{config.ICON_FILENAME}' not found. Using default icon.")
            # Create a simple default image if icon file not found
            width = 64
            height = 64
            color1 = "black"
            color2 = "white"
            image = Image.new("RGB", (width, height), color1)
            # You might want a more sophisticated default icon
    except Exception as e:
        print(f"Error loading icon: {e}. Using default icon.")
        image = Image.new("RGB", (64, 64), "black")  # Fallback default

    tray_icon = pystray.Icon(
        "kokoro_reader",
        icon=image,
        title="Kokoro Reader",
        menu=create_menu(),  # Dynamically create menu
    )
    print("System tray icon setup complete. Running...")
    # Run the icon loop (this blocks until exit_action stops it)
    # Needs to run in the main thread or a dedicated non-daemon thread
    tray_icon.run()


# --- Main Execution (Modified) ---
if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser(description="Kokoro Reader TTS Application")
    parser.add_argument(
        "-l",
        "--lang_code",
        type=str,
        default=config.DEFAULT_LANG_CODE,
        choices=config.AVAILABLE_VOICES.keys(),
        help=f"Language code to use (e.g., {list(config.AVAILABLE_VOICES.keys())}). Default: {config.DEFAULT_LANG_CODE}",
    )
    parser.add_argument(
        "-v",
        "--voice",
        type=str,
        default=config.DEFAULT_VOICE,
        help=f"Full path to the .pt voice file. Default: {config.DEFAULT_VOICE}",
    )
    parser.add_argument(
        "-s",
        "--speed",
        type=float,
        default=config.DEFAULT_SPEED,
        choices=config.AVAILABLE_SPEEDS.keys(),
        help=f"Playback speed (e.g., {list(config.AVAILABLE_SPEEDS.keys())}). Default: {config.DEFAULT_SPEED}",
    )
    args = parser.parse_args()

    # Update global state from potentially overridden args
    current_lang_code = args.lang_code
    current_voice = args.voice
    current_speed = args.speed

    print("Starting Kokoro Reader Application...")
    # Initialize TTS using current state
    init_success = initialize_tts(lang_code=current_lang_code, voice_name=current_voice)

    # --- Add GC call after successful initialization --- #
    if init_success:
        print("Initialization successful. Performing garbage collection...")
        gc_init_count = gc.collect()
        print(f"Garbage collection after init complete (collected: {gc_init_count}).")
    else:
        print("TTS Initialization failed. Exiting.")
        # Optionally, exit here or handle the failure gracefully
        # sys.exit(1) # Example: exit if initialization fails
    # ------------------------------------------------- #

    # Start the keyboard listener using the new module
    if kokoro_pipeline:
        # Pass only the main processing function to the listener module
        # The lock is now handled solely within process_selected_text
        hotkey_listener.start_listener(process_selected_text)

        # Setup and run the system tray icon in the main thread
        setup_tray()
    else:
        print("Exiting due to TTS initialization failure.")
