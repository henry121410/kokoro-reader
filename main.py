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
import os  # <<< Added import os

# Assuming kokoro is installed and accessible
from kokoro import KPipeline, KModel

import config  # Import the new config file
import clipboard_handler  # <<< Import the new module
import hotkey_listener  # <<< Import the new module
import audio_player  # <<< Import the new module
import tray_app  # <<< Import the new module

# --- Global Variables ---
# current_keys = set() # <<< Removed global current_keys
processing_lock = threading.Lock()  # Keep the lock here, pass it to listener
kokoro_pipeline = None
# keyboard_listener = None # <<< Listener object managed within hotkey_listener module now
# tray_icon = None # <<< Removed
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


# --- System Tray Functions (REMOVED) ---
# def set_voice(icon, item): ...
# def set_speed(icon, item): ...
# def exit_action(icon, item): ...
# def create_menu(): ...
# def update_menu(): ...
# def setup_tray(): ...
# --- Moved and adapted to tray_app.py ---


# --- Test Function (Needs update or removal) ---
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


# --- Main Application Logic Functions (Callbacks for tray_app) ---
def update_tray_menu():
    """Schedules an update for the tray menu (e.g., after state changes)."""
    tray_app.schedule_menu_update()


def handle_set_voice(new_voice_path):
    """Handles voice change requests from the tray menu."""
    global current_voice, current_lang_code, kokoro_pipeline

    if current_voice != new_voice_path:
        print(f"Tray requested switch voice to [{current_lang_code}] {new_voice_path}")

        # Re-initialize TTS in a background thread
        def reinit_task():
            if initialize_tts(current_lang_code, new_voice_path):
                global current_voice  # Only update global on success
                current_voice = new_voice_path
                update_tray_menu()  # Update menu to reflect change
            else:
                print("Failed to switch voice via tray. TTS might be unavailable.")

        threading.Thread(target=reinit_task, daemon=True).start()


def handle_set_speed(new_speed_value):
    """Handles speed change requests from the tray menu."""
    global current_speed
    if abs(current_speed - new_speed_value) > 0.01:
        print(f"Tray set speed to {new_speed_value}x")
        current_speed = new_speed_value
        update_tray_menu()


def handle_replay_last():
    """Handles replay request from the tray menu."""
    # Need access to pipeline, voice, speed
    if kokoro_pipeline:
        audio_player.replay_last(kokoro_pipeline, current_voice, current_speed)
    else:
        print("Cannot replay, TTS pipeline not available.")


def handle_exit():
    """Handles exit request from the tray menu."""
    print("Exit requested via tray menu...")
    hotkey_listener.stop_listener()
    audio_player.stop_all_playback()
    tray_app.stop_tray_app()  # Signal tray app to stop
    # Main thread should unblock after tray stops


# --- Main Execution (Modified with timing) ---
if __name__ == "__main__":
    start_time = time.time()
    print(f"[{start_time:.2f}] Starting application...")

    parser = argparse.ArgumentParser(description="Kokoro Reader TTS Application")
    parser.add_argument(
        "-l",
        "--lang_code",
        type=str,
        default=config.DEFAULT_LANG_CODE,
        choices=config.AVAILABLE_VOICES.keys(),
        help=f"Language code. Default: {config.DEFAULT_LANG_CODE}",
    )
    parser.add_argument(
        "-v",
        "--voice",
        type=str,
        default=config.DEFAULT_VOICE,
        help=f"Voice file path. Default: {config.DEFAULT_VOICE}",
    )
    parser.add_argument(
        "-s",
        "--speed",
        type=float,
        default=config.DEFAULT_SPEED,
        choices=config.AVAILABLE_SPEEDS.keys(),
        help=f"Playback speed. Default: {config.DEFAULT_SPEED}",
    )
    args = parser.parse_args()
    parse_time = time.time()
    print(f"[{parse_time:.2f}] Args parsed (took {parse_time - start_time:.2f}s)")

    current_lang_code = args.lang_code
    current_voice = args.voice
    current_speed = args.speed

    print("Starting Kokoro Reader Application...")  # Keep old log for context maybe
    init_start_time = time.time()
    init_success = initialize_tts(lang_code=current_lang_code, voice_name=current_voice)
    init_end_time = time.time()
    print(
        f"[{init_end_time:.2f}] TTS Initialization finished (took {init_end_time - init_start_time:.2f}s)"
    )

    if init_success:
        gc_start_time = time.time()
        print(f"[{gc_start_time:.2f}] Performing garbage collection...")
        gc_init_count = gc.collect()
        gc_end_time = time.time()
        print(
            f"[{gc_end_time:.2f}] Garbage collection complete (collected: {gc_init_count}, took {gc_end_time - gc_start_time:.2f}s)"
        )

        listener_start_time = time.time()
        print(f"[{listener_start_time:.2f}] Starting hotkey listener...")
        # Pass only the main processing function to the listener module
        hotkey_listener.start_listener(process_selected_text)
        listener_end_time = time.time()
        print(
            f"[{listener_end_time:.2f}] Hotkey listener thread started (took {listener_end_time - listener_start_time:.2f}s)"
        )

        tray_start_time = time.time()
        print(f"[{tray_start_time:.2f}] Starting Tray application...")
        # Define functions to pass state/actions to tray_app
        get_voice_state = lambda: current_voice
        get_speed_state = lambda: current_speed
        get_last_text_state = lambda: audio_player.get_last_spoken_text()

        try:
            # This call will block until the tray app is stopped
            tray_app.start_tray_app(
                get_current_voice_func=get_voice_state,
                get_current_speed_func=get_speed_state,
                get_last_spoken_text_func=get_last_text_state,
                set_voice_handler_func=handle_set_voice,
                set_speed_handler_func=handle_set_speed,
                replay_handler_func=handle_replay_last,
                exit_handler_func=handle_exit,  # Pass the exit handler
            )
            # Code here runs only after tray_app stops normally
            tray_end_time = time.time()
            print(
                f"[{tray_end_time:.2f}] Tray application stopped normally (ran for {tray_end_time - tray_start_time:.2f}s)"
            )

        except Exception as e:
            print(f"FATAL: Tray application failed to run: {e}")
            traceback.print_exc()
            # Attempt cleanup even on tray error
            handle_exit()
        # --------------------------------- #

        print("Main thread unblocked after tray exit.")
        # Final cleanup actions if needed after everything stops

    else:
        print("Exiting due to TTS initialization failure.")

    end_time = time.time()
    print(
        f"[{end_time:.2f}] Kokoro Reader application finished (Total time: {end_time - start_time:.2f}s)."
    )
