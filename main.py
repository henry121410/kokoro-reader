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
from kokoro.pipeline import KPipeline

# from kokoro.kmodel import KModel # Original line causing error
from kokoro import KModel  # Try importing directly from the main library

import config  # Import the new config file

# --- Global Variables ---
current_keys = set()
processing_lock = threading.Lock()
key_controller = keyboard.Controller()
kokoro_pipeline = None
keyboard_listener = None
tray_icon = None
# Initialize global state from config defaults
current_lang_code = config.DEFAULT_LANG_CODE
current_voice = config.DEFAULT_VOICE
current_speed = config.DEFAULT_SPEED
last_spoken_text = None
stop_playback_flag = threading.Event()


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


# --- Core TTS and Playback (Modified to handle interruption and update last_spoken_text correctly) ---
def speak_text(text):
    """Generate audio for a given text segment and play it. Handles interruption."""
    global last_spoken_text, stop_playback_flag  # Include stop_playback_flag

    if not kokoro_pipeline:
        print("TTS engine not initialized or failed. Skipping.")
        return False
    if not text or not text.strip():
        # print("Skipping empty text segment.")
        return True  # Consider empty text a success in the sequence

    is_error_message = text == config.COPY_FAILED_MESSAGE
    text_to_log = text[:50].replace("\n", " ") + ("..." if len(text) > 50 else "")

    if is_error_message:
        print(f"Speaking error message: {text}")
    else:
        print(
            f"Generating audio with [{current_lang_code}] {current_voice.split('/')[-1]} at {current_speed}x speed for: {text_to_log}"
        )
        # DO NOT update last_spoken_text here anymore

    # --- Check stop flag before generation ---
    if stop_playback_flag.is_set():
        print(
            f"   Playback interrupted before generation for: '{text_to_log}' - [speak_text]"
        )
        return False  # Indicate stopped

    try:
        all_audio_segments = []
        generator = kokoro_pipeline(text, voice=current_voice, speed=current_speed)
        yielded_results_count = 0

        for i, result in enumerate(generator):
            # --- Check stop flag during generation (if long generation) ---
            if stop_playback_flag.is_set():
                print(
                    f"   Playback interrupted during generation yield {i} for: '{text_to_log}' - [speak_text]"
                )
                # Ensure generator is properly closed if possible? Might not be necessary.
                return False  # Indicate stopped

            yielded_results_count += 1
            audio_segment = None
            if (
                isinstance(result, KPipeline.Result)
                and hasattr(result, "output")
                and isinstance(result.output, KModel.Output)
                and hasattr(result.output, "audio")
            ):
                audio_tensor = result.output.audio
                if audio_tensor is not None and isinstance(audio_tensor, torch.Tensor):
                    audio_segment = audio_tensor.detach().cpu().numpy()
                    # print(f"         Got audio segment (shape: {audio_segment.shape}) - [speak_text]") # Less verbose
                    all_audio_segments.append(audio_segment)
                # else: # Less verbose
                # print("         Result contained None or invalid audio tensor. - [speak_text]")
            # else: # Less verbose
            # print(f"         Warning: Unexpected result format or missing audio data: {result} - [speak_text]")

        # print(f"   Generator finished. Total items yielded: {yielded_results_count} - [speak_text]") # Less verbose

        if not all_audio_segments:
            if yielded_results_count == 0:
                print(
                    f"TTS generation yielded nothing for: '{text_to_log}' - [speak_text]"
                )
            else:
                print(
                    f"TTS generation produced no valid audio segments for: '{text_to_log}' - [speak_text]"
                )
            return False

        full_audio = np.concatenate(all_audio_segments)

        # --- Check stop flag before playback ---
        if stop_playback_flag.is_set():
            print(
                f"   Playback interrupted before playing: '{text_to_log}' - [speak_text]"
            )
            return False  # Indicate stopped

        print(
            f"   Playing audio ({len(full_audio)/config.SAMPLE_RATE:.2f}s) for: '{text_to_log}' - [speak_text]"
        )
        sd.play(full_audio, config.SAMPLE_RATE)

        # --- Wait loop with stop check ---
        playback_start_time = time.time()
        playback_duration = len(full_audio) / config.SAMPLE_RATE
        while sd.get_stream().active:  # Check if stream is active
            if stop_playback_flag.is_set():
                print(
                    f"   Playback interrupted during playback for: '{text_to_log}' - [speak_text]"
                )
                sd.stop()  # Actively stop sounddevice playback
                return False  # Indicate stopped
            time.sleep(0.05)  # Short sleep to avoid busy-waiting
            # Optional timeout check (can be added if streams hang)
            # if time.time() - playback_start_time > playback_duration + 10: # 10 sec buffer
            #     print(f"   WARN: Playback timeout suspected for: '{text_to_log}'")
            #     sd.stop()
            #     return False

        # --- Check flag *after* playback finishes naturally ---
        # This catches cases where the flag was set right as playback ended.
        if stop_playback_flag.is_set():
            print(
                f"   Playback stopped by flag just after finishing naturally: '{text_to_log}' - [speak_text]"
            )
            # sd.stop() # Stop again just in case? Likely redundant.
            return False  # Still consider it stopped

        print(f"   Playback finished for: '{text_to_log}' - [speak_text]")
        # Update last_spoken_text *only* on successful completion and if not error msg
        if not is_error_message:
            last_spoken_text = text
        return True  # Indicate success

    except sd.PortAudioError as pae:
        print(f"Error playing audio (SoundDevice PortAudioError): {pae} - [speak_text]")
        sd.stop()  # Attempt to stop sounddevice on error
        return False
    except RuntimeError as rte:
        print(
            f"Error during TTS generation (RuntimeError likely from model): {rte} - [speak_text]"
        )
        return False
    except Exception as e:
        print(
            f"Error during TTS or playback for '{text_to_log}': {type(e).__name__} - {e} - [speak_text]"
        )
        traceback.print_exc()  # Print stack trace for unexpected errors
        sd.stop()  # Attempt to stop sounddevice on generic error
        return False
    # No finally block needed unless specific cleanup is required


# --- Function for sequential speaking (Modified for interruption) ---
def speak_sentences_sequentially(full_text):
    """
    Speaks text, splitting it into chunks intelligently and handling interruption.
    """
    global stop_playback_flag  # Include flag

    if not kokoro_pipeline or not full_text:
        print("TTS Pipeline not ready or text is empty.")
        return

    print(
        f"Processing text intelligently (Length: {len(full_text)}): {full_text[:50]}... - [seq]"
    )

    current_pos = 0
    delimiters = "。！？….?!"
    segment_index = 0

    while current_pos < len(full_text):
        # --- Check stop flag at start of loop ---
        if stop_playback_flag.is_set():
            print("   Sequence interrupted by flag at start of loop. - [seq]")
            break  # Exit the loop

        segment_index += 1
        # Determine the end boundary for the potential chunk
        end_boundary = min(current_pos + config.MAX_CHUNK_CHARS, len(full_text))
        potential_chunk = full_text[current_pos:end_boundary]

        actual_chunk = ""
        next_pos = end_boundary  # Default next position

        # If the potential chunk IS the rest of the text AND its length is <= MAX_CHUNK_CHARS
        is_last_part = end_boundary == len(full_text)
        if is_last_part and len(potential_chunk) <= config.MAX_CHUNK_CHARS:
            actual_chunk = potential_chunk
            next_pos = len(full_text)  # Ensure loop terminates
            print(
                f"   Segment {segment_index} (Final Short): Taking remaining {len(actual_chunk)} chars."
            )
        else:
            # Need to split or take the max chunk. Find the last suitable delimiter.
            best_split_index = -1  # Relative to the start of the *original* text
            # Search backwards within the potential chunk for the last delimiter
            for i in range(len(potential_chunk) - 1, -1, -1):
                if potential_chunk[i] in delimiters:
                    # Store the index relative to the start of full_text
                    best_split_index = current_pos + i
                    break

            # Check if a valid split point was found WITHIN the current search range
            # (Must be after the current position) - this check might be redundant
            # because we search backwards from end_boundary, but safe to keep.
            if best_split_index >= current_pos:
                # Found a delimiter, split here (include the delimiter)
                chunk_end_pos = best_split_index + 1
                actual_chunk = full_text[current_pos:chunk_end_pos]
                next_pos = chunk_end_pos
                print(
                    f"   Segment {segment_index}: Splitting at delimiter, chunk length {len(actual_chunk)}."
                )
            else:
                # No suitable delimiter found within the potential chunk.
                # Take the whole potential chunk (up to MAX_CHUNK_CHARS).
                actual_chunk = potential_chunk
                next_pos = end_boundary  # Move position to the end of this hard chunk
                print(
                    f"   Segment {segment_index}: No delimiter found, taking max chunk up to {len(actual_chunk)} chars."
                )

        # --- Speak the determined chunk --- #
        chunk_to_speak = actual_chunk.strip()
        if chunk_to_speak:
            # --- Check stop flag before speaking chunk ---
            if stop_playback_flag.is_set():
                print("   Sequence interrupted before speaking next chunk. - [seq]")
                break  # Exit the loop

            print(
                f"      Speaking chunk {segment_index}: '{chunk_to_speak[:40].replace(chr(10), ' ')}...' - [seq]"
            )
            success = speak_text(
                chunk_to_speak
            )  # speak_text now handles its own stop checks
            if not success:
                # If speak_text returned False, it could be an error OR an interruption.
                # Check the flag again to be sure why it failed.
                if stop_playback_flag.is_set():
                    print(
                        f"      Segment {segment_index} interrupted by flag. Stopping sequence. - [seq]"
                    )
                else:
                    # Logged already in speak_text if it was a generation/playback error
                    print(
                        f"      Failed to speak segment {segment_index} (error or interruption). Stopping sequence. - [seq]"
                    )
                break  # Stop processing further chunks on failure or interruption
            # Optional pause after speaking a chunk
            # time.sleep(0.05)
        else:
            print(f"   Segment {segment_index}: Skipped empty chunk.")

        current_pos = next_pos  # Move to the next position

    # --- Log sequence completion status ---
    if stop_playback_flag.is_set():
        print("   Sequence processing stopped due to interruption flag. - [seq]")
    elif current_pos >= len(full_text):
        print("   Finished processing all text successfully. - [seq]")
    else:
        print(
            "   Sequence processing stopped prematurely (likely error in last chunk). - [seq]"
        )

    # Clear the flag? Let process_selected_text handle it for safety.


# --- Hotkey Processing (Modified for interruption and refined logic) ---
def process_selected_text():
    """Handles the hotkey trigger, clipboard operations, and initiates TTS, including interruption."""
    global stop_playback_flag, last_spoken_text  # Include flag and last text

    if not processing_lock.acquire(blocking=False):
        print("Already processing. Ignoring concurrent hotkey trigger.")
        return

    print("\n--- New Hotkey Trigger Detected ---")  # Add newline for clarity
    # 1. Signal any ongoing playback to stop
    print("   Signalling previous playback to stop...")
    stop_playback_flag.set()
    # Increase delay slightly to allow system/audio to potentially settle after stop signal
    time.sleep(0.15)  # Increased from 0.08 to 0.15

    original_clipboard_content = pyperclip.paste()
    text_to_process = None
    newly_copied_text = None

    try:
        # 2. Attempt copy (Clear flag *before* simulation)
        stop_playback_flag.clear()  # Prepare for potential new playback
        print("   Simulating Ctrl+C (Attempt 1 with delays)...")
        key_controller.press(keyboard.Key.ctrl)
        time.sleep(0.03)
        key_controller.press("c")
        time.sleep(0.03)
        key_controller.release("c")
        key_controller.release(keyboard.Key.ctrl)
        time.sleep(0.1)  # Main wait for clipboard update

        current_clipboard = pyperclip.paste()

        if current_clipboard and current_clipboard != original_clipboard_content:
            print("   Got text from clipboard on Attempt 1.")
            newly_copied_text = current_clipboard
        else:
            # Retry Logic
            print("   Clipboard unchanged on Attempt 1. Retrying...")
            time.sleep(0.20)  # Wait longer before retry read
            # Optional: Re-simulate Ctrl+C here if first read fails often
            current_clipboard = pyperclip.paste()  # Read again

            if current_clipboard and current_clipboard != original_clipboard_content:
                print("   Got text on Retry.")
                newly_copied_text = current_clipboard
            else:
                print("   Clipboard content still unchanged or empty after retry.")
                # Fallback logic uses last_spoken_text (last *completed* segment)

        # 3. Decide what to speak and launch thread
        # --- Ensure flag is clear before starting new thread ---
        stop_playback_flag.clear()

        if newly_copied_text:
            text_to_process = newly_copied_text
            print(
                f"   Processing newly copied text (Length: {len(text_to_process)}): {text_to_process[:60]}..."
            )
            # Start sequential speaking in a thread
            seq_thread = threading.Thread(
                target=speak_sentences_sequentially,
                args=(text_to_process,),
                daemon=True,
            )
            seq_thread.start()
        else:  # Copy failed or yielded nothing new
            if last_spoken_text:
                # Repeat last *completed* segment
                text_to_process = last_spoken_text
                print(f"   Repeating last completed segment: {text_to_process[:60]}...")
                # Speak directly as it's a single segment
                tts_thread = threading.Thread(
                    target=speak_text, args=(text_to_process,), daemon=True
                )
                tts_thread.start()
            else:
                # Speak failure message
                text_to_process = config.COPY_FAILED_MESSAGE
                print("   Nothing spoken previously, speaking failure message.")
                tts_thread = threading.Thread(
                    target=speak_text, args=(text_to_process,), daemon=True
                )
                tts_thread.start()

    except Exception as e:
        print(f"Error in hotkey processing: {type(e).__name__} - {e}")
        traceback.print_exc()  # Print full traceback for debugging
    finally:
        # Ensure flag is cleared in case of exception before starting thread
        stop_playback_flag.clear()
        processing_lock.release()
        # Attempt explicit garbage collection
        collected_count = gc.collect()
        # print(f"--- Hotkey processing finished (GC collected: {collected_count}) ---") # Optional: Log GC count
        print("--- Hotkey processing finished ---")


# --- Hotkey Listener Callbacks ---
def on_press(key):
    """Handles key press events for hotkey detection."""
    global current_keys
    try:
        # Normalize modifier keys for consistent tracking
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
        # print(f"DEBUG: Key pressed: {key}, Current keys: {current_keys}") # Commented out

        # --- Robust Hotkey Check using Modifiers, Character, OR VK Code --- #
        # 1. Check if all required modifiers are currently held
        modifiers_held = config.HOTKEY_MODIFIERS.issubset(current_keys)

        # 2. Check if the key JUST pressed matches either the target character OR the target VK code
        key_char = getattr(key, "char", None)
        key_vk = getattr(key, "vk", None)
        is_target_key = key_char == config.HOTKEY_CHAR or key_vk == config.HOTKEY_VK

        # 3. Ensure no *other* modifiers are accidentally held
        other_modifiers_pressed = any(
            m in current_keys
            for m in (keyboard.Key.shift, keyboard.Key.alt, keyboard.Key.cmd)
            if m not in config.HOTKEY_MODIFIERS
        )

        # 4. Trigger only if required modifiers are held, target key matches (char or vk), and no other modifiers are pressed
        if modifiers_held and is_target_key and not other_modifiers_pressed:
            # print(f"DEBUG: Hotkey trigger condition met (Modifiers: {modifiers_held}, Char: '{key_char}', VK: {key_vk})") # Commented out
            if processing_lock.acquire(blocking=False):
                processing_lock.release()
                print(
                    f"Hotkey triggered: {config.HOTKEY_MODIFIERS} + Char('{config.HOTKEY_CHAR}')/VK({config.HOTKEY_VK})"
                )
                process_thread = threading.Thread(
                    target=process_selected_text, daemon=True
                )
                process_thread.start()
            else:
                print("DEBUG: Lock busy, ignoring concurrent hotkey trigger.")
        # ------------------------------------------------------------- #

    except AttributeError:
        # getattr handles missing attributes gracefully, this might not be needed often
        pass
    except Exception as e:
        print(f"Error in on_press: {e}")


def on_release(key):
    """Handles key release events for hotkey detection."""
    global current_keys
    try:
        # Normalize modifier keys before attempting removal
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
        # print(f"DEBUG: Key released: {key}, Current keys: {current_keys}") # Commented out

    except Exception as e:
        print(f"Error in on_release: {e}")


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
    """Callback function when Exit is selected."""
    print("Exit selected. Stopping services...")
    if keyboard_listener:
        keyboard_listener.stop()
    if tray_icon:
        tray_icon.stop()
    # sys.exit() # Might be too abrupt, let threads finish if possible


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

    # --- Replay Last --- #
    replay_menu_text = (
        f"Replay Last Completed ({len(last_spoken_text)} chars)"  # Clarify 'Completed'
        if last_spoken_text
        else "Replay Last (Nothing Completed)"
    )

    def replay_action_func(icon, item):
        global stop_playback_flag  # Access flag
        if last_spoken_text:
            print(f"Replaying last completed segment: {last_spoken_text[:60]}...")
            # Signal stop for any current playback first
            stop_playback_flag.set()
            time.sleep(0.08)
            stop_playback_flag.clear()  # Clear before starting replay
            # Start speak_text in a thread
            threading.Thread(
                target=speak_text, args=(last_spoken_text,), daemon=True
            ).start()

    menu_items.append(pystray.Menu.SEPARATOR)
    menu_items.append(
        pystray.MenuItem(
            replay_menu_text, replay_action_func, enabled=bool(last_spoken_text)
        )
    )
    # ------------------- #

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

    # Start the keyboard listener in a separate thread
    if kokoro_pipeline:  # Check if pipeline is valid after potential init failure
        print(
            f"Starting hotkey listener ({config.HOTKEY_MODIFIERS} + Char('{config.HOTKEY_CHAR}')/VK({config.HOTKEY_VK}))..."
        )
        listener = keyboard.Listener(on_press=on_press, on_release=on_release)
        listener.start()
        # Update listener started message to use config
        modifier_names = " + ".join(
            str(k).split(".")[-1] for k in config.HOTKEY_MODIFIERS
        )
        print(
            f"Listener started. Press {modifier_names} + '{config.HOTKEY_CHAR}' key after selecting text."
        )

        # Setup and run the system tray icon in the main thread
        setup_tray()
    else:
        print("Exiting due to TTS initialization failure.")
