# audio_player.py - Handles TTS audio generation via pipeline, playback, chunking, and interruption.

import time
import threading
import numpy as np
import sounddevice as sd
import torch
import traceback

# Assuming config.py exists in the same directory or python path
import config

# Module-level state for playback control
_stop_playback_flag = threading.Event()
_last_spoken_text = None  # Stores the text of the last *successfully completed* segment

# --- Public Functions ---


def speak_text(text, pipeline, voice, speed):
    """
    Generates audio for a single text segment using the provided pipeline and plays it.
    Handles interruption via the module's stop flag.
    Updates _last_spoken_text on successful completion.

    Args:
        text (str): The text to speak.
        pipeline: The initialized KPipeline object.
        voice (str): The voice identifier (e.g., path).
        speed (float): The playback speed.

    Returns:
        bool: True if playback completed successfully without interruption, False otherwise.
    """
    global _last_spoken_text  # Allow modification

    if not pipeline:
        print("Error (audio_player): TTS pipeline not provided. Skipping.")
        return False
    if not text or not text.strip():
        return True  # Consider empty text a success

    is_error_message = text == config.COPY_FAILED_MESSAGE  # Check against config
    text_to_log = text[:50].replace("\\n", " ") + ("..." if len(text) > 50 else "")

    # --- Check stop flag before generation ---
    if _stop_playback_flag.is_set():
        print(
            f"   Playback interrupted before generation for: '{text_to_log}' - [player]"
        )
        return False

    print(f"   Generating audio for: '{text_to_log}'...")  # Simpler log
    # DO NOT update last_spoken_text here

    try:
        all_audio_segments = []
        generator = pipeline(text, voice=voice, speed=speed)  # Use passed arguments
        yielded_results_count = 0

        for i, result in enumerate(generator):
            # --- Check stop flag during generation ---
            if _stop_playback_flag.is_set():
                print(
                    f"   Playback interrupted during generation yield {i} for: '{text_to_log}' - [player]"
                )
                return False

            yielded_results_count += 1
            audio_segment = None
            # Simplified audio extraction logic (assuming correct result structure)
            if (
                hasattr(result, "output")
                and hasattr(result.output, "audio")
                and isinstance(result.output.audio, torch.Tensor)
            ):
                audio_tensor = result.output.audio
                if audio_tensor is not None:
                    audio_segment = audio_tensor.detach().cpu().numpy()
                    all_audio_segments.append(audio_segment)

        if not all_audio_segments:
            print(
                f"   TTS generation produced no valid audio for: '{text_to_log}' - [player]"
            )
            return False

        full_audio = np.concatenate(all_audio_segments)

        # --- Check stop flag before playback ---
        if _stop_playback_flag.is_set():
            print(f"   Playback interrupted before playing: '{text_to_log}' - [player]")
            return False

        print(
            f"   Playing audio ({len(full_audio)/config.SAMPLE_RATE:.2f}s) for: '{text_to_log}' - [player]"
        )
        sd.play(full_audio, config.SAMPLE_RATE)

        # --- Wait loop with stop check ---
        while sd.get_stream().active:
            if _stop_playback_flag.is_set():
                print(
                    f"   Playback interrupted during playback for: '{text_to_log}' - [player]"
                )
                sd.stop()  # Actively stop sounddevice playback
                return False
            time.sleep(0.05)

        # --- Check flag *after* playback finishes naturally ---
        if _stop_playback_flag.is_set():
            print(
                f"   Playback stopped by flag just after finishing naturally: '{text_to_log}' - [player]"
            )
            return False

        print(f"   Playback finished for: '{text_to_log}' - [player]")
        # Update last_spoken_text *only* on successful completion and if not error msg
        if not is_error_message:
            _last_spoken_text = text
        return True

    except sd.PortAudioError as pae:
        print(f"Error playing audio (SoundDevice PortAudioError): {pae} - [player]")
        sd.stop()
        return False
    except RuntimeError as rte:
        print(
            f"Error during TTS generation (RuntimeError likely from model): {rte} - [player]"
        )
        return False
    except Exception as e:
        print(
            f"Error during TTS or playback for '{text_to_log}': {type(e).__name__} - {e} - [player]"
        )
        traceback.print_exc()
        sd.stop()
        return False


def speak_sequentially(full_text, pipeline, voice, speed):
    """
    Speaks long text by splitting it into chunks and playing them sequentially.
    Handles interruption.

    Args:
        full_text (str): The entire text to speak.
        pipeline: The initialized KPipeline object.
        voice (str): The voice identifier.
        speed (float): The playback speed.
    """
    if not pipeline or not full_text:
        print(
            "Error (audio_player): Pipeline not ready or text is empty for sequential speaking."
        )
        return

    print(
        f"Processing text sequentially (Length: {len(full_text)}): {full_text[:50]}... - [player]"
    )

    current_pos = 0
    delimiters = '。！？….?!"'  # Added quote mark as potential delimiter
    segment_index = 0

    while current_pos < len(full_text):
        # --- Check stop flag at start of loop ---
        if _stop_playback_flag.is_set():
            print("   Sequence interrupted by flag at start of loop. - [player]")
            break

        segment_index += 1
        end_boundary = min(current_pos + config.MAX_CHUNK_CHARS, len(full_text))
        potential_chunk = full_text[current_pos:end_boundary]
        actual_chunk = ""
        next_pos = end_boundary

        # Simplified chunking logic (similar to before)
        best_split_index = -1
        for i in range(len(potential_chunk) - 1, -1, -1):
            if potential_chunk[i] in delimiters:
                best_split_index = current_pos + i + 1  # Split *after* delimiter
                break
        if best_split_index > current_pos and best_split_index <= end_boundary:
            actual_chunk = full_text[current_pos:best_split_index]
            next_pos = best_split_index
        elif end_boundary == len(full_text):  # Last part
            actual_chunk = potential_chunk
            next_pos = len(full_text)
        else:  # No suitable delimiter found, take max chunk
            actual_chunk = potential_chunk
            next_pos = end_boundary

        # --- Speak the determined chunk ---
        chunk_to_speak = actual_chunk.strip()
        if chunk_to_speak:
            # --- Check stop flag before speaking chunk ---
            if _stop_playback_flag.is_set():
                print("   Sequence interrupted before speaking next chunk. - [player]")
                break

            print(
                f"      Speaking chunk {segment_index}: '{chunk_to_speak[:40].replace(chr(10), ' ')}...' - [player]"
            )
            # Call the internal speak_text function
            success = speak_text(chunk_to_speak, pipeline, voice, speed)
            if not success:
                if _stop_playback_flag.is_set():
                    print(
                        f"      Segment {segment_index} interrupted by flag. Stopping sequence. - [player]"
                    )
                else:
                    print(
                        f"      Failed to speak segment {segment_index} (error). Stopping sequence. - [player]"
                    )
                break  # Stop processing further chunks

        current_pos = next_pos

    # --- Log sequence completion status ---
    if _stop_playback_flag.is_set():
        print("   Sequence processing stopped due to interruption flag. - [player]")
    elif current_pos >= len(full_text):
        print("   Finished processing all text successfully. - [player]")
    else:
        print(
            "   Sequence processing stopped prematurely (likely error in last chunk). - [player]"
        )


def stop_all_playback():
    """Signals any ongoing playback to stop immediately."""
    print("   Signalling audio player to stop playback...")
    _stop_playback_flag.set()
    # sd.stop() # Optional immediate stop


def clear_stop_flag():
    """Clears the internal stop playback flag."""
    # print("   Clearing audio player stop flag...") # Optional debug log
    _stop_playback_flag.clear()


def replay_last(pipeline, voice, speed):
    """Replays the last successfully spoken text segment."""
    global _last_spoken_text
    if _last_spoken_text:
        print(f"Replaying last completed segment: {_last_spoken_text[:60]}...")
        stop_all_playback()  # Signal any current playback to stop
        time.sleep(0.08)  # Short pause
        clear_stop_flag()  # <<< Use the new function here
        # Speak the last text in a new thread
        threading.Thread(
            target=speak_text,
            args=(_last_spoken_text, pipeline, voice, speed),
            daemon=True,
        ).start()
    else:
        print("Nothing spoken previously to replay.")


def get_last_spoken_text():
    """Returns the last successfully spoken text segment."""
    global _last_spoken_text
    return _last_spoken_text
