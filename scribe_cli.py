import mlx_whisper
import sounddevice as sd
import numpy as np
import scipy.io.wavfile as wav
import threading
import queue
import time
import os
import tempfile
import ollama
from datetime import datetime
import sys

# --- CONFIGURATION ---
# The Ear: Fast, accurate, optimized for M-chips
WHISPER_MODEL = "mlx-community/whisper-small-mlx"
# The Brain: Small, fast, low-RAM usage
OLLAMA_MODEL = "qwen3:0.6b" 

SAMPLE_RATE = 16000
CHUNK_DURATION = 5  # Transcribe every 7 seconds
SUMMARY_INTERVAL = 15 # Process summaries every 4 transcribed chunks (approx 28s)

# --- STATE MANAGEMENT ---
recording = False
transcript_text = [] # Stores full text log
audio_buffer = [] # Stores raw audio for WAV dump
summary_log = ""
last_summary_idx = 0
stream = None

# --- QUEUES ---
audio_q = queue.Queue()
result_q = queue.Queue()

# --- WORKER FUNCTIONS ---
def audio_callback(indata, frames, time, status):
    """Real-time audio capture callback"""
    if status:
        print(status, file=sys.stderr)
    audio_q.put(indata.copy())

def processing_thread():
    """Background thread: Consumes audio -> MLX Whisper -> Text Queue"""
    print(f"🚀 Loading Whisper Model: {WHISPER_MODEL}...")
    
    while True:
        # Collect audio chunks
        audio_chunk = []
        while len(audio_chunk) < (SAMPLE_RATE * CHUNK_DURATION):
            if audio_q.empty():
                time.sleep(0.1)
                continue
            data = audio_q.get().flatten()
            audio_chunk.extend(data)
        
        # Convert to numpy array
        np_audio = np.array(audio_chunk, dtype=np.float32)

        # Transcribe
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tmp:
            wav.write(tmp.name, SAMPLE_RATE, np_audio)
            
            try:
                result = mlx_whisper.transcribe(tmp.name, path_or_hf_repo=WHISPER_MODEL)
                text = result['text'].strip()
                
                if text:
                    timestamp = datetime.now().strftime("%H:%M:%S")
                    formatted = f"**[{timestamp}]** {text}"
                    result_q.put((formatted, np_audio))
            except Exception as e:
                print(f"Transcription Error: {e}", file=sys.stderr)

# --- LLM FUNCTION ---
def generate_summary(text_block):
    prompt = f"""
    [INST]
    You are a strategic advisor. Analyze this transcript segment.
    
    TASK:
    1. Detect the Tone.
    2. Find 2 hidden implications (what is being avoided or implied?).
    3. Formulate 1 sharp follow-up question.

    REQUIRED OUTPUT FORMAT:
    Mood: [Tone]
    - [Implication 1]
    - [Implication 2]
    - Qn: [Your Strategic Question?]

    RULES:
    - Keep points under 15 words.
    - No intro text.
    - The last point MUST start with "Qn:".

    TRANSCRIPT SEGMENT:
    {text_block}
    [/INST]
    """
    # print("\n" + "="*40)
    # print("📤 SENDING PROMPT TO OLLAMA:")
    # # print(prompt) # This is too verbose for stdout
    # print("="*40 + "\n")

    try:
        response = ollama.chat(
            model=OLLAMA_MODEL, 
            messages=[{'role': 'user', 'content': prompt}],
            options={'temperature': 0.6}
        )
        content = response['message']['content']

        # print("\n" + "="*40)
        # print("📥 RECEIVED RESPONSE:")
        print(content)
        # print("="*40 + "\n")
        return content
    except Exception as e:
        print(f"❌ ERROR: {e}", file=sys.stderr)
        return f"⚠️ LLM Error: {e}"

def select_input_device():
    """Lists available input devices and prompts the user to select one."""
    try:
        device_list = sd.query_devices()
        input_devices = [d for d in device_list if d['max_input_channels'] > 0]
        
        if not input_devices:
            print("No audio input devices found.", file=sys.stderr)
            sys.exit(1)

        print("Available input devices:")
        for i, d in enumerate(input_devices):
            print(f"{i}: {d['name']}")
        
        default_idx = -1
        for i, d in enumerate(input_devices):
            if "BlackHole" in d['name']:
                default_idx = i
                break
        
        prompt = f"Select input device number (default: {default_idx if default_idx != -1 else 0}): "
        try:
            choice = input(prompt)
            if choice == "" and default_idx != -1:
                 dev_idx = default_idx
            else:
                dev_idx = int(choice or (default_idx if default_idx != -1 else 0))

            selected_device = input_devices[dev_idx]
            return selected_device['index']
        except (ValueError, IndexError):
            print("Invalid selection. Using default device.", file=sys.stderr)
            return sd.default.device['input']

    except Exception as e:
        print(f"Could not query audio devices: {e}", file=sys.stderr)
        sys.exit(1)


def save_data():
    print("\n💾 Saving meeting data...")
    # Save Audio
    if audio_buffer:
        full_audio = np.concatenate(audio_buffer)
        wav.write("meeting_audio.wav", SAMPLE_RATE, full_audio)
        print("✅ Saved 'meeting_audio.wav'")
        
    # Save Text
    if transcript_text:
        with open("meeting_transcript.txt", "w") as f:
            f.write("\n".join(transcript_text))
        print("✅ Saved 'meeting_transcript.txt'")

def main():
    global recording, transcript_text, audio_buffer, summary_log, last_summary_idx, stream

    # Start the background processing thread
    proc_thread = threading.Thread(target=processing_thread, daemon=True)
    proc_thread.start()

    # Select device and start recording
    selected_device_id = select_input_device()
    print("\n▶️  Press Enter to start listening, or Ctrl+C to exit.")
    input()


    try:
        print(f"🎙️ Listening on device {selected_device_id}... Press Ctrl+C to stop.")
        stream = sd.InputStream(device=selected_device_id, channels=1, samplerate=SAMPLE_RATE, callback=audio_callback)
        stream.start()
        recording = True

        while recording:
            try:
                # Process incoming data from queue
                while not result_q.empty():
                    text, raw_audio = result_q.get()
                    transcript_text.append(text)
                    audio_buffer.append(raw_audio)
                    print(text) # Print live transcript

                # Check if we have enough new text to summarize
                current_len = len(transcript_text)
                if current_len - last_summary_idx >= SUMMARY_INTERVAL:
                    recent_lines = transcript_text[last_summary_idx:]
                    clean_text = " ".join([line.split("]** ")[-1] for line in recent_lines])
                    
                    print("\n🧠 Generating insights...")
                    new_summary = generate_summary(clean_text)
                    timestamp = datetime.now().strftime("%H:%M")
                    summary_log += f"\n\n**Time {timestamp}**\n{new_summary}"
                    last_summary_idx = current_len
                
                time.sleep(0.5)

            except KeyboardInterrupt:
                recording = False
                print("\n🛑 Stopping listening...")

    finally:
        if stream is not None:
            stream.stop()
            stream.close()
        save_data()
        print("Exiting.")


if __name__ == "__main__":
    main()
