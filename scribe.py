import streamlit as st
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

# --- CONFIGURATION ---
# The Ear: Fast, accurate, optimized for M-chips
WHISPER_MODEL = "mlx-community/whisper-small-mlx"
# The Brain: Small, fast, low-RAM usage
OLLAMA_MODEL = "qwen3:0.6b" 

SAMPLE_RATE = 16000
CHUNK_DURATION = 7  # Transcribe every 5 seconds
SUMMARY_INTERVAL = 25 # Process summaries every 10 transcribed chunks (approx 50s)

# --- STATE MANAGEMENT ---
if 'recording' not in st.session_state:
    st.session_state.recording = False
if 'transcript_text' not in st.session_state:
    st.session_state.transcript_text = [] # Stores full text log
if 'audio_buffer' not in st.session_state:
    st.session_state.audio_buffer = [] # Stores raw audio for WAV dump
if 'summary_log' not in st.session_state:
    st.session_state.summary_log = ""
if 'last_summary_idx' not in st.session_state:
    st.session_state.last_summary_idx = 0

# --- QUEUES ---
# We use queues to pass data safely between the background audio thread and the UI
@st.cache_resource
def get_queues():
    return queue.Queue(), queue.Queue()

audio_q, result_q = get_queues()

# --- WORKER FUNCTIONS ---
def audio_callback(indata, frames, time, status):
    """Real-time audio capture callback"""
    if status:
        print(status)
    audio_q.put(indata.copy())

def processing_thread():
    """Background thread: Consumes audio -> MLX Whisper -> Text Queue"""
    print(f"🚀 Loading Whisper Model: {WHISPER_MODEL}...")
    
    while True:
        # Collect 5 seconds of audio
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
        # We must write to a temp file because MLX Whisper expects a file path
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tmp:
            wav.write(tmp.name, SAMPLE_RATE, np_audio)
            
            try:
                result = mlx_whisper.transcribe(tmp.name, path_or_hf_repo=WHISPER_MODEL)
                text = result['text'].strip()
                
                if text:
                    timestamp = datetime.now().strftime("%H:%M:%S")
                    formatted = f"**[{timestamp}]** {text}"
                    # Send both text and raw audio to the main thread
                    result_q.put((formatted, np_audio))
            except Exception as e:
                print(f"Transcription Error: {e}")

# Start the processing thread only once
if 'thread_started' not in st.session_state:
    t = threading.Thread(target=processing_thread, daemon=True)
    t.start()
    st.session_state.thread_started = True

# --- LLM FUNCTION ---
def generate_summary(text_block):
    # ... prompt definition ...
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

    # 👇 ADD THIS: Print the Prompt to your Terminal
    print("\n" + "="*40)
    print("📤 SENDING PROMPT TO OLLAMA:")
    print(prompt)
    print("="*40 + "\n")

    try:
        response = ollama.chat(
            model=OLLAMA_MODEL, 
            messages=[{'role': 'user', 'content': prompt}],
            options={'temperature': 0.6}
        )
        content = response['message']['content']

        # 👇 ADD THIS: Print the Response to your Terminal
        print("\n" + "="*40)
        print("📥 RECEIVED RESPONSE:")
        print(content)
        print("="*40 + "\n")

        # ... rest of your formatting code ...
        return content

    except Exception as e:
        print(f"❌ ERROR: {e}") # Print errors too
        return f"⚠️ LLM Error: {e}"
    
    
# --- UI LAYOUT ---
st.set_page_config(layout="wide", page_title="Live Scribe M4")
st.title("🎙️ Live Scribe (M4 Edition)")

# Sidebar for Setup
with st.sidebar:
    st.header("Setup")
    device_list = sd.query_devices()
    input_names = [d['name'] for d in device_list if d['max_input_channels'] > 0]
    
    # Auto-select BlackHole if available
    default_idx = 0
    for i, name in enumerate(input_names):
        if "BlackHole" in name:
            default_idx = i
            break
            
    selected_device = st.selectbox("Input Source", input_names, index=default_idx)
    st.info("ℹ️ Select 'BlackHole 2ch' to capture system audio (Teams/Zoom).")

col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Live Transcript")
    # Using a container with fixed height for scrolling
    transcript_box = st.container(height=500, border=True)
    
    # Process incoming data from queue
    while not result_q.empty():
        text, raw_audio = result_q.get()
        st.session_state.transcript_text.append(text)
        st.session_state.audio_buffer.append(raw_audio)
    
    # Display lines
    with transcript_box:
        for line in st.session_state.transcript_text:
            st.markdown(line)

with col2:
    st.subheader("AI Summaries (Llama 3.2)")
    summary_box = st.container(height=500, border=True)
    
    # Check if we have enough new text to summarize
    current_len = len(st.session_state.transcript_text)
    if current_len - st.session_state.last_summary_idx >= SUMMARY_INTERVAL:
        # Get the new block of text
        recent_lines = st.session_state.transcript_text[st.session_state.last_summary_idx:]
        # Strip timestamps for cleaner LLM input
        clean_text = " ".join([line.split("]** ")[-1] for line in recent_lines])
        
        with st.spinner("Generating insights..."):
            new_summary = generate_summary(clean_text)
            timestamp = datetime.now().strftime("%H:%M")
            st.session_state.summary_log += f"\n\n**Time {timestamp}**\n{new_summary}"
            st.session_state.last_summary_idx = current_len

    with summary_box:
        st.markdown(st.session_state.summary_log)

# --- CONTROLS ---
st.divider()
c1, c2 = st.columns(2)

with c1:
    # Toggle Recording
    btn_label = "🛑 Stop Listening" if st.session_state.recording else "▶️ Start Listening"
    if st.button(btn_label, type="primary" if not st.session_state.recording else "secondary"):
        st.session_state.recording = not st.session_state.recording
        
        if st.session_state.recording:
            # Start Audio Stream
            try:
                dev_idx = input_names.index(selected_device)
                # We need the actual Device ID from the original list
                real_id = [d for d in device_list if d['name'] == selected_device][0]['index']
                
                stream = sd.InputStream(device=real_id, channels=1, samplerate=SAMPLE_RATE, callback=audio_callback)
                stream.start()
                st.session_state.stream = stream
                st.toast(f"Listening on {selected_device}...")
            except Exception as e:
                st.error(f"Could not start audio: {e}")
                st.session_state.recording = False
        else:
            # Stop Audio Stream
            if 'stream' in st.session_state:
                st.session_state.stream.stop()
                st.session_state.stream.close()

with c2:
    if st.button("💾 Save Meeting Data"):
        # Save Audio
        if st.session_state.audio_buffer:
            full_audio = np.concatenate(st.session_state.audio_buffer)
            wav.write("meeting_audio.wav", SAMPLE_RATE, full_audio)
            
        # Save Text
        with open("meeting_transcript.txt", "w") as f:
            f.write("\n".join(st.session_state.transcript_text))
        
        st.success("Saved 'meeting_audio.wav' and 'meeting_transcript.txt' to project folder!")

# Refresh UI to keep data flowing
if st.session_state.recording:
    time.sleep(1)
    st.rerun()