# Live Scribe (M4 Edition)

This is a Streamlit application that provides live transcription and summarization of audio.

## Features

-   **Live Transcription:** Uses `mlx_whisper` to transcribe audio in real-time.
-   **AI Summaries:** Uses `ollama` with `llama3.2:1b` to generate summaries of the transcript.
-   **Audio Recording:** Can record audio from a selected input source.
-   **Data Saving:** Saves the meeting audio and transcript to files.

## Development Environment

This application was developed and tested on a **MacBook Air M4 with 16GB of RAM**.

-   **Whisper Model:** `mlx-community/whisper-small-mlx`
-   **Ollama Model:** `llama3.2:1b`

Due to hardware constraints, these models were chosen to maintain a stable memory utilization of approximately **12GB**.

## How to Run

1.  Install the dependencies:
    ```bash
    pip install -r requirements.txt
    ```
2.  Run the Streamlit app:
    ```bash
    streamlit run scribe.py
    ```

## Configuration

-   **Whisper Model:** `mlx-community/whisper-small-mlx`
-   **Ollama Model:** `llama3.2:1b`
-   **Sample Rate:** 16000
-   **Chunk Duration:** 7 seconds
-   **Summary Interval:** 10 transcribed chunks (approx 50 seconds)

## Prerequisites and Setup

### Phase 1: System-Level Prerequisites
These are the tools that must be installed on your Mac before any Python code will work. Open your Terminal and run these commands one by one.

**Install Homebrew (If you haven't already):**

```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

**Install Audio & Video Drivers:**

*   `portaudio`: Required for the microphone script to access your hardware.
*   `ffmpeg`: Required by Whisper to process audio files.
*   `blackhole-2ch`: The virtual cable to route Teams/Browser audio into the script.

```bash
brew install portaudio ffmpeg
brew install --cask blackhole-2ch
```

**Important:** You may be asked to restart your Mac after installing BlackHole. If so, restart before proceeding.

**Install Ollama (The AI Brain):**

```bash
brew install ollama
```

Start and Prep Ollama: Open a new terminal tab and run `ollama serve` to start the background process. Then, in your main terminal, pull the memory-efficient model we selected:

```bash
ollama pull llama3.2
```

### The Bridge (Audio Routing)
Since you cannot plug a cable from your speaker to your microphone, we use a virtual cable.

**Install BlackHole (Virtual Driver)**

This acts as a virtual audio cable.

```bash
brew install blackhole-2ch
```
(Or download the installer from the BlackHole website)

**Create a Multi-Output Device (Crucial Step)**

If you just select "BlackHole" as your output, you won't hear the Teams call.

1.  Open **Audio MIDI Setup** (`cmd+space`, type "Audio MIDI").
2.  Click the **+** button (bottom left) -> **Create Multi-Output Device**.
3.  In the list on the right, check both:
    *   **BlackHole 2ch**
    *   **MacBook Pro Speakers** (or your headphones/external speakers).
4.  **Master Device**: Set to your Speakers/Headphones (this keeps clock sync correct).

**Set Your Mac's Output**

1.  Go to **System Settings -> Sound -> Output**.
2.  Select the **Multi-Output Device** you just created.

**Note:** You cannot change volume via the keyboard keys while using this. Set your volume comfortably before switching.
