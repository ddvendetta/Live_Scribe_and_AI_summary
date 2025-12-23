import ollama
import os
from datetime import datetime

# --- CONFIGURATION ---
OLLAMA_MODEL = "qwen3:0.6b"
TRANSCRIPT_FILE = "/Users/farhankhairuddin/Project_11/XV_GM_Townhall_year_end_2025/meeting_transcript.txt"
SUMMARY_INTERVAL = 25  # Process summaries every 10 transcribed chunks

# --- LLM FUNCTION ---
def generate_summary(text_block, model):
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
    try:
        response = ollama.chat(
            model=model,
            messages=[{'role': 'user', 'content': prompt}],
            options={'temperature': 0.6}
        )
        return response

    except Exception as e:
        print(f"❌ ERROR: {e}") # Print errors too
        return None

def main():
    """
    Reads a transcript file, processes it in chunks, and generates summaries,
    saving the output to a markdown file.
    """
    if not os.path.exists(TRANSCRIPT_FILE):
        print(f"❌ ERROR: Transcript file not found at '{TRANSCRIPT_FILE}'")
        return

    # Generate filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # Clean model name for filename
    model_name_clean = OLLAMA_MODEL.replace(":", "_").replace("/", "_")
    output_filename = f"summary_test_{model_name_clean}_{timestamp}.md"

    with open(TRANSCRIPT_FILE, "r") as f:
        lines = f.readlines()

    # Strip timestamps and speaker info for cleaner LLM input
    clean_lines = [line.split("]** ")[-1].strip() for line in lines if "]** " in line]

    print(f"📄 Processing {len(clean_lines)} lines from '{TRANSCRIPT_FILE}'...")
    print(f"💾 Output will be saved to '{output_filename}'")

    with open(output_filename, "w") as out_file:
        for i in range(0, len(clean_lines), SUMMARY_INTERVAL):
            chunk = clean_lines[i:i + SUMMARY_INTERVAL]
            text_block = " ".join(chunk)

            if text_block.strip():
                print(f"--- Processing lines {i+1}-{i+len(chunk)} ---")
                
                # Write the input chunk to the markdown file
                out_file.write("### ✍️ Prompt Input (Lines {}-{}):\n\n".format(i+1, i+len(chunk)))
                for line in chunk:
                    out_file.write(f"- {line}\n")
                out_file.write("\n")

                # Generate the summary
                response = generate_summary(text_block, OLLAMA_MODEL)

                if response:
                    # --- 1. Extract the Raw Data ---
                    summary = response['message']['content']
                    input_tokens = response.get('prompt_eval_count', 0)
                    output_tokens = response.get('eval_count', 0)
                    eval_duration_ns = response.get('eval_duration', 1)
                    total_duration_ns = response.get('total_duration', 1)

                    # --- 2. Calculate Speed (Tokens per second) ---
                    tokens_per_second = (output_tokens / eval_duration_ns) * 1e9 if eval_duration_ns > 0 else 0
                    total_time_s = total_duration_ns / 1e9

                    # --- 3. Print the Statistics to Console ---
                    print(f"--- Session Stats ---")
                    print(f"Input Tokens:  {input_tokens}")
                    print(f"Output Tokens: {output_tokens}")
                    print(f"Speed:         {tokens_per_second:.2f} tokens/s")
                    print(f"Total Time:    {total_time_s:.2f}s")

                    # --- 4. Format Statistics for Markdown ---
                    stats_md = (
                        f"\n\n**Performance Metrics:**\n"
                        f"- **Input Tokens:** {input_tokens}\n"
                        f"- **Output Tokens:** {output_tokens}\n"
                        f"- **Speed:** {tokens_per_second:.2f} tokens/s\n"
                        f"- **Total Time:** {total_time_s:.2f}s"
                    )
                else:
                    summary = "⚠️ LLM Error: No response generated."
                    stats_md = ""

                # Write the summary and stats to the markdown file
                out_file.write("### 🤖 Model Response:\n\n")
                out_file.write(summary)
                out_file.write(stats_md)
                out_file.write("\n\n---\n\n")

    print(f"✅ Done! Summary saved to '{output_filename}'")


if __name__ == "__main__":
    main()
