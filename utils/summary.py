import os
import glob
import json
from transformers import pipeline

# Initialize summarizer
summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")

def generate_summary(transcript):
    try:
        length = min(75, max(20, len(transcript.split()) // 2))
        summary = summarizer(transcript, max_length=length, min_length=15, do_sample=False)
        return summary[0]['summary_text']
    except Exception as e:
        print(f"❌ Summary generation failed: {e}")
        return "Summary unavailable."

def read_transcript_from_metadata(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            if isinstance(data, dict) and 'transcript' in data:
                return data['transcript']
            elif isinstance(data, str):
                return data  # if it's plain text
            else:
                return json.dumps(data)  # fallback
    except Exception as e:
        print(f"❌ Failed to read transcript file: {e}")
        return None

if __name__ == "__main__":
    # 🔍 Locate latest metadata JSON file
    script_dir = os.path.dirname(os.path.abspath(__file__))
    recordings_dir = os.path.abspath(os.path.join(script_dir, "..", "recordings"))
    metadata_files = sorted(glob.glob(os.path.join(recordings_dir, "metadata_*.json")), reverse=True)

    if not metadata_files:
        print("❌ No metadata_*.json file found in recordings/")
        exit()

    transcript_path = metadata_files[0]
    print(f"📂 Using metadata file: {os.path.basename(transcript_path)}")

    transcript = read_transcript_from_metadata(transcript_path)
    if transcript:
        print("📝 Transcript loaded successfully.")
        summary = generate_summary(transcript)
        print("\n📌 Summary:\n")
        print(summary)
    else:
        print("⚠️ No transcript to summarize.")
