from utils.recorder import VoiceActivatedLectureRecorder  # this is your new class
from utils.drive_upload import upload_to_drive
from utils.whatsapp import send_whatsapp_summary
from utils.summary import generate_summary
import datetime

def main():
    print("🎥 Starting Voice-Activated Recorder...")

    # Step 1: Initialize and Record
    recorder = VoiceActivatedLectureRecorder()
    result = recorder.listen_for_commands()  # this will block until STOP is heard

    if result is None:
        print("❌ Recording failed or was cancelled.")
        return

    video_path, transcript, keywords, highlights = result

    # Step 2: Generate summary
    print("🧠 Generating Summary...")
    summary_text = generate_summary(transcript)

    # Step 3: Upload to Drive
    print("☁️ Uploading to Google Drive...")
    drive_link = upload_to_drive(video_path)

    # Step 4: Send on WhatsApp
    print("📲 Sending WhatsApp Message...")
    today = datetime.datetime.now().strftime("%B %d")
    message = f"""🎓 Lecture - {today}
🔗 Video + Notes: {drive_link}
🧠 Summary: {summary_text}
📌 Key Points: {", ".join(keywords) if keywords else "N/A"}
⭐ Highlight: {highlights if highlights else "None"}"""

    send_whatsapp_summary(message)
    print("✅ All done!")

if __name__ == "__main__":
    main()
