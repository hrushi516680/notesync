from utils.recorder import record_lecture
from utils.drive_upload import upload_to_drive
from utils.whatsapp import send_whatsapp_summary
from utils.summary import generate_summary
import datetime

def main():
    print("🎥 Starting Lecture Recording...")
    video_path, transcript, keywords, highlights = record_lecture()

    print("🧠 Generating Summary...")
    summary_text = generate_summary(transcript)

    print("☁️ Uploading to Google Drive...")
    drive_link = upload_to_drive(video_path)

    print("📲 Sending WhatsApp Message...")
    today = datetime.datetime.now().strftime("%B %d")
    message = f"""🎓 Lecture - {today}
🔗 Video + Notes: {drive_link}
📌 Key Points: {", ".join(keywords) if keywords else "N/A"}
🧠 Highlight: {highlights if highlights else "None"}"""
    send_whatsapp_summary(message)

if __name__ == "__main__":
    main()
