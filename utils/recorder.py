import time
import os

def record_lecture():
    # Placeholder for full audio/video recording logic
    # Let's assume it records and returns:
    # - path to saved video file
    # - transcribed text
    # - detected keywords
    # - highlight notes (e.g., "exam tip at 24:30")

    print("[DEBUG] Recording lecture...")
    time.sleep(2)  # simulate delay

    video_path = os.path.join("recordings", "lecture_sample.mp4")
    transcript = "Today we discussed capacitors. Exam tip: understand charge storage at 24:30."
    keywords = ["capacitors", "exam tip"]
    highlights = "Exam tip at 24:30"

    return video_path, transcript, keywords, highlights
