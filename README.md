# NoteSync
An 🧠 AI Lecture Recorder with Transcript & Summary

This is a smart AI-powered tool that **records your video lectures**, **extracts audio**, **generates a transcript**, and **summarizes it into neat notes** — all automatically.

Ideal for teachers, students, and content creators who want to deliver spoken content and instantly get it converted into organized written material.

---

## 🚀 Features

- 🎥 **Webcam Recording**: Captures high-quality video (with audio) using OpenCV.
- 🎙️ **Audio Capture**: Simultaneous audio recorded and saved separately.
- 📝 **Transcription**: Converts spoken content to text using Whisper AI.
- 📄 **Summarization**: Auto-generates short notes using BART from HuggingFace.
- ⏱️ **Timed + Manual Stop**: Records up to 5 mins or can be stopped early by hitting `Enter`.
- 📁 **Auto File Creation**: Saves `.mp4` video, `.wav` audio, transcript `.txt`, and summary `.txt` in a neatly organized `output/` folder.

---

## 📁 Output Example

| File Type     | Description                      | Format                       |
|---------------|----------------------------------|------------------------------|
| 🎬 Video       | Full recording with audio        | `YYYY-MM-DD_HH-MM-SS_video.mp4` |
| 🔊 Audio       | Extracted from mic               | `YYYY-MM-DD_HH-MM-SS_audio.wav` |
| 📝 Transcript  | Converted from speech using Whisper | `YYYY-MM-DD_HH-MM-SS_transcript.txt` |
| 📄 Summary     | Generated from transcript using BART | `YYYY-MM-DD_HH-MM-SS_summary.txt` |

---

### ✅ Prerequisites

- Python 3.8+
- FFmpeg (must be installed and available in PATH)


## **▶️ Usage**
- python prototype.py

- Webcam and microphone start recording immediately.
- Press Enter anytime to stop (max time: 5 minutes).
- Outputs are saved with current timestamp in the /output folder.


## **🧠 How It Works**
- Starts recording your webcam + mic
- Saves:
  - Video (.mp4)
  - Audio (.wav)
- Generates:
- Transcript using OpenAI Whisper
- Summary using HuggingFace Transformers
- Saves all outputs in a timestamped format for future reference


## **🛠 Tech Stack**
- Layer	Technology
- Audio/Video	OpenCV, SoundDevice, FFmpeg
- Transcription	OpenAI Whisper
- Summarization	HuggingFace Transformers (BART)
- Language	Python 3.8+


## **✨Future Improvements**
- GUI version
- PDF/Doc export of notes
- Integration with Google Drive/Cloud upload
- Multiple language support
- Speaker diarization (who said what)

## **📄 License**
**MIT License. Free for personal and educational use.**

## **🙏 Model Credits**
- OpenAI Whisper
- HuggingFace Transformers
- FFmpeg

## **Drawbacks**
- Doesn't able to give Live Captions on recording video.
- The Audio from Video has been Seperated and Saving Seperately.
