from transformers import pipeline

def generate_summary(transcript):
    try:
        summarizer = pipeline(
            "summarization",
            model="sshleifer/distilbart-cnn-12-6",
            revision="a4f8f3e"
        )

        length = min(75, max(20, len(transcript.split()) // 2))  # auto size
        summary = summarizer(transcript, max_length=length, min_length=15, do_sample=False)
        return summary[0]['summary_text']

    except Exception as e:
        print(f"❌ Summary generation failed: {e}")
        return "Summary unavailable."

