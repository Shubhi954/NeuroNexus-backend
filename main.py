import os
import shutil
import subprocess
import uuid
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse

app = FastAPI()


WHISPER_PATH = r"C:\Users\harsh\OneDrive\Desktop\tying\NeuroNexus-backend\whisper\whisper-cli.exe"
MODEL_PATH = r"C:\Users\harsh\OneDrive\Desktop\tying\NeuroNexus-backend\whisper\models\ggml-base.en.bin"


# -------------------------------------------------------
# Convert uploaded audio to WAV
# -------------------------------------------------------
def convert_to_wav(input_file: str) -> str:
    output_file = f"converted_{uuid.uuid4()}.wav"

    result = subprocess.run(
        ["ffmpeg", "-y", "-i", input_file, output_file],
        capture_output=True,
        text=True
    )

    if result.returncode != 0:
        print(result.stderr)
        raise Exception("FFmpeg conversion failed")

    return output_file


# -------------------------------------------------------
# Get Audio Duration (Dynamic)
# -------------------------------------------------------
def get_audio_duration(file_path: str) -> float:
    result = subprocess.run(
        [
            "ffprobe",
            "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            file_path
        ],
        capture_output=True,
        text=True
    )

    if result.returncode != 0:
        raise Exception("Failed to get audio duration")

    return float(result.stdout.strip())


# -------------------------------------------------------
# Transcribe using whisper.cpp
# -------------------------------------------------------
def transcribe_local(file_path: str) -> str:

    command = [
        WHISPER_PATH,
        "-m", MODEL_PATH,
        "-f", file_path,
        "-otxt"
    ]

    result = subprocess.run(
        command,
        capture_output=True,
        text=True
    )

    if result.returncode != 0:
        print(result.stderr)
        raise Exception("Whisper execution failed")

    txt_file = file_path + ".txt"

    if not os.path.exists(txt_file):
        raise Exception("Transcript file not created")

    with open(txt_file, "r", encoding="utf-8") as f:
        transcript = f.read().strip()

    os.remove(txt_file)

    return transcript


# -------------------------------------------------------
# Analyze Voice Endpoint
# -------------------------------------------------------
@app.post("/analyze-voice")
async def analyze_voice(audio_file: UploadFile = File(...)):

    try:
        # Save uploaded file
        original_filename = f"upload_{uuid.uuid4()}_{audio_file.filename}"

        with open(original_filename, "wb") as buffer:
            shutil.copyfileobj(audio_file.file, buffer)

        # Convert to wav
        wav_file = convert_to_wav(original_filename)

        # Get duration dynamically
        duration_seconds = get_audio_duration(wav_file)
        duration_minutes = duration_seconds / 60

        # Transcribe
        transcript = transcribe_local(wav_file)

        # Cleanup audio files
        os.remove(original_filename)
        os.remove(wav_file)

        # ----------------------------
        # TEXT FEATURE EXTRACTION
        # ----------------------------

        words = transcript.lower().split()
        total_words = len(words)

        # Speech Rate (Words Per Minute)
        speech_rate = (
            total_words / duration_minutes
            if duration_minutes > 0 else 0
        )

        # Lexical Diversity
        unique_words = len(set(words))
        lexical_diversity = (
            unique_words / total_words
            if total_words > 0 else 0
        )

        # Average Word Length
        avg_word_length = (
            sum(len(word) for word in words) / total_words
            if total_words > 0 else 0
        )

        # Average Sentence Length
        sentences = [s for s in transcript.split(".") if s.strip()]
        avg_sentence_length = (
            total_words / len(sentences)
            if len(sentences) > 0 else 0
        )

        # Filler Word Count
        filler_words = ["um", "uh", "like", "you know"]
        filler_count = sum(words.count(f) for f in filler_words)

        # Repetition Ratio
        repetition_ratio = (
            1 - (unique_words / total_words)
            if total_words > 0 else 0
        )

        # ----------------------------
        # SCORING LOGIC 
        # ----------------------------

        score = 100

        # Speech Rate (Normal 120–160 WPM)
        if speech_rate < 90:
            score -= 20
        elif speech_rate < 110:
            score -= 10

        # Lexical Diversity
        if lexical_diversity < 0.4:
            score -= 20
        elif lexical_diversity < 0.5:
            score -= 10

        # Short sentences
        if avg_sentence_length < 5:
            score -= 15

        # Very small vocabulary usage
        if avg_word_length < 3:
            score -= 10

        # Too many filler words
        if filler_count > 3:
            score -= 10

        # High repetition
        if repetition_ratio > 0.5:
            score -= 10

        score = max(score, 0)

        # ----------------------------
        # RESPONSE
        # ----------------------------

        return JSONResponse(content={
            "transcript": transcript,
            "features": {
                "duration_seconds": round(duration_seconds, 2),
                "speech_rate_wpm": round(speech_rate, 2),
                "lexical_diversity": round(lexical_diversity, 2),
                "avg_word_length": round(avg_word_length, 2),
                "avg_sentence_length": round(avg_sentence_length, 2),
                "filler_count": filler_count,
                "repetition_ratio": round(repetition_ratio, 2)
            },
            "score": score
        })

    except Exception as e:
        print("ERROR:", str(e))
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )
