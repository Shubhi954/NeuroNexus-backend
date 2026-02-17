import os
import shutil
import subprocess
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse



app = FastAPI()


def transcribe_local(file_path):

    whisper_path = "whisper/whisper-cli.exe"
    model_path = "whisper/models/ggml-base.en.bin"

    command = [
        whisper_path,
        "-m", model_path,
        "-f", file_path,
        "-otxt"
    ]

    subprocess.run(command)

    txt_file = file_path + ".txt"

    with open(txt_file, "r", encoding="utf-8") as f:
        transcript = f.read().strip()

    return transcript


@app.post("/analyze-voice")
async def analyze_voice(audio_file: UploadFile = File(...)):

    temp_filename = "temp_audio.wav"

    # Save uploaded file
    with open(temp_filename, "wb") as buffer:
        shutil.copyfileobj(audio_file.file, buffer)

    # Use local whisper
    transcript = transcribe_local(temp_filename)

    os.remove(temp_filename)

    # -------- Feature Calculation --------
    total_words = len(transcript.split())

    # Since we don't have timestamps now,
    # assume average 15 second recording
    speech_rate = total_words / 15

    # -------- Score --------
    score = 100

    if speech_rate < 1.5:
        score -= 20

    score = max(score, 0)

    return JSONResponse(content={
        "transcript": transcript,
        "features": {
            "speech_rate": round(speech_rate, 2)
        },
        "score": score
    })

