import os
from faster_whisper import WhisperModel
from pydub import AudioSegment
import numpy as np

# =========================
# Whisper 模型
# =========================

print("Loading Whisper model...")

model = WhisperModel(
    "base",
    device="cpu",
    compute_type="int8"
)

print("Whisper loaded.")


# =========================
# 录音函数
# =========================

def is_silent(audio_path, threshold=-30):
    audio = AudioSegment.from_wav(audio_path)

    print("Audio volume, threshold=-30:", audio.dBFS)

    return audio.dBFS < threshold

def listen():

    print("Please speak...")

    wav_file = "test.wav"

    # 删除旧文件
    if os.path.exists(wav_file):
        os.remove(wav_file)

    # 录音
    os.system(

        "arecord "
        "-D plughw:2,0 "
        "-f S16_LE "
        "-r 16000 "
        "-d 7 "
        "test.wav"
    )

    # 判断文件是否存在
    if not os.path.exists(wav_file):

        print("Recording failed.")

        return ""

    try:

        print("Whisper transcribing...")

        if is_silent("test.wav"):
            print("Silent audio detected, skip transcription.")
            return "", ""

        segments, info = model.transcribe(
            wav_file,
            beam_size=5,
            vad_filter=True
        )

        print("Detected language:", info.language)

        text = ""

        for segment in segments:

            text += segment.text

        text = text.strip()

        print("Recognized text:", text)

        return text, info.language

    except Exception as e:

        print("Whisper transcription failed:", e)

        return "", ""
