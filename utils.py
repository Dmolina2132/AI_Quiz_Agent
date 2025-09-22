from state import AgentState
import subprocess
from openai import OpenAI
from dotenv import load_dotenv
from pathlib import Path
import openai
import os
import time
import threading

os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"
import pygame

load_dotenv()
client = OpenAI()


LANGUAGE_CODES = {
    "en": "English",
    "es": "Spanish",
    "fr": "French",
    "de": "German",
    "zh": "Chinese",
    "hi": "Hindi",
    "ar": "Arabic",
    "ru": "Russian",
    "ja": "Japanese",
    "pt": "Portuguese",
    "bn": "Bengali",
    "pa": "Punjabi",
    "jv": "Javanese",
    "ko": "Korean",
    "vi": "Vietnamese",
    "it": "Italian",
    "tr": "Turkish",
    "fa": "Persian",
    "ur": "Urdu",
    "th": "Thai",
    "ms": "Malay",
    "id": "Indonesian",
    "ta": "Tamil",
    "mr": "Marathi",
    "te": "Telugu",
    "gu": "Gujarati",
    "pl": "Polish",
    "uk": "Ukrainian",
    "ro": "Romanian",
    "nl": "Dutch",
    "el": "Greek",
    "hu": "Hungarian",
    "sv": "Swedish",
    "fi": "Finnish",
    "he": "Hebrew",
    "no": "Norwegian",
    "da": "Danish",
}


def should_continue(state: AgentState) -> bool:
    """Function to check if we should continue the quiz."""
    if (state["current_question"] == state["quiz_length"]) & (
        state["current_stage"] == state["quiz_stages"]
    ):
        return "end"
    else:
        return "continue"


def play_sound_with_path(path: str, use_subprocess: bool = False) -> str:
    """Play a sound effect depending on the quiz outcome."""
    if use_subprocess:
        subprocess.Popen(["afplay", path])
    else:
        os.system("afplay " + path)


def play_sound(sound_type: str, use_subprocess: bool = False) -> str:
    """Play a sound effect depending on the quiz outcome."""
    main_path = "quiz_sounds/"
    sound_dict = {
        "correct": "tadaa.wav",
        "wrong": "wrong.mp3",
        "final_win": "win.wav",
        "final_lose": "wah_wah.wav",
    }
    sound = sound_dict[sound_type]

    if use_subprocess:
        subprocess.Popen(["afplay", main_path + sound])
    else:
        os.system("afplay " + main_path + sound)


def translate(text: str, target_lang: str = "es") -> str:
    """Translate text into the target language using OpenAI GPT."""
    if target_lang == "en":
        return text
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": f"You are a translator. Translate everything into {target_lang}.",
            },
            {"role": "user", "content": text},
        ],
    )
    return response.choices[0].message.content


def quiz_speak(text: str, lang: str = "en", use_subprocess: bool = False) -> str:
    """Speak the text you want to say using OpenAI, default is English, but you can change the language if you need it"""
    speechfile_path = Path("speech.mp3")
    with openai.audio.speech.with_streaming_response.create(
        model="gpt-4o-mini-tts",
        voice="nova",
        input=text,
        extra_query={
            "language": LANGUAGE_CODES.get(lang, "English"),
            "style": "sarcastic and cocky",
            "emotion": "happy",
            "pace": "medium",
        },
    ) as response:
        response.stream_to_file(speechfile_path)
    if use_subprocess:
        subprocess.Popen(["afplay", speechfile_path])
    else:
        os.system("afplay " + str(speechfile_path))
    return speechfile_path


pygame.mixer.init()


def play_background_music(
    file_path: str = "quiz_sounds/intro_music.wav", volume: float = 0.3
):
    """
    Play background music in a loop at a given volume.
    volume: float between 0.0 (mute) and 1.0 (máximo)
    """
    pygame.mixer.music.load(file_path)
    pygame.mixer.music.set_volume(volume)  # Establece volumen
    pygame.mixer.music.play(-1)  # Loop infinito


def delayed_music(
    file_path: str = "quiz_sounds/intro_music.wav",
    delay: float = 2,
    volume: float = 0.3,
):
    """Play music after a delay (non-blocking)"""

    def worker():
        time.sleep(delay)  # espera sin bloquear el hilo principal
        pygame.mixer.music.load(file_path)
        pygame.mixer.music.set_volume(volume)
        pygame.mixer.music.play(-1)  # loop infinito

    threading.Thread(target=worker, daemon=True).start()


def stop_background_music(fade_ms: int = 3000):
    pygame.mixer.music.fadeout(fade_ms)
