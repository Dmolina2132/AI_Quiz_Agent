from langchain_core.tools import tool
from gtts import gTTS
import os
from pathlib import Path
import openai
from dotenv import load_dotenv
import subprocess
from utils import LANGUAGE_CODES, play_sound_with_path

# Here we define all the tools that will be used by the agent
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")


@tool
def speak(text: str, lang: str = "en"):
    """Speak the test you want to say using pytt, default is English, but you can change the language if you need it"""
    # Generate audio with gTTS
    tts = gTTS(text=text, lang=lang)
    tts.save("saludo.mp3")  # Save the file

    # Play the audio
    # On macOS
    os.system("afplay saludo.mp3")


@tool
def speak_pytt(text: str, lang: str = "en"):
    """Speak the test you want to say using pytt, default is English, but you can change the language if you need it"""
    # Generate audio with gTTS
    tts = gTTS(text=text, lang=lang)
    tts.save("saludo.mp3")  # Save the file

    # Play the audio
    # On macOS
    os.system("afplay saludo.mp3")


@tool
def speak_openai(text: str, lang: str = "en"):
    """Speak the test you want to say using OpenAI, default is English, but you can change the language if you need it"""
    speechfile_path = Path("speech.mp3")
    with openai.audio.speech.with_streaming_response.create(
        model="gpt-4o-mini-tts",
        voice="nova",
        input=text,
        extra_query={
            "language": LANGUAGE_CODES.get(lang, "English"),
            "style": "friendly and enthusiastic",
            "emotion": "friendly",
            "pace": "medium",
        },
    ) as response:
        response.stream_to_file(speechfile_path)
    subprocess.Popen(["afplay", speechfile_path])


@tool
def public_reaction(reaction: str):
    """Play a sound effect to show your reaction as the public
    Available input options:
    - cheer
    - applause
    - boo
    - laugh
    - shock
    """
    folder_path = "public_sounds/"
    path_mapping = {
        "cheer": "cheering.wav",
        "applause": "applause.wav",
        "boo": "boo.wav",
        "laugh": "laugh.mp3",
        "shock": "shock.wav",
    }
    play_sound_with_path(
        folder_path + path_mapping.get(reaction, "cheering.wav"), use_subprocess=True
    )
