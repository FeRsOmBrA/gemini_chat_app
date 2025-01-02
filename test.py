from openai import OpenAI
import streamlit as st
client = OpenAI(api_key=st.secrets["openai"]["OPENAI_API_KEY"])
# supported formats mp3, mp4, mpeg, mpga, m4a, wav, and webm.
audio_file = open("audio.mp3", "rb")
transcription = client.audio.transcriptions.create(
    model="whisper-1",
    file=audio_file
)

print(transcription.text)
