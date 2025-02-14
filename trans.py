
import streamlit as st
import whisper
import requests
import os
from gtts import gTTS
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# --------------------------
# CONFIGURE API KEYS
# --------------------------
GEMINI_API_KEY = os.getenv("gemini")

# --------------------------
# LOAD WHISPER MODEL ONCE
# --------------------------
@st.cache_resource
def load_whisper_model():
    return whisper.load_model("small")

whisper_model = load_whisper_model()

# --------------------------
# FUNCTION: Speech to Text (Whisper)
# --------------------------
def convert_speech_to_text(audio_path):
    try:
        result = whisper_model.transcribe(audio_path)
        return result["text"]
    except Exception as e:
        return f"Error: {str(e)}"

# --------------------------
# FUNCTION: Translate Text (Gemini API)
# --------------------------
def translate_text(text, target_language):
    try:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent?key={GEMINI_API_KEY}"
        headers = {"Content-Type": "application/json"}
        data = {
            "contents": [{"parts": [{"text": f"Translate this to {target_language}: {text}"}]}]
        }

        response = requests.post(url, json=data, headers=headers)

        if response.status_code == 200:
            response_json = response.json()
            translated_text = response_json.get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", "")
            return translated_text if translated_text else "Translation not found"
        else:
            return f"API Error: {response.status_code} - {response.text}"

    except Exception as e:
        return f"Error: {str(e)}"

# --------------------------
# FUNCTION: Text to Speech (Using gTTS)
# --------------------------
def text_to_speech(text, language):
    try:
        tts = gTTS(text=text, lang=language)
        audio_path = "output.mp3"
        tts.save(audio_path)
        return audio_path
    except Exception as e:
        return f"Error: {str(e)}"

# --------------------------
# STREAMLIT UI
# --------------------------
st.set_page_config(page_title="AI Speech Translator", page_icon="🔊", layout="centered")

st.title("🔊 AI Speech Translator")
st.write("Convert Speech to Text, Translate it, and Generate Speech in Another Language.")

# Sidebar for language selection
target_language = st.sidebar.selectbox("Select Target Language:", ["fr", "es", "de", "hi", "ur", "en", "it", "nl", "pt", "ru", "zh", "ja", "ko"])

# Tabs for different functionalities
tab1, tab2, tab3 = st.tabs(["🎤 Speech to Speech", "📝 Text to Speech", "🎙 Speech to Text"])

# --------------------------
# TAB 1: Speech to Speech
# --------------------------
with tab1:
    st.header("🎤 Speech to Speech Translation")
    audio_file = st.file_uploader("Upload an audio file", type=["wav", "mp3", "m4a"])

    if st.button("Convert & Translate"):
        if audio_file is not None:
            with open("temp_audio.wav", "wb") as f:
                f.write(audio_file.read())

            text = convert_speech_to_text("temp_audio.wav")
            translated_text = translate_text(text, target_language)
            audio_path = text_to_speech(translated_text, target_language)

            st.success(f"Transcribed Text: {text}")
            st.success(f"Translated Text: {translated_text}")
            st.audio(audio_path, format="audio/mp3")
        else:
            st.error("Please upload an audio file.")

# --------------------------
# TAB 2: Text to Speech
# --------------------------
with tab2:
    st.header("📝 Text to Speech Conversion")
    input_text = st.text_area("Enter text to convert:")
    
    if st.button("Generate Speech"):
        if input_text:
            translated_text = translate_text(input_text, target_language)
            audio_path = text_to_speech(translated_text, target_language)

            st.success(f"Translated Text: {translated_text}")
            st.audio(audio_path, format="audio/mp3")
        else:
            st.error("Please enter some text.")

# --------------------------
# TAB 3: Speech to Text
# --------------------------
with tab3:
    st.header("🎙 Speech to Text")
    audio_file = st.file_uploader("Upload an audio file for transcription", type=["wav", "mp3", "m4a"], key="stt")

    if st.button("Transcribe"):
        if audio_file is not None:
            with open("temp_audio.wav", "wb") as f:
                f.write(audio_file.read())

            text = convert_speech_to_text("temp_audio.wav")
            translated_text = translate_text(text, target_language)

            st.success(f"Transcribed Text: {text}")
            st.success(f"Translated Text: {translated_text}")
        else:
            st.error("Please upload an audio file.")

st.write("🔹 Developed by Sami - AI Speech Translator powered by Whisper & Gemini.")
