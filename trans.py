import streamlit as st
import whisper
import os
from gtts import gTTS
import google.generativeai as genai
from tempfile import NamedTemporaryFile

# --------------------------
# SET PAGE CONFIG
# --------------------------
st.set_page_config(page_title="AI Speech Translator", page_icon="🎙️", layout="wide")

# Custom CSS for UI Styling
st.markdown("""
    <style>
        body {
            background-color: #1e1e1e;
            color: #d4af37;
        }
        .stButton>button {
            background-color: #8b5e3c;
            color: white;
            border-radius: 8px;
            padding: 10px;
        }
        .stTextArea textarea, .stSelectbox select {
            background-color: #3e2723;
            color: white;
            border-radius: 8px;
        }
        .stAudio audio {
            width: 100%;
        }
        .css-1d391kg p {
            color: #d4af37;
        }
    </style>
""", unsafe_allow_html=True)

# --------------------------
# CONFIGURE API KEYS
# --------------------------
GEMINI_API_KEY = st.secrets["google"]["gemini_api_key"]
genai.configure(api_key=GEMINI_API_KEY)

# --------------------------
# LOAD WHISPER MODEL
# --------------------------
@st.cache_resource
def load_whisper():
    model = whisper.load_model("tiny", device="cpu")  # Use "tiny" for lower resource usage
    model.to("cpu")
    return model

whisper_model = load_whisper()

# --------------------------
# FUNCTION: Speech to Text (Whisper)
# --------------------------
def convert_speech_to_text(audio_file):
    try:
        with NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
            temp_file.write(audio_file.read())
            audio_path = temp_file.name
        
        result = whisper_model.transcribe(audio_path)
        os.unlink(audio_path)
        return result["text"]
    except Exception as e:
        return f"Error: {str(e)}"

# --------------------------
# FUNCTION: Translate Text (Gemini API)
# --------------------------
def translate_text(text, target_language):
    try:
        model = genai.GenerativeModel("gemini-1.5-flash")
        prompt = f"Translate this text to {target_language}: {text}"
        response = model.generate_content(prompt, request_options={"timeout": 10})
        return response.text
    except Exception as e:
        return f"Error: {str(e)}"

# --------------------------
# FUNCTION: Text to Speech (Using gTTS)
# --------------------------
def text_to_speech(text, language):
    try:
        tts = gTTS(text=text, lang=language)
        with NamedTemporaryFile(delete=False, suffix=".mp3") as temp_file:
            audio_path = temp_file.name
            tts.save(audio_path)
        return audio_path
    except Exception as e:
        return f"Error: {str(e)}"

# --------------------------
# STREAMLIT UI LAYOUT
# --------------------------
st.sidebar.markdown("<h1 style='text-align: center; color :solid black'>🎙️ AI Speech Translator</h1>", unsafe_allow_html=True)
st.sidebar.markdown("<p style='text-align: center;'>A simple tool for speech-to-text, text-to-speech, and translations(speech to Speech)* 🎧</p>", unsafe_allow_html=True)

language_map = {
    "fr": "French",
    "es": "Spanish",
    "de": "German",
    "hi": "Hindi",
    "ur": "Urdu",
    "en": "English",
    "it": "Italian",
    "nl": "Dutch",
    "pt": "Portuguese",
    "ru": "Russian",
    "zh": "Chinese",
    "ja": "Japanese",
    "ko": "Korean"
}

mode = st.sidebar.radio("Select Mode", ["Speech to Speech", "Text to Speech", "Speech to Text"], index=0)

if mode == "Speech to Speech":
    st.subheader("🎤 Upload Audio for Translation")
    audio_file = st.file_uploader("Upload an audio file", type=["wav", "mp3"])
    target_language = st.selectbox("Select Target Language", list(language_map.keys()), format_func=lambda x: language_map[x])
    
    if st.button("Translate Speech") and audio_file:
        text = convert_speech_to_text(audio_file)
        st.write(f"Original Text: {text}")
        translated_text = translate_text(text, language_map[target_language])
        st.write(f"Translated Text: {translated_text}")
        audio_path = text_to_speech(translated_text, target_language)
        
        if os.path.exists(audio_path):
            st.subheader("🎧 Translated Speech")
            st.audio(audio_path, format="audio/mp3")
            os.unlink(audio_path)
        else:
            st.error("Error generating speech")

elif mode == "Text to Speech":
    st.subheader("📝 Enter Text for Speech Synthesis")
    input_text = st.text_area("Enter text to convert to speech")
    language = st.selectbox("Select Language", list(language_map.keys()), format_func=lambda x: language_map[x])
    
    if st.button("Generate Speech") and input_text:
        translated_text = translate_text(input_text, language_map[language])
        audio_path = text_to_speech(translated_text, language)
        if os.path.exists(audio_path):
            st.subheader("🔊 Generated Speech")
            st.audio(audio_path, format="audio/mp3")
            os.unlink(audio_path)
        else:
            st.error("Error generating speech")

elif mode == "Speech to Text":
    st.subheader("🎙️ Upload Audio for Transcription")
    audio_file = st.file_uploader("Upload an audio file", type=["wav", "mp3"])
    language = st.selectbox("Select Language", list(language_map.keys()), format_func=lambda x: language_map[x])
    
    if st.button("Convert to Text") and audio_file:
        text = convert_speech_to_text(audio_file)
        translated_text = translate_text(text, language_map[language])
        st.subheader("📄 Transcribed Text")
        st.info(translated_text)

st.markdown("""
    <hr>
    <p style="text-align: center; color: #d4af37;">© 2025 AI Speech Translator | Built by Samiullah</p>
""", unsafe_allow_html=True)
