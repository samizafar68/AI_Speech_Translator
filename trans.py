import gradio as gr
import whisper
import requests
import os
from gtts import gTTS  # Import gTTS for Text-to-Speech
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# --------------------------
# CONFIGURE API KEYS (Replace with yours)
# --------------------------
GEMINI_API_KEY = os.getenv("gemini")


# --------------------------
# LOAD WHISPER MODEL ONCE
# --------------------------
whisper_model = whisper.load_model("large")

# --------------------------
# FUNCTION: Speech to Text (Whisper)
# --------------------------
def convert_speech_to_text(audio_file):
    try:
        audio_path = "temp_audio.wav"
        with open(audio_path, "wb") as f:
            f.write(audio_file.read())
        
        result = whisper_model.transcribe(audio_path)
        os.remove(audio_path)  # Clean up temporary file
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
# GRADIO INTERFACE
# --------------------------

def speech_to_speech(audio_file, target_language):
    text = convert_speech_to_text(audio_file)
    translated_text = translate_text(text, target_language)
    audio_path = text_to_speech(translated_text, target_language)
    return audio_path

def text_to_speech_fn(input_text, language):
    translated_text = translate_text(input_text, language)
    audio_path = text_to_speech(translated_text, language)
    return audio_path

def speech_to_text_fn(audio_file, language):
    text = convert_speech_to_text(audio_file)
    translated_text = translate_text(text, language)
    return translated_text

# Define Gradio Interface
iface = gr.Interface(
    fn=speech_to_speech,
    inputs=[
        gr.inputs.Audio(source="microphone", type="file"),
        gr.inputs.Dropdown(["fr", "es", "de", "hi", "ur", "en", "it", "nl", "pt", "ru", "zh", "ja", "ko"], label="Target Language")
    ],
    outputs=gr.outputs.Audio(label="Translated Speech"),
    live=True,
    title="AI Speech Translator"
)

iface.add_component(
    gr.Interface(
        fn=text_to_speech_fn,
        inputs=[gr.inputs.Textbox(label="Text to Convert to Speech"), gr.inputs.Dropdown(["fr", "es", "de", "hi", "ur", "en", "it", "nl", "pt", "ru", "zh", "ja", "ko"], label="Language")],
        outputs=gr.outputs.Audio(label="Generated Speech")
    ),
    "Text to Speech"
)

iface.add_component(
    gr.Interface(
        fn=speech_to_text_fn,
        inputs=[gr.inputs.Audio(source="microphone", type="file"), gr.inputs.Dropdown(["fr", "es", "de", "hi", "ur", "en", "it", "nl", "pt", "ru", "zh", "ja", "ko"], label="Language")],
        outputs=gr.outputs.Textbox(label="Transcribed Text")
    ),
    "Speech to Text"
)

# Launch the interface
iface.launch()
