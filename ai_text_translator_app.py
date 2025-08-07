import gradio as gr
from deep_translator import GoogleTranslator
from gtts import gTTS
import tempfile
import speech_recognition as sr

# Language Dictionary
language_dict = {
    'Arabic': 'ar', 'Bengali': 'bn', 'Chinese (Simplified)': 'zh-CN', 'Chinese (Traditional)': 'zh-TW',
    'Dutch': 'nl', 'English': 'en', 'French': 'fr', 'German': 'de', 'Gujarati': 'gu', 'Hindi': 'hi',
    'Italian': 'it', 'Japanese': 'ja', 'Kannada': 'kn', 'Korean': 'ko', 'Malayalam': 'ml',
    'Marathi': 'mr', 'Nepali': 'ne', 'Persian': 'fa', 'Portuguese': 'pt', 'Punjabi': 'pa',
    'Russian': 'ru', 'Spanish': 'es', 'Tamil': 'ta', 'Telugu': 'te', 'Turkish': 'tr',
    'Ukrainian': 'uk', 'Urdu': 'ur', 'Vietnamese': 'vi'
}

language_names = list(language_dict.keys())

# Translate Function
def translate_text(text, target_lang):
    target_code = language_dict[target_lang]
    translated = GoogleTranslator(source='auto', target=target_code).translate(text)
    return translated

# Text to Speech Function
def text_to_speech(text, lang_name):
    lang_code = language_dict.get(lang_name, 'en')
    tts = gTTS(text=text, lang=lang_code)
    temp_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3").name
    tts.save(temp_path)
    return temp_path

# Full Gradio App
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("<h1 style='text-align: center;'>🌐 AI Voice Translator</h1>")

    with gr.Row():
        with gr.Column():
          mic_input = gr.Audio(type="filepath", label="🎤 Speak (Speech-to-Text)")
          text_input = gr.Textbox(lines=3, label="✍️ Or Type Text")
          target_lang = gr.Dropdown(choices=language_names, label="🌎 Target Language", value="Hindi")
          translate_btn = gr.Button("🚀 Translate")

        with gr.Column():
            translation_output = gr.Textbox(label="📝 Translated Text", lines=3, interactive=False)
            audio_output = gr.Audio(label="🔊 Hear Translation", autoplay=True)

    def process(audio, typed_text, lang):
        input_text = ""

        if audio is not None:
            recognizer = sr.Recognizer()
            try:
                with sr.AudioFile(audio) as source:
                    audio_data = recognizer.record(source)
                    input_text = recognizer.recognize_google(audio_data)
            except Exception as e:
                input_text = "Speech recognition failed. Try again."
        elif typed_text.strip():
            input_text = typed_text
        else:
            return "Please provide input.", None

        translated = translate_text(input_text, lang)
        speech_file = text_to_speech(translated, lang)
        return translated, speech_file


    translate_btn.click(
        fn=process,
        inputs=[mic_input, text_input, target_lang],
        outputs=[translation_output, audio_output]
    )
from langdetect import detect

def detect_language(text):
    try:
        lang_code = detect(text)
        for name, code in language_dict.items():
            if code == lang_code:
                return name
        return lang_code
    except:
        return "Unknown"
detected_lang_output = gr.Textbox(label="🌐 Detected Language", interactive=False)
sentiment_output = gr.Textbox(label="🧠 Sentiment", interactive=False)


demo.launch()
