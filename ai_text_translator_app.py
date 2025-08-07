import gradio as gr
import speech_recognition as sr
import pyttsx3
from gtts import gTTS
from transformers import pipeline
import re
import os
import tempfile
from datetime import datetime
import threading
from langdetect import detect

class CreativeTranslatorApp:
    def __init__(self):
        # Supported languages with their ISO 639-1 codes and Helsinki-NLP model prefixes
        self.languages = {
            "English": "en",
            "Spanish": "es",
            "French": "fr",
            "German": "de",
            "Italian": "it",
            "Portuguese": "pt",
            "Russian": "ru",
            "Chinese": "zh",
            "Japanese": "ja",
            "Arabic": "ar",
            "Hindi": "hi",
            "Dutch": "nl"
        }
        self.translation_pipelines = {}
        self.model_cache = {}  # Cache loaded models to avoid reloading
        try:
            # Initialize sentiment analyzer
            self.sentiment_analyzer = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
        except Exception as e:
            print(f"Error initializing sentiment analyzer: {e}")
            self.sentiment_analyzer = None

        # Initialize speech components
        self.recognizer = sr.Recognizer()
        self.tts_engine = pyttsx3.init()
        self.tts_engine.setProperty('rate', 150)
        self.voices = self.tts_engine.getProperty('voices')
        self.translation_history = []
        self.is_recording = False
        self.recording_thread = None
        self.audio_data = None

    def load_translation_pipeline(self, source_lang, target_lang):
        """Dynamically load translation pipeline for the given language pair."""
        if source_lang == target_lang:
            return None  # No translation needed for same language
        model_key = f"{source_lang}-{target_lang}"
        if model_key not in self.model_cache:
            try:
                model_name = f"Helsinki-NLP/opus-mt-{source_lang}-{target_lang}"
                self.model_cache[model_key] = pipeline("translation", model=model_name)
            except Exception as e:
                print(f"Error loading model {model_name}: {e}")
                return None
        return self.model_cache[model_key]

    def start_recording(self):
        if not self.is_recording:
            self.is_recording = True
            self.recording_thread = threading.Thread(target=self._record_audio, daemon=True)
            self.recording_thread.start()
            return "Recording started..."
        return "Already recording."

    def stop_recording(self):
        if self.is_recording:
            self.is_recording = False
            self.recording_thread.join()
            try:
                text = self.recognizer.recognize_google(self.audio_data, language="en-US")
                return text, "Speech recognized successfully!"
            except sr.UnknownValueError:
                return "", "Could not understand audio."
            except Exception as e:
                return "", f"Error: {str(e)}"
        return "", "Not recording."

    def _record_audio(self):
        try:
            with sr.Microphone() as source:
                self.recognizer.adjust_for_ambient_noise(source, duration=1)
                self.audio_data = self.recognizer.listen(source, timeout=5, phrase_time_limit=30)
        except sr.WaitTimeoutError:
            self.audio_data = None
            self.is_recording = False

    def apply_mexican_slang(self, text):
        slang_map = {
            "hola": "qué onda",
            "amigo": "carnal",
            "bien": "chido",
            "casa": "cantón",
            "dinero": "lana",
            "rápido": "de volada"
        }
        for formal, slang in slang_map.items():
            text = re.sub(r'\b' + formal + r'\b', slang, text, flags=re.IGNORECASE)
        return text

    def apply_casual_slang(self, text, lang):
        if lang == "en":
            slang_map = {
                "hello": "yo",
                "friend": "buddy",
                "good": "cool",
                "house": "pad",
                "money": "cash",
                "quickly": "pronto"
            }
        else:
            slang_map = {
                "hola": "qué tal",
                "amigo": "compa",
                "bien": "padre",
                "casa": "criba",
                "dinero": "varos",
                "rápido": "pronto"
            }
        for formal, slang in slang_map.items():
            text = re.sub(r'\b' + formal + r'\b', slang, text, flags=re.IGNORECASE)
        return text

    def apply_shakespearean_style(self, text):
        shakespeare_map = {
            "you": "thou",
            "are": "art",
            "hello": "hail",
            "good": "noble",
            "friend": "good sir",
            "quickly": "with haste"
        }
        for modern, old in shakespeare_map.items():
            text = re.sub(r'\b' + modern + r'\b', old, text, flags=re.IGNORECASE)
        return text

    def prose_to_poetry(self, text):
        sentences = text.split('.')
        poem = ""
        for i, sentence in enumerate(sentences):
            if sentence.strip():
                poem += sentence.strip() + ",\n"
                if i % 2 == 1:
                    poem += "\n"
        return poem.rstrip(',\n')

    def adjust_tone_by_sentiment(self, text):
        if self.sentiment_analyzer:
            sentiment = self.sentiment_analyzer(text)[0]
            if sentiment['label'] == 'POSITIVE':
                return text + " 😊"
            elif sentiment['label'] == 'NEGATIVE':
                return text + " 😔"
        return text

    def translate(self, input_text, target_lang, dialect, sentiment_adjust):
        if not input_text:
            return "Please enter text to translate.", ""

        try:
            # Auto-detect source language
            source_lang_code = detect(input_text)
            source_lang = next((lang for lang, code in self.languages.items() if code == source_lang_code), "English")
            target_lang_code = self.languages.get(target_lang, "en")

            # Handle special modes
            if dialect in ["Formal to Casual", "Prose to Poetry"]:
                if dialect == "Formal to Casual":
                    translated = self.apply_casual_slang(input_text, "en")
                else:
                    translated = self.prose_to_poetry(input_text)
            else:
                # Check if source and target languages are the same
                if source_lang_code == target_lang_code:
                    translated = input_text
                else:
                    # Load translation pipeline
                    pipeline = self.load_translation_pipeline(source_lang_code, target_lang_code)
                    if pipeline is None:
                        raise Exception(f"No translation model available for {source_lang} to {target_lang}.")
                    result = pipeline(input_text)
                    translated = result[0].get('translation_text', 'Translation failed')

                # Apply dialect transformations
                if target_lang == "Spanish" and dialect == "Mexican Spanish":
                    translated = self.apply_mexican_slang(translated)
                elif dialect == "Casual Slang":
                    lang = "es" if target_lang == "Spanish" else "en"
                    translated = self.apply_casual_slang(translated, lang)
                elif dialect == "Shakespearean" and target_lang == "English":
                    translated = self.apply_shakespearean_style(translated)

            if sentiment_adjust:
                translated = self.adjust_tone_by_sentiment(translated)

            # Save to history
            self.translation_history.append({
                "input": input_text,
                "output": translated,
                "mode": f"{source_lang} to {target_lang}",
                "dialect": dialect,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            })

            return translated, "Translation completed!"
        except Exception as e:
            return f"Translation failed: {str(e)}", f"Error: {str(e)}"

    def text_to_speech(self, text, voice_type):
        if not text:
            return None, "No text to play."

        try:
            if voice_type == "Robotic (pyttsx3)":
                self.tts_engine.setProperty('voice', self.voices[0].id if self.voices else None)
                self.tts_engine.say(text)
                self.tts_engine.runAndWait()
                return None, "Audio playback completed!"
            else:
                tts = gTTS(text=text, lang='en' if voice_type == "Natural (English)" else 'es', tld='com' if voice_type == "Natural (English)" else 'es')
                temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp3')
                tts.save(temp_file.name)
                return temp_file.name, "Audio file generated!"
        except Exception as e:
            return None, f"Audio error: {str(e)}"

    def get_history(self):
        if not self.translation_history:
            return "No translation history available."
        history_text = ""
        for entry in self.translation_history:
            history_text += f"[{entry['timestamp']}] {entry['mode']} ({entry['dialect']}):\nInput: {entry['input']}\nOutput: {entry['output']}\n\n"
        return history_text

    def add_custom_slang(self, formal, slang, lang):
        if lang == "English":
            self.apply_casual_slang = lambda text, _: self.apply_casual_slang(text, "en").replace(formal, slang)
        else:
            self.apply_casual_slang = lambda text, _: self.apply_casual_slang(text, "es").replace(formal, slang)
        return f"Added custom slang: '{formal}' -> '{slang}' for {lang}"

def create_ui():
    app = CreativeTranslatorApp()
    
    with gr.Blocks(theme=gr.themes.Soft(), css="""
        .gradio-container {background: linear-gradient(135deg, #1e1e2f, #2a2a3b);}
        .gr-button {background-color: #ff6f61 !important; color: white !important; border-radius: 10px !important;}
        .gr-button:hover {background-color: #ff8a80 !important;}
        .gr-textbox, .gr-dropdown {background-color: #2a2a3b !important; color: white !important; border-radius: 10px !important;}
        .gr-label {color: #ffffff !important; font-size: 16px !important;}
    """) as demo:
        gr.Markdown("# 🌍 GLOBALINGO", elem_classes="header")
        gr.Markdown("Translate with style, sentiment, and voice! Unique features beyond standard translators.", elem_classes="subheader")

        with gr.Row():
            with gr.Column(scale=2):
                input_text = gr.Textbox(lines=5, label="Input Text", placeholder="Enter text or use microphone...")
                with gr.Row():
                    start_button = gr.Button("🎙️ Start Recording")
                    stop_button = gr.Button("🛑 Stop Recording")
                audio_output = gr.Textbox(label="Speech Recognition Status", interactive=False)

            with gr.Column(scale=1):
                target_lang = gr.Dropdown(
                    choices=list(app.languages.keys()),
                    label="Target Language",
                    value="Spanish"
                )
                dialect = gr.Dropdown(
                    choices=["Standard", "Mexican Spanish", "Casual Slang", "Shakespearean", "Formal to Casual", "Prose to Poetry"],
                    label="Dialect/Style",
                    value="Standard"
                )
                sentiment_adjust = gr.Checkbox(label="Adjust Tone by Sentiment", value=False)

        translate_button = gr.Button("Translate")
        output_text = gr.Textbox(lines=5, label="Translated Text", interactive=False)
        status = gr.Textbox(label="Status", interactive=False)

        with gr.Row():
            voice_type = gr.Dropdown(
                choices=["Robotic (pyttsx3)", "Natural (English)", "Natural (Spanish)"],
                label="Voice Type",
                value="Natural (English)"
            )
            play_button = gr.Button("🔊 Play Output")
            audio_player = gr.Audio(label="Audio Output", interactive=False)

        with gr.Accordion("Translation History"):
            history_button = gr.Button("View History")
            history_output = gr.Textbox(label="Translation History", lines=5, interactive=False)

        with gr.Accordion("Custom Slang Dictionary"):
            formal_word = gr.Textbox(label="Formal Word", placeholder="e.g., hello")
            slang_word = gr.Textbox(label="Slang Word", placeholder="e.g., yo")
            slang_lang = gr.Dropdown(choices=["English", "Spanish"], label="Language", value="English")
            add_slang_button = gr.Button("Add Custom Slang")
            slang_status = gr.Textbox(label="Slang Status", interactive=False)

        # Event handlers
        start_button.click(
            fn=app.start_recording,
            outputs=audio_output
        )
        stop_button.click(
            fn=app.stop_recording,
            outputs=[input_text, audio_output]
        )
        translate_button.click(
            fn=app.translate,
            inputs=[input_text, target_lang, dialect, sentiment_adjust],
            outputs=[output_text, status]
        )
        play_button.click(
            fn=app.text_to_speech,
            inputs=[output_text, voice_type],
            outputs=[audio_player, status]
        )
        history_button.click(
            fn=app.get_history,
            outputs=history_output
        )
        add_slang_button.click(
            fn=app.add_custom_slang,
            inputs=[formal_word, slang_word, slang_lang],
            outputs=slang_status
        )

    return demo

if __name__ == "__main__":
    demo = create_ui()
    demo.launch()
