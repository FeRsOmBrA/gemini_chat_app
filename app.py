# -*- coding: utf-8 -*-
"""
Complete Streamlit app with the requested improvements:

1) **Two Conversation Modes** (select via sidebar):
   - "Text + Images Only" (no voice).
   - "Voice Mode" (Gemini 2.0 + Whisper for transcription + voice output).
2) **Fixed 'Part object is not subscriptable' Error** by referencing `part.text`,
   `part.executable_code`, etc. directly instead of `part["text"]`.
3) **Chat Input Widget** is placed at the bottom of the screen, with chat messages
   above it. We use `streamlit_float` and some CSS to ensure correct positioning.
4) **No Deprecated OpenAI usage**: Instead we demonstrate the snippet with
   `from openai import OpenAI` and `client.audio.transcriptions.create` to do Whisper.
   (If you prefer the `openai` standard library usage, adapt accordingly.)

Additional notes:
- Make sure you have installed:
    pip install --upgrade streamlit-chat-widget
    pip install streamlit-float
    pip install google-generativeai
    pip install google-genai
    pip install PyMuPDF
    pip install Pillow
    pip install openai
    
- Ensure your `st.secrets` includes:
    st.secrets["openai"]["OPENAI_API_KEY"]
    st.secrets["google"]["GOOGLE_API_KEY"]
    st.secrets["firebase"]["my_project_settings"]
      (and other relevant keys)

- This example stores text & audio from the assistant in Firebase 
  alongside user messages. Chat messages are replayed from `st.session_state`
  at the top of the page.

- If you need additional styling adjustments, you can expand the CSS 
  in the `st.markdown(...)` block below.

"""

import streamlit as st
import streamlit.components.v1 as components
import datetime
import fitz  # PyMuPDF
import clipboard
import firebase_admin
from firebase_admin import credentials, db

# google.generativeai for text-based model calls
import google.generativeai as text_genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold

# google.genai for multimodal Gemini 2.0 (voice out)
import google.genai as voice_genai
from google.genai import types as voice_types

# We'll demonstrate the 'from openai import OpenAI' approach (no deprecated usage).
from openai import OpenAI

import base64
import wave
import asyncio
import contextlib
import logging
import os
import json

from io import BytesIO
from PIL import Image
from streamlit.runtime.secrets import AttrDict

# Chat input widget (text + audio)
from streamlit_chat_widget import chat_input_widget

# Floating container for placing the chat widget at the bottom
from streamlit_float import float_init


def initialize_firebase():
    global openai_client
    if not firebase_admin._apps:
        firebase_credentials: AttrDict
        firebase_credentials = st.secrets['firebase']['my_project_settings']

        cred = credentials.Certificate(firebase_credentials.to_dict())
        firebase_admin.initialize_app(cred, {
            'databaseURL': st.secrets['firebase']['dburl']
        })

    # Configure Google Generative AI (text)
    text_genai.configure(api_key=st.secrets['google']["GOOGLE_API_KEY"])

    # Configure OpenAI API client (Whisper)
    openai_client = OpenAI(api_key=st.secrets["openai"]["OPENAI_API_KEY"])


initialize_firebase()

# Firebase references
chats_ref = db.reference('chats')
parameters_ref = db.reference('parameters')
system_ref = db.reference('system_instruction')


def save_chat_to_firebase(chat_name, messages, timestamp):
    if chat_name:
        chats_ref.child(chat_name).set({
            'timestamp': timestamp,
            'messages': messages
        })


def load_chats_from_firebase():
    return chats_ref.get() or {}


def save_parameters_to_firebase(parameters):
    parameters_ref.set(parameters)


def load_parameters_from_firebase(default_config):
    return parameters_ref.get() or default_config


def save_system_instruction_to_firebase(instruction):
    system_ref.set(instruction)


def load_system_instruction_from_firebase():
    return system_ref.get() or ""


def copy_manual(text):
    clipboard.copy(text)


def copy_button(text, key=None):
    """Renders a small copy-to-clipboard button next to the text."""
    copy_html = f"""
    <style>
        .copy-container_{key} {{
            display: inline-flex;
            position: relative;
            align-items: center;
            margin-left: 8px;
        }}
        .copy-button_{key} {{
            cursor: pointer;
            border: none;
            background: transparent;
            font-size: 0.8em;
            color: #555;
            transition: color 0.3s ease, transform 0.3s ease;
            padding: 4px;
            border-radius: 50%;
        }}
        .copy-button_{key}:hover {{
            color: #000;
            transform: scale(1.1);
        }}
        .copy-button_{key}:active {{
            transform: scale(0.95);
        }}
        .tooltip_{key} {{
            visibility: hidden;
            width: 80px;
            background-color: #333;
            color: #fff;
            text-align: center;
            border-radius: 6px;
            padding: 3px 0;
            position: absolute;
            bottom: 120%;
            left: 50%;
            transform: translateX(-50%);
            opacity: 0;
            transition: opacity 0.2s;
            font-size: 0.7em;
            z-index: 1;
        }}
        .copy-button_{key}:hover + .tooltip_{key} {{
            visibility: visible;
            opacity: 1;
        }}
    </style>
    <div class="copy-container_{key}">
        <textarea id="textToCopy_{key}" style="position: absolute; left: -9999px;">{text}</textarea>
        <button class="copy-button_{key}" onclick="copyToClipboard_{key}()" title="Copy to Clipboard">
            ⧉
        </button>
        <span class="tooltip_{key}">Copied!</span>
    </div>
    <script>
        function copyToClipboard_{key}() {{
            var copyText = document.getElementById("textToCopy_{key}");
            copyText.select();
            copyText.setSelectionRange(0, 99999);
            navigator.clipboard.writeText(copyText.value).then(function(){{}}, function(err) {{
                console.error('Error copying text: ', err);
            }});
        }}
    </script>
    """
    components.html(copy_html, height=40, scrolling=False)


def resize_image(image: Image.Image, scale_factor: float = 2.0) -> Image.Image:
    """Helper to enlarge images."""
    new_size = (int(image.width * scale_factor),
                int(image.height * scale_factor))
    resized_image = image.resize(new_size, Image.Resampling.BICUBIC)
    return resized_image


def extract_text_with_page_numbers(pdf_file):
    """Extract text from PDF with page numbers."""
    try:
        pdf_document = fitz.open(stream=pdf_file.read(), filetype="pdf")
        text_paginated = ""
        for page_num in range(len(pdf_document)):
            page = pdf_document.load_page(page_num)
            text_paginated += f"--- Page {page_num + 1} ---\n"
            text_paginated += page.get_text()
            text_paginated += "\n\n"
        return text_paginated
    except Exception as e:
        st.error(f"An error occurred while processing the PDF: {e}")
        return None


def upload_to_gemini(file_obj, mime_type=None):
    """Uploads a file to Gemini (via google.generativeai)."""
    try:
        file = text_genai.upload_file(file_obj, mime_type=mime_type)
        return file
    except Exception:
        return None


default_generation_config = {
    "temperature": 1.0,
    "top_p": 0.95,
    "top_k": 40,
    "max_output_tokens": 8192,
    "response_mime_type": "text/plain",
}

if "saved_chats" not in st.session_state:
    st.session_state["saved_chats"] = load_chats_from_firebase()

if "generation_config" not in st.session_state:
    st.session_state["generation_config"] = load_parameters_from_firebase(
        default_generation_config
    )

if "system_instruction" not in st.session_state:
    st.session_state["system_instruction"] = load_system_instruction_from_firebase()

if "current_chat" not in st.session_state:
    st.session_state["current_chat"] = []

# Avoid re-uploading the same image in a single turn
if "image_processed" not in st.session_state:
    st.session_state["image_processed"] = False

# To avoid infinite reruns, track the last input
if "last_user_input" not in st.session_state:
    st.session_state["last_user_input"] = None

st.sidebar.header("Configuration")

conversation_mode = st.sidebar.radio(
    "Select conversation mode:",
    ["Text + Images Only", "Voice Conversation with Gemini"]
)

models = []
try:
    for m in text_genai.list_models():
        model_name = m.name.replace("models/", "")
        models.append(model_name)
except Exception as e:
    st.sidebar.error(f"Error listing models: {e}")

if models:
    try:
        default_model = (
            "gemini-2.0-flash-exp"
            if "gemini-2.0-flash-exp" in models
            else ("gemini-1.5-pro-latest" if "gemini-1.5-pro-latest" in models else models[0])
        )
        selected_model = st.sidebar.selectbox(
            "Select the text model",
            models,
            index=models.index(default_model)
        )
    except ValueError:
        selected_model = models[0]
else:
    selected_model = "gemini-1.5-pro-latest"

st.sidebar.subheader("Generation Parameters")
st.session_state["generation_config"]["temperature"] = st.sidebar.slider(
    "Temperature",
    0.1,
    2.0,
    value=st.session_state["generation_config"].get("temperature", 1.0),
    step=0.1
)
st.session_state["generation_config"]["top_p"] = st.sidebar.slider(
    "Top P",
    0.1,
    1.0,
    value=st.session_state["generation_config"].get("top_p", 0.95),
    step=0.05
)
st.session_state["generation_config"]["top_k"] = st.sidebar.slider(
    "Top K",
    1,
    100,
    value=st.session_state["generation_config"].get("top_k", 40),
    step=1
)
st.session_state["generation_config"]["max_output_tokens"] = st.sidebar.slider(
    "Max Output Tokens",
    100,
    8192,
    value=st.session_state["generation_config"].get("max_output_tokens", 8192),
    step=100
)

st.session_state["system_instruction"] = st.sidebar.text_area(
    "System Instruction",
    st.session_state["system_instruction"]
)

st.session_state['image_uploaded'] = None


if st.sidebar.button("Save Configuration"):
    save_parameters_to_firebase(st.session_state["generation_config"])
    save_system_instruction_to_firebase(st.session_state["system_instruction"])
    st.sidebar.success("Configuration saved to Firebase.")

system_instruction_default = """
You are a multilingual AI assistant capable of adapting to various tasks as requested by the user. 
Respond in the same language as the user's prompt. Ensure your answers are formatted in Markdown, 
including LaTeX for math, code snippets, and images where applicable. 
Never prepend messages with 'Assistant:' or 'User:'. 
You can answer with code, execute code, and show results.
"""
if st.session_state["system_instruction"].strip():
    combined_instruction = system_instruction_default + \
        "\n" + st.session_state["system_instruction"]
else:
    combined_instruction = system_instruction_default

try:
    text_model = text_genai.GenerativeModel(
        model_name=selected_model,
        generation_config=st.session_state["generation_config"],
        system_instruction=combined_instruction,
        tools="code_execution",
    )
except Exception as e:
    st.sidebar.error(f"Error initializing text model: {e}")

st.sidebar.header("Chat Management")
chat_name = st.sidebar.text_input(
    "Chat Name", placeholder="Enter a name to save this chat")

if st.sidebar.button("Save Current Chat"):
    if chat_name:
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        st.session_state["saved_chats"][chat_name] = {
            "timestamp": timestamp,
            "messages": st.session_state["current_chat"],
        }
        save_chat_to_firebase(
            chat_name, st.session_state["current_chat"], timestamp)
        st.sidebar.success(f"Chat '{chat_name}' saved!")
    else:
        st.sidebar.warning("Please provide a name.")

if st.sidebar.button("Clear Current Chat"):
    st.session_state["current_chat"] = []
    st.rerun()

saved_chat_names = list(st.session_state["saved_chats"].keys())
selected_chat = st.sidebar.selectbox(
    "Load a Saved Chat",
    ["Select"] + saved_chat_names
)
if selected_chat != "Select":
    st.session_state["current_chat"] = st.session_state["saved_chats"][selected_chat]["messages"]
    st.sidebar.success(f"Chat '{selected_chat}' loaded.")

st.sidebar.header("PDF to Text Converter")
uploaded_pdf = st.sidebar.file_uploader("Upload a PDF", type="pdf")
if uploaded_pdf is not None:
    text = extract_text_with_page_numbers(uploaded_pdf)
    if text:
        st.sidebar.success("PDF successfully converted.")
        select_encoding = st.sidebar.selectbox(
            "Select text encoding", ["utf-8", "latin-1", "windows-1252"]
        )
        output_path = uploaded_pdf.name.replace(".pdf", "_output.txt")
        with open(output_path, "w", encoding=select_encoding, errors="replace") as output_file:
            output_file.write(text)

        st.sidebar.download_button(
            "Save Extracted Text",
            text,
            f"{uploaded_pdf.name.replace('.pdf', '_output.txt')}",
            mime="text/plain"
        )
        st.sidebar.button("Copy Extracted Text",
                          on_click=copy_manual, args=(text,))
        if text == clipboard.paste():
            st.sidebar.success("Text copied to clipboard!")

if selected_chat != "Select":
    st.header(selected_chat)
else:
    st.header("Chat Session")

for idx, message in enumerate(st.session_state["current_chat"]):
    role = message.get("role", "user")
    with st.chat_message(role):
        parts = message.get("parts", [])
        for part in parts:
            text_content = getattr(part, "text", None)
            code_content = getattr(part, "executable_code", None)
            inline_data = getattr(part, "inline_data", None)
            mime_type = getattr(part, "mime_type", None)
            data_value = getattr(part, "data", None)

            if text_content is None and isinstance(part, dict) and "text" in part:
                text_content = part["text"]
            if code_content is None and isinstance(part, dict) and "executable_code" in part:
                code_content = part["executable_code"]
            if inline_data is None and isinstance(part, dict) and "inline_data" in part:
                inline_data = part["inline_data"]
            if mime_type is None and isinstance(part, dict) and "mime_type" in part:
                mime_type = part["mime_type"]
            if data_value is None and isinstance(part, dict) and "data" in part:
                data_value = part["data"]

            if text_content:
                st.markdown(text_content)
                if role == "assistant":
                    copy_button(text_content, key=f"copy_{idx}")

            if code_content:
                lang = code_content["language"]["name"].lower(
                ) if "language" in code_content else "python"
                code = code_content["code"] if "code" in code_content else ""
                st.markdown(f"```{lang}\n{code}\n```")

            if inline_data:
                try:
                    image_to_display = Image.open(BytesIO(inline_data.data))
                    st.image(image_to_display)
                except:
                    pass

            if mime_type and data_value:
                if mime_type.startswith("image/"):
                    try:
                        image_bytes = base64.b64decode(data_value) if isinstance(
                            data_value, str) else data_value
                        st.image(image_bytes, use_container_width=True)
                    except:
                        pass
                elif mime_type.startswith("audio/"):
                    try:
                        audio_bytes = base64.b64decode(data_value) if isinstance(
                            data_value, str) else data_value
                        st.audio(audio_bytes)
                    except:
                        pass


def generate_text_response(user_message: str) -> str:
    chat = text_model.start_chat(history=st.session_state["current_chat"])
    response = chat.send_message(
        user_message,
        safety_settings={
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
        }
    )
    image_to_display = None
    if not response or len(response.candidates) == 0:
        return ""
    full_text = ""
    for p in response.parts:
        if p.text:
            full_text += p.text + "\n"
        if p.executable_code:
            lang = p.executable_code.language.name.lower()
            code = p.executable_code.code
            full_text += f"```{lang}\n{code}\n```\n"
        if p.inline_data:
            try:
                image_to_display = Image.open(BytesIO(p.inline_data.data))

            except:
                pass
    return full_text.strip(), image_to_display


voice_client = voice_genai.Client(http_options={'api_version': 'v1alpha'})
VOICE_MODEL = "gemini-2.0-flash-exp"


@contextlib.contextmanager
def wave_file(filename, channels=1, rate=24000, sample_width=2):
    with wave.open(filename, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(sample_width)
        wf.setframerate(rate)
        yield wf


async def async_enumerate(aiter):
    i = 0
    async for x in aiter:
        yield i, x
        i += 1


async def do_voice_generation(user_prompt: str) -> bytes:
    config = {"generation_config": {"response_modalities": ["AUDIO"]}}
    file_name = "gemini_voice_output.wav"
    async with voice_client.aio.live.connect(model=VOICE_MODEL, config=config) as session:
        await session.send(user_prompt, end_of_turn=True)
        turn = session.receive()
        with wave_file(file_name) as wav:
            async for _, response in async_enumerate(turn):
                if response.data:
                    wav.writeframes(response.data)
    with open(file_name, "rb") as f:
        return f.read()


def generate_voice_response(prompt: str) -> bytes:
    return asyncio.run(do_voice_generation(prompt))


def transcribe_with_whisper(audio_bytes: bytes) -> str:
    with open("temp_user_audio.wav", "wb") as tmp:
        tmp.write(audio_bytes)
    with open("temp_user_audio.wav", "rb") as audio_file:
        try:
            result = openai_client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file
            )
            return result.text
        except Exception as e:
            st.warning(f"Whisper transcription failed: {e}")
            return ""



float_init()
widget_container = st.container()
with widget_container:
    user_input = chat_input_widget()

widget_container.float(
    "display: flex; align-items: center; justify-content: center; "
    "overflow:  visible; flex-direction: column; "
    "position: fixed; bottom: 15px;"
)

st.markdown("""
<style>
.stCustomComponentV1{
    max-height: 200px;
}
</style>
""", unsafe_allow_html=True)

# Process new input exactly once, then rerun.
if user_input and user_input != st.session_state["last_user_input"]:
    st.session_state["last_user_input"] = user_input
    user_text_prompt = None

    if "text" in user_input:
        text_val = user_input["text"].strip()
        if text_val:
            user_text_prompt = text_val
            st.session_state["current_chat"].append({
                "role": "user",
                "parts": [{"text": text_val}]
            })

    if "audioFile" in user_input:
        audio_bytes = bytes(user_input["audioFile"])
        transcript = transcribe_with_whisper(audio_bytes)
        if transcript:
            user_text_prompt = transcript
            st.session_state["current_chat"].append({
                "role": "user",
                "parts": [{"text": transcript}]
            })

    if user_text_prompt and conversation_mode == "Text + Images Only":
        try:
            with st.chat_message("user"):
                st.markdown(user_text_prompt)
                if st.session_state["image_uploaded"] is not None:
                    st.image(st.session_state["image_uploaded"])

            assistant_text, image = generate_text_response(user_text_prompt)
            
            if assistant_text and image is None:
                st.session_state["current_chat"].append({
                    "role": "assistant",
                    "parts": [
                        {"text": assistant_text},


                    ]
                })
            elif assistant_text and image is not None:
                st.session_state["current_chat"].append({
                    "role": "assistant",
                    "parts": [
                        {"text": assistant_text},
                        {"mime_type": "image/jpeg", "data": image}
                    ]
                })
        except Exception as e:
            st.session_state["current_chat"].append({
                "role": "assistant",
                "parts": [{"text": f"Error: {e}"}]
            })

    elif user_text_prompt and conversation_mode == "Voice Conversation with Gemini":
        try:
            
            assistant_audio_bytes = generate_voice_response(
                user_text_prompt) if user_text_prompt else None
            audio_b64 = base64.b64encode(
                assistant_audio_bytes).decode() if assistant_audio_bytes else None

            if audio_b64:
                parts_to_store.append(
                    {"mime_type": "audio/wav", "data": audio_b64})

            st.session_state["current_chat"].append({
                "role": "assistant",
                "parts": parts_to_store
            })
        except Exception as e:
            st.session_state["current_chat"].append({
                "role": "assistant",
                "parts": [{"text": f"Error generating assistant response: {e}"}]
            })

    st.rerun()

uploaded_image = st.sidebar.file_uploader(
    "Upload an image (PNG/JPG/JPEG)",
    type=["png", "jpg", "jpeg"],
    key="uploaded_image"
)
if uploaded_image and not st.session_state.get("image_processed", False):
    
    image = Image.open(uploaded_image)
    st.session_state['image_uploaded'] = image
    st.sidebar.image(image, use_container_width=True)
    if conversation_mode == "Text + Images Only":
        try:
            if uploaded_image.type == "image/png":
                image = image.convert("RGB")
            buffer = BytesIO()
            image.save(buffer, format="JPEG")
            base64_data = base64.b64encode(buffer.getvalue()).decode("utf-8")
            st.session_state["current_chat"].append({
                "role": "user",
                "parts": [
                    {"mime_type": "image/jpeg", "data": base64_data}
                ]
            })
        except Exception as e:
            st.sidebar.error(f"Could not encode image: {e}")
    else:
        st.sidebar.info(
            "You are in voice mode. Images can still be stored but not processed for voice."
        )
    st.session_state["image_processed"] = True

if st.session_state.get("image_processed", False):
    st.session_state["image_processed"] = False
