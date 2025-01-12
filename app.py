import streamlit as st
import streamlit.components.v1 as components
from streamlit.runtime.uploaded_file_manager import UploadedFile
import datetime
import fitz  # PyMuPDF
import clipboard
import firebase_admin
from firebase_admin import credentials, db, auth


import requests
# google.generativeai for text-based model calls
import google.generativeai as text_genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold

# google.genai for multimodal Gemini 2.0 (voice out)
import google.genai as voice_genai
from google.genai import types as voice_types

from openai import OpenAI

import base64
import wave
import asyncio
import contextlib

from io import BytesIO
from PIL import Image
from streamlit.runtime.secrets import AttrDict

# Chat input widget (text + audio)
from streamlit_chat_widget import chat_input_widget

# Floating container for placing the chat widget at the bottom
from streamlit_float import float_init

# 1) Implement memory in voice mode


def build_voice_conversation_context(last_n=10) -> str:
    """
    Builds a simple text-based conversation string from the last n user 
    and assistant messages to provide context in voice mode.
    """
    messages = st.session_state["current_chat"][-last_n * 2:]
    context_str = ""
    for msg in messages:
        role = msg.get("role")
        parts = msg.get("parts", [])
        text_content = []
        for part in parts:
            if "text" in part:
                text_content.append(part["text"])
        if text_content:
            if role == "user":
                context_str += "User: " + " ".join(text_content) + "\n"
            elif role == "assistant":
                context_str += "Assistant: " + " ".join(text_content) + "\n"
    return context_str


# 3) Add login for saving the information separately by user
#    and implement a guest login which turns off saving features
if "logged_in" not in st.session_state:
    st.session_state["logged_in"] = False
if "user_id" not in st.session_state:
    st.session_state["user_id"] = None


def login(email, password):
    """
    Login an existing user with email/password, or register if not found.
    """
    user: auth.UserRecord = None
    try:
        # Attempt to sign in using Firebase's REST API
        payload = {
            "email": email,
            "password": password,
            "returnSecureToken": True
        }
        api_key = st.secrets["firebase"]["WEB_API_KEY"]
        resp = requests.post(
            f"https://identitytoolkit.googleapis.com/v1/accounts:signInWithPassword?key={
                api_key}",
            data=payload
        )
        data = resp.json()
        if "idToken" in data:
            user = auth.get_user_by_email(email)
        else:
            raise Exception("Email or password is incorrect.")
    except Exception as e:
        st.sidebar.error(str(e))

    if user:
        st.session_state["logged_in"] = True
        st.session_state["user_id"] = user.uid
        st.session_state["display_name"] = user.display_name
        st.rerun()


def register(email, password, display_name=None):
    try:
        user = auth.create_user(
            email=email, password=password, display_name=display_name)
        st.sidebar.success("Registration successful. Please login.")
        return user

    except auth.EmailAlreadyExistsError:
        st.sidebar.error("Email already registered.")
        return None
    except Exception as e:
        st.sidebar.error(f"Registration failed: {e}")
        return None


def logout():
    st.session_state["logged_in"] = False
    st.session_state["user_id"] = None
    st.session_state["display_name"] = None
    st.rerun()
###################################################################
# --------------- END NEW/UPDATED CODE BLOCK ABOVE --------------- #
###################################################################


def inializate():
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


inializate()


# ------------------ UPDATED TO SAVE/LOAD PER USER ---------------- #
def save_chat_to_firebase(chat_name, messages, timestamp):
    """
    Saves the chat under the logged-in user's reference, 
    or does nothing if user is a guest or not logged in.
    """
    if chat_name and st.session_state["logged_in"] and st.session_state["user_id"] != "guest":
        user_id = st.session_state["user_id"]
        user_chats_ref = db.reference(f'users/{user_id}/chats')
        user_chats_ref.child(chat_name).set({
            'timestamp': timestamp,
            'messages': messages
        })


def load_chats_from_firebase():
    """
    Returns the chats of the logged-in user, or empty dict if not logged in or if guest.
    """
    if st.session_state["logged_in"] and st.session_state["user_id"] != "guest":
        user_id = st.session_state["user_id"]
        user_chats_ref = db.reference(f'users/{user_id}/chats')
        return user_chats_ref.get() or {}
    else:
        return {}

# ----------------------------------------------------------------- #


parameters_ref = db.reference('parameters')
system_ref = db.reference('system_instruction')


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


# ------------------ LOAD STATE AFTER USER LOGIN ------------------ #
if "saved_chats" not in st.session_state:
    st.session_state["saved_chats"] = {}

if "generation_config" not in st.session_state:
    st.session_state["generation_config"] = load_parameters_from_firebase(
        default_generation_config)

if "system_instruction" not in st.session_state:
    st.session_state["system_instruction"] = load_system_instruction_from_firebase()

if "current_chat" not in st.session_state:
    st.session_state["current_chat"] = []

# Avoid re-uploading the same image in a single turn
if "file_processed" not in st.session_state:
    st.session_state["file_processed"] = False

# avoid infinite reruns when the user input an audio
if "audio_uploaded" not in st.session_state:
    st.session_state["audio_uploaded"] = False

# To avoid infinite reruns, track the last input
if "last_user_input" not in st.session_state:
    st.session_state["last_user_input"] = None
# --------------------------------------------------------------- #


###################################################################
# 4) Add possibility of adding video or audio as input in the same
#    component that handles images (for text/image mode).
###################################################################


st.sidebar.header("Login / User Session")
if not st.session_state["logged_in"]:
    login_mode = st.sidebar.radio(
        "Select Mode", ["Login", "Register", "Guest"])

    if login_mode == "Login":
        email = st.sidebar.text_input("Email")
        password = st.sidebar.text_input("Password", type="password")
        if st.sidebar.button("Login"):
            login(email, password)

    elif login_mode == "Register":
        name = st.sidebar.text_input("Name")
        email = st.sidebar.text_input("Email")
        password = st.sidebar.text_input("Password", type="password")
        if st.sidebar.button("Register"):
            register(email, password, name)

    elif login_mode == "Guest":
        if st.sidebar.button("Continue as Guest"):
            st.session_state["logged_in"] = True
            st.session_state["user_id"] = "guest"
            st.session_state["display_name"] = "Guest"
            st.rerun()
else:
    st.sidebar.success(f"Logged in as {st.session_state['display_name']}")
    if st.sidebar.button("Logout"):
        logout()

st.sidebar.header("Configuration")

conversation_mode = st.sidebar.radio(
    "Select conversation mode:",
    ["Text + Media", "Voice Conversation with Gemini"]
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

if conversation_mode == "Voice Conversation with Gemini":
    available_voices = ["Puck", "Charon", "Kore", "Fenrir", "Aoede"]
    selected_voice = st.sidebar.selectbox("Select voice", available_voices)
    st.session_state["generation_config"]["voice_name"] = selected_voice
else:
    st.session_state["generation_config"].pop("voice_name", None)


if st.sidebar.button("Save Configuration"):
    save_parameters_to_firebase(st.session_state["generation_config"])
    save_system_instruction_to_firebase(st.session_state["system_instruction"])
    st.sidebar.success("Configuration saved to Firebase.")


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

st.session_state["video_uploaded"] = None
st.session_state["audio_uploaded"] = None

system_instruction_default = """
You are a multilingual AI assistant capable of adapting to various tasks as requested by the user. 
Respond in the same language as the user's prompt. Ensure your answers are formatted in Markdown, 
including LaTeX for math, code snippets, and images where applicable always with no exception. 
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

# --------------- LOAD CHATS AFTER LOGIN --------------
if st.session_state["logged_in"]:
    st.session_state["saved_chats"] = load_chats_from_firebase()
# -----------------------------------------------------

st.session_state['chat_name'] = st.sidebar.text_input(
    "Chat Name", placeholder="Enter a name to save this chat"
)

if st.sidebar.button("Save Current Chat"):
    if st.session_state['chat_name']:
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        st.session_state["saved_chats"][st.session_state['chat_name']] = {
            "timestamp": timestamp,
            "messages": st.session_state["current_chat"],
        }
        save_chat_to_firebase(
            st.session_state['chat_name'], st.session_state["current_chat"], timestamp)
        st.sidebar.success(f"Chat '{st.session_state['chat_name']}' saved!")
    else:
        st.sidebar.warning("Please provide a name.")

if st.sidebar.button("Clear Current Chat"):
    st.session_state["current_chat"] = []
    st.rerun()

if not 'selected_chat' in st.session_state:
    st.session_state['selected_chat'] = "Select"


def update_chat():

    if st.session_state['current_chat'] != []:
        st.session_state['current_chat'] = []
    selected_chat = st.session_state['selected_chat']
    if selected_chat != "Select":
        st.session_state["current_chat"] = st.session_state["saved_chats"][selected_chat]["messages"]
        st.sidebar.success(f"Chat '{selected_chat}' loaded.")


saved_chat_names = list(st.session_state["saved_chats"].keys(
)) if st.session_state["logged_in"] else []

st.session_state['selected_chat'] = st.sidebar.selectbox(
    "Load a Saved Chat",
    ["Select"] + saved_chat_names,
)
if st.session_state['selected_chat'] != "Select":
    if st.sidebar.button("Load Chat"):
        update_chat()
    if st.sidebar.button("Delete Chat"):
        # delete from firebase
        chat_name = st.session_state['selected_chat']
        user_id = st.session_state["user_id"]
        user_chats_ref = db.reference(f'users/{user_id}/chats')
        user_chats_ref.child(chat_name).delete()
        # delete from session state
        del st.session_state["saved_chats"][chat_name]
        st.sidebar.success(f"Chat '{chat_name}' deleted.")
        st.session_state['selected_chat'] = "Select"
        st.session_state['current_chat'] = []
        st.rerun()
    if st.sidebar.button("Save chat"):
        chat_name = st.session_state['selected_chat']
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        st.session_state["saved_chats"][chat_name] = {
            "timestamp": timestamp,
            "messages": st.session_state["current_chat"],
        }
        save_chat_to_firebase(
            chat_name, st.session_state["current_chat"], timestamp)
        st.sidebar.success(f"Chat '{chat_name}' saved!")


#

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
        st.sidebar.button("Add to context", on_click=st.session_state["current_chat"].append(
            {"role": "user", "parts": [{"text": text}]}))
if st.session_state['selected_chat'] != "Select":
    st.header(st.session_state['selected_chat'])
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
                elif mime_type.startswith("video/"):
                    try:
                        video_bytes = base64.b64decode(data_value) if isinstance(
                            data_value, str) else data_value
                        st.video(video_bytes)
                    except:
                        pass


def generate_text_response(user_message: str):
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


def generate_text_response_with_audio_input(audio_file: UploadedFile):
    st.session_state["current_chat"].append({
        "role": "user",
        "parts": [
            {"mime_type": audio_file.type, "data": base64.b64encode(
                audio_file.read()).decode("utf-8")}
        ]
    })

    chat = text_model.start_chat(history=st.session_state["current_chat"])

    content: text_genai.types.ContentType = [
        {
            "text": " "
        }

    ]
    response = chat.send_message(
        content,
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


voice_client = voice_genai.Client(http_options={
                                  'api_version': 'v1alpha'}, api_key=st.secrets["google"]["GOOGLE_API_KEY"])
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
    """
    Sends the final_prompt (including conversation context for memory in voice mode)
    to the Gemini voice model, and writes the streaming audio to a WAV file.
    """
    # 1) Implement memory in voice mode by building conversation context:
    conversation_context = build_voice_conversation_context(10)
    final_prompt = conversation_context + "\nUser: " + user_prompt

    voice_cfg = None
    if st.session_state["generation_config"].get("voice_name"):
        voice_cfg = voice_types.VoiceConfig(
            prebuilt_voice_config=voice_types.PrebuiltVoiceConfig(
                voice_name=st.session_state["generation_config"]["voice_name"]
            )
        )

    system_instr = {
        "parts": [{"text": st.session_state["system_instruction"]}]
    }

    config = voice_types.LiveConnectConfig(
        generation_config=voice_types.GenerationConfig(
            temperature=st.session_state["generation_config"].get(
                "temperature", 1.0),
            top_p=st.session_state["generation_config"].get("top_p", 0.95),
            max_output_tokens=st.session_state["generation_config"].get(
                "max_output_tokens", 8192),
        ),
        speech_config=voice_types.SpeechConfig(
            voice_config=voice_cfg
        ) if voice_cfg else None,
        response_modalities=["AUDIO"],
        system_instruction=system_instr,
    )

    file_name = "gemini_voice_output.wav"
    async with voice_client.aio.live.connect(model=VOICE_MODEL, config=config) as session:
        await session.send(final_prompt, end_of_turn=True)
        turn = session.receive()
        with wave_file(file_name) as wav:
            async for _, response in async_enumerate(turn):
                if response.data:
                    wav.writeframes(response.data)

    with open(file_name, "rb") as f:
        return f.read()


def generate_voice_response(prompt: str) -> bytes:
    return asyncio.run(do_voice_generation(prompt))


# 2) Fix transcription in mobile (basic check / try-except fallback)
def transcribe_with_whisper(audio_bytes: bytes) -> str:
    with open("temp_user_audio.wav", "wb") as tmp:
        tmp.write(audio_bytes)
    with open("temp_user_audio.wav", "rb") as audio_file:
        try:
            result = openai_client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file,

            )
            return result.text
        except Exception as e:
            st.warning(f"Whisper transcription failed: {e}")
            return " "  # fallback empty


float_init()
widget_container = st.container()
with widget_container:
    user_input = st.chat_input()

widget_container.float(
    "display: flex; align-items: center; justify-content: center; "
    "overflow:  visible; flex-direction: column; "
    "position: fixed; bottom: 15px;"
)

# 6) Animate the container with the messages
st.markdown(
    """
    <style>
    .stChatMessage {
      animation: fadeInUp 0.5s ease-in-out;
    }
    @keyframes fadeInUp {
        0% {
            opacity: 0;
            transform: translate3d(0, 30%, 0);
        }
        100% {
            opacity: 1;
            transform: none;
        }
    }
    #MainMenu {visibility: hidden;}
    header {visibility: hidden;}
    footer {visibility: hidden;}
    
    [data-testid="stChatInput"] {
        position: fixed;
        bottom: 55px ; /* Ajusta este valor según tus necesidades */
        z-index: 1000; /* Garantiza que el input esté por encima de otros elementos */
    }

    
    </style>
    """,
    unsafe_allow_html=True
)


###################################################################
# 4) We unify image/video/audio in the same uploader for text mode.
###################################################################

st.sidebar.header("Media Upload")
uploaded_media = st.sidebar.file_uploader(
    "Upload image/video/audio (for text+image mode)",
    type=["png", "jpg", "jpeg", "mp4", "mov", "avi", "wav", "mp3", "m4a"]
)

# record audio

recoreder_audio = st.sidebar.audio_input("Record audio")
# accept the audio

if recoreder_audio is not None:
    st.session_state["audio_uploaded"] = recoreder_audio
    if st.sidebar.button("Send audio"):
        if conversation_mode == "Text + Media":
            with st.chat_message("user"):
                st.audio(st.session_state["audio_uploaded"])
            assistant_text, image = generate_text_response_with_audio_input(
                st.session_state["audio_uploaded"])
            if assistant_text and image is None:
                st.session_state["current_chat"].append({
                    "role": "assistant",
                    "parts": [
                        {"text": assistant_text},
                    ]
                })
            elif assistant_text and image is not None:
                # If there's also an image to return
                buffer = BytesIO()
                image.convert("RGB").save(buffer, format="JPEG")
                image_data = buffer.getvalue()
                image_b64 = base64.b64encode(image_data).decode("utf-8")

                st.session_state["current_chat"].append({
                    "role": "assistant",
                    "parts": [
                        {"text": assistant_text},
                        {"mime_type": "image/jpeg", "data": image_b64},
                    ]
                })

            st.rerun()

        elif conversation_mode == "Voice Conversation with Gemini":
            try:
                st.session_state["current_chat"].append({
                    "role": "user",
                    "parts": [{"mime_type": "audio/wav", "data": base64.b64encode(
                        st.session_state["audio_uploaded"].read()).decode("utf-8")}]
                })
                with st.chat_message("user"):
                    st.audio(st.session_state["audio_uploaded"])
                parts_to_store = []
                audio_text = transcribe_with_whisper(
                    st.session_state["audio_uploaded"].read())
                assistant_audio_bytes = generate_voice_response(
                    audio_text) if audio_text else None
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


if uploaded_media and not st.session_state.get("file_processed", False):

    if conversation_mode == "Text + Media":
        try:
            file_type = uploaded_media.type
            if file_type.startswith("image/"):
                image = Image.open(uploaded_media)
                st.sidebar.image(image, use_container_width=True)

                if file_type == "image/png":
                    image = image.convert("RGB")
                buffer = BytesIO()
                image.convert("RGB").save(buffer, format="JPEG")

                base64_data = base64.b64encode(
                    buffer.getvalue()).decode("utf-8")
                st.session_state["current_chat"].append({
                    "role": "user",
                    "parts": [
                        {"mime_type": "image/jpeg", "data": base64_data}
                    ]
                })

            elif file_type.startswith("video/"):

                # Convert the video to base64
                video_bytes = uploaded_media.read()
                video_b64 = base64.b64encode(video_bytes).decode("utf-8")
                st.sidebar.video(video_bytes)
                st.session_state["current_chat"].append({
                    "role": "user",
                    "parts": [
                        {"mime_type": file_type, "data": video_b64}
                    ]
                })
                st.session_state["video_uploaded"] = uploaded_media

                st.sidebar.info(f"Video uploaded as {file_type}")

            elif file_type.startswith("audio/"):
                # Convert the audio to base64
                audio_bytes = uploaded_media.read()
                audio_b64 = base64.b64encode(audio_bytes).decode("utf-8")
                st.sidebar.audio(audio_bytes)
                st.session_state["current_chat"].append({
                    "role": "user",
                    "parts": [
                        {"mime_type": file_type, "data": audio_b64}
                    ]
                })
                st.session_state["audio_uploaded"] = uploaded_media

        except Exception as e:
            st.sidebar.error(f"Could not encode media: {e}")
    else:
        st.sidebar.info(
            "You are in voice mode. Media can be stored but will not be processed for voice generation."
        )
    st.session_state["file_processed"] = True
    # remove uploaded media to avoid reprocessing


# If the file dont exist, reset the state
if not uploaded_media:
    st.session_state["file_processed"] = False


# Process new input exactly once, then rerun.
if user_input and user_input != st.session_state["last_user_input"]:
    st.session_state["last_user_input"] = user_input
    user_text_prompt = None

    # 5) Ensure user response is always appended, even if assistant doesn't respond

    if user_input is not None:
        text_val = user_input.strip()
        if text_val:
            user_text_prompt = text_val
            st.session_state["current_chat"].append({
                "role": "user",
                "parts": [{"text": text_val}]
            })

    if user_text_prompt and conversation_mode == "Text + Media":
        try:
            with st.chat_message("user"):
                st.markdown(user_text_prompt)

            assistant_text, image = generate_text_response(user_text_prompt)

            if assistant_text and image is None:
                st.session_state["current_chat"].append({
                    "role": "assistant",
                    "parts": [
                        {"text": assistant_text},
                    ]
                })
            elif assistant_text and image is not None:
                # If there's also an image to return
                buffer = BytesIO()
                image.convert("RGB").save(buffer, format="JPEG")

                image_data = buffer.getvalue()
                image_b64 = base64.b64encode(image_data).decode("utf-8")

                st.session_state["current_chat"].append({
                    "role": "assistant",
                    "parts": [
                        {"text": assistant_text},
                        {"mime_type": "image/jpeg", "data": image_b64},
                    ]
                })
        except Exception as e:
            st.session_state["current_chat"].append({
                "role": "assistant",
                "parts": [{"text": f"Error: {e}"}]
            })

    elif user_text_prompt and conversation_mode == "Voice Conversation with Gemini":
        try:

            with st.chat_message("user"):
                st.markdown(user_text_prompt)

            parts_to_store = []
            # Voice generation with memory
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


