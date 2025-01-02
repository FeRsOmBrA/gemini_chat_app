import streamlit as st
import streamlit.components.v1 as components
import datetime
import fitz  # PyMuPDF
import clipboard
import firebase_admin
from firebase_admin import credentials, db
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold
from io import BytesIO
from PIL import Image
import base64
from st_multimodal_chatinput import multimodal_chatinput

from streamlit.runtime.secrets import AttrDict
# -------------------------------
# Initialize Firebase
# -------------------------------


def initialize():
    if not firebase_admin._apps:
        firebase_credentials: AttrDict
        firebase_credentials = st.secrets['firebase']['my_project_settings']
      

        cred = credentials.Certificate(firebase_credentials.to_dict())

        firebase_admin.initialize_app(cred, {
            'databaseURL': 'https://gemini-app-dbbdb-default-rtdb.firebaseio.com/'
        })
        genai.configure(api_key=st.secrets['google']["GOOGLE_API_KEY"])


initialize()

# Database references
chats_ref = db.reference('chats')
parameters_ref = db.reference('parameters')
system_ref = db.reference('system_instruction')

# Firebase interaction functions


def save_chat_to_firebase(chat_name, messages, timestamp):
    """
    Save the entire 'messages' list (which can include base64 images)
    directly into Firebase.
    """
    if chat_name:
        chats_ref.child(chat_name).set({
            'timestamp': timestamp,
            'messages': messages
        })


def load_chats_from_firebase():
    """
    Loads all chats from Firebase.
    Returns a dictionary keyed by chat_name with 'timestamp' and 'messages'.
    """
    return chats_ref.get() or {}


def save_parameters_to_firebase(parameters):
    parameters_ref.set(parameters)


def load_parameters_from_firebase(default_config):
    return parameters_ref.get() or default_config


def save_system_instruction_to_firebase(instruction):
    system_ref.set(instruction)


def load_system_instruction_from_firebase():
    return system_ref.get() or ""

# -------------------------------
# Utility Functions
# -------------------------------


def copy_manual(text):
    clipboard.copy(text)


def copy_button(text, key=None):
    copy_html = f"""
    <style>
        .copy-container_{key} {{
            display: flex;
            position: relative;
            align-items: center;
            justify-content: flex-start;
        }}
        .copy-button_{key} {{
            cursor: pointer;
            border: none;
            background: transparent;
            font-size: 0.8em;
            color: #555;
            transition: color 0.3s ease, transform 0.3s ease;
            padding: 8px;
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
            width: 120px;
            background-color: #333;
            color: #fff;
            text-align: center;
            border-radius: 6px;
            padding: 5px 0;
            position: absolute;
            bottom: 125%;
            left: 50%;
            transform: translateX(-50%);
            opacity: 0;
            transition: opacity 0.3s;
            font-size: 0.8em;
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
            <svg xmlns="http://www.w3.org/2000/svg" height="24" width="24" viewBox="0 0 24 24" fill="currentColor">
                <path d="M16 1H4c-1.1 0-2 .9-2 2v14h2V3h12V1zm3 4H8c-1.1 0-2 .9-2 2v14c0
                1.1.9 2 2 2h11c1.1 0 2-.9
                2-2V7c0-1.1-.9-2-2-2zm0
                16H8V7h11v14z"/>
            </svg>
        </button>
        <div class="tooltip_{key}">Copied!</div>
        <script>
            function copyToClipboard_{key}() {{
                var copyText = document.getElementById("textToCopy_{key}");
                copyText.select();
                copyText.setSelectionRange(0, 99999);
                navigator.clipboard.writeText(copyText.value).then(function() {{}}, function(err) {{
                    console.error('Error copying text: ', err);
                }});
            }}
        </script>
    </div>
    """
    components.html(copy_html, height=60, scrolling=False)


def resize_image(image: Image.Image, scale_factor: float = 2.0) -> Image.Image:
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
    """Uploads the given file to Gemini."""
    try:
        file = genai.upload_file(file_obj, mime_type=mime_type)
        return file
    except Exception as e:
        return None


# -------------------------------
# Load Session State
# -------------------------------
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

# We'll use this flag to control image uploading only once per user turn
if "image_processed" not in st.session_state:
    st.session_state["image_processed"] = False

# -------------------------------
# Sidebar Configuration
# -------------------------------
st.sidebar.header("Configuration")

# Model Selector
models = []
try:
    for m in genai.list_models():
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
            "Select the model",
            models,
            index=models.index(default_model)
        )
    except ValueError:
        selected_model = models[0]
else:
    selected_model = "gemini-1.5-pro-latest"

# Generation parameters
st.sidebar.subheader("Generation Parameters")
st.session_state["generation_config"]["temperature"] = st.sidebar.slider(
    "Temperature", min_value=0.1, max_value=2.0,
    value=st.session_state["generation_config"].get("temperature", 1.0), step=0.1
)
st.session_state["generation_config"]["top_p"] = st.sidebar.slider(
    "Top P", min_value=0.1, max_value=1.0,
    value=st.session_state["generation_config"].get("top_p", 0.95), step=0.05
)
st.session_state["generation_config"]["top_k"] = st.sidebar.slider(
    "Top K", min_value=1, max_value=100,
    value=st.session_state["generation_config"].get("top_k", 40), step=1
)
st.session_state["generation_config"]["max_output_tokens"] = st.sidebar.slider(
    "Max Output Tokens", min_value=100, max_value=8192,
    value=st.session_state["generation_config"].get("max_output_tokens", 8192), step=100
)

# System instruction (user-provided)
st.session_state["system_instruction"] = st.sidebar.text_area(
    "System Instruction", st.session_state["system_instruction"]
)

# Button to save config and system instruction
if st.sidebar.button("Save Configuration"):
    save_parameters_to_firebase(st.session_state["generation_config"])
    save_system_instruction_to_firebase(st.session_state["system_instruction"])
    st.sidebar.success("Configuration successfully saved to Firebase.")

# Combine default instruction with user-provided instruction
system_instruction_default = """
You are a multilingual AI assistant capable of adapting to various tasks as requested by the user. Respond in the same language as the user's prompt. Ensure your answers are correctly formatted in Markdown, including LaTeX for math, code snippets, and images where applicable. Also, you will never add things like "Assistant: " or "User: " to the chat, so you should not add them either. Also, you are capable of answering with code, executing the code, and showing the results.
"""

if st.session_state["system_instruction"].strip():
    combined_instruction = system_instruction_default + \
        "\n" + st.session_state["system_instruction"]
else:
    combined_instruction = system_instruction_default

# Initialize the model
try:
    model = genai.GenerativeModel(
        model_name=selected_model,
        generation_config=st.session_state["generation_config"],
        system_instruction=combined_instruction,
        tools="code_execution",
    )
except Exception as e:
    st.sidebar.error(f"Error initializing model: {e}")

# -------------------------------
# Chat Management
# -------------------------------
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
        st.sidebar.success(f"Chat '{chat_name}' saved to Firebase!")
    else:
        st.sidebar.warning("Please provide a name to save the chat.")

if st.sidebar.button("Clear Current Chat"):
    st.session_state["current_chat"] = []
    st.rerun()

# Load a saved chat
saved_chat_names = list(st.session_state["saved_chats"].keys())
selected_chat = st.sidebar.selectbox(
    "Load a Saved Chat", ["Select"] + saved_chat_names)
if selected_chat != "Select":
    st.session_state["current_chat"] = st.session_state["saved_chats"][selected_chat]["messages"]
    st.sidebar.success(f"Chat '{selected_chat}' loaded from Firebase.")

# -------------------------------
# PDF to Text Converter
# -------------------------------
st.sidebar.header("PDF to Text Converter")
uploaded_pdf = st.sidebar.file_uploader("Upload a PDF file here", type="pdf")

if uploaded_pdf is not None:
    text = extract_text_with_page_numbers(uploaded_pdf)
    if text:
        st.sidebar.success("PDF successfully converted.")
        select_encoding = st.sidebar.selectbox(
            "Select text encoding", ["utf-8", "latin-1", "windows-1252"])
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

# -------------------------------
# Main Interface
# -------------------------------

# If a chat is selected, show it as header
if selected_chat != "Select":
    st.header(selected_chat)

# --------------------------------------------
# Display the current chat from session state
# --------------------------------------------
for idx, message in enumerate(st.session_state["current_chat"]):
    role = message.get("role", "user")
    with st.chat_message(role):
        parts = message.get("parts", [])
        for part in parts:
            # --- IMAGE PERSISTENCE CHANGES ---
            # 1. If this part is a base64 image, display it from base64
            if "data" in part:
                try:
                    mime_type = part.get("mime_type", "image/jpeg")
                    base64_data = part.get("data", "")
                    if base64_data:
                        image_bytes = base64.b64decode(base64_data)
                        image = Image.open(BytesIO(image_bytes))
                        st.image(image,
                                 use_container_width=True)
                except Exception as e:
                    st.warning(f"Could not load image from base64: {e}")

            # 2. If this part is text
            elif "text" in part:
                st.markdown(part["text"])
                if role == "model":
                    copy_button(part["text"], key=idx)

            # 3. If this part is the genai.types.File / or has 'uri' + 'mime_type'
            elif isinstance(part, dict) and "uri" in part and "mime_type" in part:
                mime_type = part.get("mime_type", "")
                uri = part.get("uri", "")
                display_name = part.get("display_name", "Uploaded File")
                if mime_type.startswith("image/"):
                    st.image(uri, caption=display_name, use_column_width=True)
                else:
                    st.markdown(f"[Download {display_name}]({uri})")
            elif "executable_code" in part:
                st.markdown(f"```{part['executable_code']['language']['name'].lower()}\n{
                            part['executable_code']['code']}\n```")

            elif "inline_data" in part:
                inline_data = part["inline_data"]
                try:
                    image_to_display = Image.open(BytesIO(inline_data.data))
                    st.image(image_to_display)
                except Exception as e:
                    st.warning(f"Failed to load inline image: {e}")


# -------------------------------
# Generate Response Function
# -------------------------------


def generate_response(user_message):
    try:
        chat = model.start_chat(history=st.session_state["current_chat"])
        response = chat.send_message(user_message,             safety_settings={
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
        })

        assistant_response_text = ""
        image_to_display = None

        # ------------------------------
        # Step 4: Process Response
        # ------------------------------
        with st.chat_message("assistant"):

            if response and len(response.candidates) > 0:
                parts = response.parts

                for part in parts:
                    if "text" in part:
                        assistant_response_text += part.text + "\n"

                    if "executable_code" in part:
                        language = part.executable_code.language.name.lower()
                        code = part.executable_code.code
                        assistant_response_text += f"```{
                            language}\n{code}\n```\n"

                    if "inline_data" in part:
                        try:
                            image_to_display = Image.open(
                                BytesIO(part.inline_data.data))
                        except Exception as e:
                            st.warning(f"Failed to load inline image: {e}")

            # Render the text
            st.markdown(assistant_response_text)

            # Render the image if present
            if image_to_display:
                st.image(image_to_display)

            # Provide a copy button for the response text
            copy_button(assistant_response_text, key=len(
                st.session_state.get("current_chat", [])))

        # ------------------------------
        # Step 5: Update Chat History
        # ------------------------------
        st.session_state.setdefault("current_chat", []).append({
            "role": "assistant",
            "parts": [{"text": assistant_response_text}]
        })

    except Exception as e:
        # Display error message in the chat
        with st.chat_message("assistant"):
            st.markdown(f"Sorry, an error occurred: **{e}**")


# -------------------------------
# Image Uploader and User Input
# -------------------------------
uploaded_image = st.sidebar.file_uploader(
    "Upload an image (PNG, JPG, JPEG)",
    type=["png", "jpg", "jpeg"],
    key="uploaded_image"
)

# Display the uploaded image preview (optional)
if uploaded_image is not None:
    image = Image.open(uploaded_image)
    resized_image = resize_image(image, scale_factor=2.0)
    st.sidebar.image(resized_image, caption="Uploaded Image",
                     use_container_width=True)

query = st.chat_input("Enter your text or question here...")

if query or (uploaded_image is not None and not st.session_state.get("image_processed", False)):

    # If user only uploads an image but no text, let's warn them
    if uploaded_image is not None and not query:
        st.sidebar.warning(
            "Please enter a prompt to accompany the uploaded image.")
    else:
        # 1) Handle user text first
        if query:
            st.session_state["current_chat"].append({
                "role": "user",
                "parts": [{"text": query}]
            })
            with st.chat_message("user"):
                st.markdown(query)

        # 2) Handle user image
        if uploaded_image is not None:
            try:
                # --- IMAGE PERSISTENCE CHANGES ---
                # if  the image is png , convert it to jpeg
                if uploaded_image.type == "image/png":
                    image = Image.open(uploaded_image)
                    image = image.convert("RGB")
                    resized_image = resize_image(image, scale_factor=2.0)

                buffer = BytesIO()
                image.save(buffer, format="JPEG")
                base64_data = base64.b64encode(
                    buffer.getvalue()).decode("utf-8")

                st.session_state["current_chat"].append({
                    "role": "user",
                    "parts": [
                        {
                            "mime_type": "image/jpeg",
                            "data": base64_data,



                        }
                    ]
                })

                # Also display it in the current chat
                with st.chat_message("user"):
                    st.image(resized_image, caption="Uploaded Image",
                             use_container_width=True)

            except Exception as e:
                st.sidebar.error(f"Failed to encode image in base64: {e}")

        # Mark the image as processed so we don't re-upload next run
        st.session_state["image_processed"] = True

        # 3) Finally, generate model response if there's a query
        if query:
            generate_response(query)

# Reset the image_processed flag after handling
if st.session_state.get("image_processed", False):
    st.session_state["image_processed"] = False
