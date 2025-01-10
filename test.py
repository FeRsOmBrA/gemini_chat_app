import os
import google.generativeai as genai

genai.configure(api_key=os.environ["GOOGLE_API_KEY"])


# Create the model
generation_config = {
    "temperature": 1,
    "top_p": 0.95,
    "top_k": 40,
    "max_output_tokens": 8192,
    "response_mime_type": "text/plain",
}

model = genai.GenerativeModel(
    model_name="gemini-1.5-pro",
    generation_config=generation_config,

)


history = [
    {
        "role": "assistant",
        "parts": [
            "You are a expert AI model"
        ],
    },
]
chat_session = model.start_chat(
    history=history

)

response = chat_session.send_message(
    
    "What was the last event in Colombia?", 
                                     
                                     tools={"google_search_retrieval": {
                                         "dynamic_retrieval_config": {
                                             "mode": "unspecified",
                                             "dynamic_threshold": 0.06}}}
                                     
                                     )


print(response.text)
