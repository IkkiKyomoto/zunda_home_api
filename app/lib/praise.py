import os
from langchain_openai import ChatOpenAI
from transformers import AutoTokenizer, AutoModelForSequenceClassification, LukeConfig
import torch
import requests
import base64

def get_praise_text(user_input: str) -> str:
    system_message: str = os.getenv("SYSTEM_MESSAGE")
    llm = ChatOpenAI(
        model="gpt-4o",
        temperature=0.5,
        max_tokens=100,        
    )
    messages = [
        ("system", system_message),
        ("human", user_input),
    ]
    res = llm.invoke(messages)
    return res.content

def get_praise_voice(praise_text: str) -> str:
    voicevox_api_type = os.getenv("VOICEVOX_API_TYPE")
    if not voicevox_api_type:
        raise Exception("Voicevox API type is required")
    if voicevox_api_type == "local":
        voicevox_baseurl = os.getenv("VOICEVOX_URL")
        speaker_id = 3
        if not voicevox_baseurl:
            raise Exception("Voicevox URL is required")
        try:
            query = requests.post(f"{voicevox_baseurl}/audio_query?text={praise_text}&speaker={speaker_id}")
            query.raise_for_status()
            audio_query = query.json()
            praise_voice_res = requests.post(f"{voicevox_baseurl}/synthesis?speaker={speaker_id}", json=audio_query)
            praise_voice_res.raise_for_status()
            praise_voice = base64.b64encode(praise_voice_res.content).decode()
            return praise_voice
        except requests.exceptions.RequestException as e:
            raise Exception(f"HTTP request failed: {e}")
    elif voicevox_api_type == "web_api":
        voicevox_baseurl = os.getenv("VOICEVOX_URL")
        speaker_id = 3
        if not voicevox_baseurl:
            raise Exception("Voicevox URL is required")
        try:
            synthesis_res = requests.get(f"{voicevox_baseurl}/v3/voicevox/synthesis/?text={praise_text}&speaker={speaker_id}")
            synthesis_res.raise_for_status()
            synthesis_json = synthesis_res.json()
            print(synthesis_json)
            praise_voice_url = synthesis_json["wavDownloadUrl"]
            print(praise_voice_url)
            praise_voice_res = requests.get(praise_voice_url)
            praise_voice = base64.b64encode(praise_voice_res.content).decode()
            return praise_voice
        except requests.exceptions.RequestException as e:
            raise Exception(f"HTTP request failed: {e}")
    
def get_feeling(praise_text: str) -> str:
    model_name = os.getenv("FEELING_MODEL")
    max_length = int(os.getenv("FEELING_MAX_LENGTH", 100))
    if not model_name:
        raise Exception("Model name is required")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        config = LukeConfig.from_pretrained(model_name, output_hidden_states=True)
        model = AutoModelForSequenceClassification.from_pretrained(model_name, config=config)
        tokens = tokenizer(praise_text, truncation=True, padding="max_length", max_length=max_length, return_tensors="pt")
        output = model(**tokens)
        max_index = torch.argmax(output.logits[0]).item()
        feelings = ["joy", "sadness", "anticipation", "surprise", "anger", "fear", "disgust", "trust"]
        return feelings[max_index]
    except Exception as e:
        raise Exception(f"Error in processing the model: {e}")