import os
from langchain_openai import ChatOpenAI
from transformers import AutoTokenizer, AutoModelForSequenceClassification, LukeConfig
import torch
import requests

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
    voicevox_baseurl = os.getenv("VOICEVOX_URL")
    speaker_id = 3
    if not voicevox_baseurl:
        raise Exception("Voicevox URL is required")
    query = requests.post(f"{voicevox_baseurl}/audio_query?text={praise_text}&speaker={speaker_id}")
    if query.status_code != 200:
        raise Exception(f"Failed to get audio query: {query.text}")
    audio_query = query.json()
    praise_voice = requests.post(f"{voicevox_baseurl}/synthesis?speaker={speaker_id}", json=audio_query)
    print(praise_voice)
    return praise_voice.json()
    
def get_feeling(praise_text: str) -> str:
    model_name = os.getenv("FEELING_MODEL")
    max_length = int(os.getenv("FEELING_MAX_LENGTH"))
    if not model_name:
        raise Exception("Model name is required")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    config = LukeConfig.from_pretrained(model_name, output_hidden_states=True)    
    model = AutoModelForSequenceClassification.from_pretrained(model_name, config=config)
    token=tokenizer(praise_text, truncation=True, padding="max_length", max_length=max_length)
    output=model(torch.tensor(token["input_ids"]).unsqueeze(0), torch.tensor(token["attention_mask"]).unsqueeze(0))
    max_index = torch.argmax(torch.tensor(output.logits[0])).item()
    if max_index == 0:
        return "joy"
    elif max_index == 1:
        return "sadness"
    elif max_index == 2:
        return "anticipation"
    elif max_index == 3:
        return "surprise"
    elif max_index == 4:
        return "anger"
    elif max_index == 5: 
        return "fear"   
    elif max_index == 6:
        return "disgust"
    elif max_index == 7:
        return "trust"