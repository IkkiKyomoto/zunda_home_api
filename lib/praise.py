import os
from langchain_openai import ChatOpenAI

from transformers import AutoTokenizer, AutoModelForSequenceClassification, LukeConfig
import torch

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
    print(messages)
    res = llm.invoke(messages)
    return res.content

def get_praise_voice_url(praise_text: str) -> str:

    return f"https://example.com/praise.mp3"

def get_feeling(praise_text: str) -> str:
    model_name = os.getenv("FEELING_MODEL")
    max_length = int(os.getenv("FEELING_MAX_LENGTH"))
    if not model_name:
        raise Exception("Model name is required")
    if not max_length:
        raise Exception("Max length is required")

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