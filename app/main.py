from fastapi import FastAPI, HTTPException
from lib import praise
import logging

app = FastAPI()

# ログ設定
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@app.get("/praise")
def get_praise(user_input: str):
    if not user_input:
        raise HTTPException(status_code=400, detail="User input is required")
    try:
        praise_text: str = praise.get_praise_text(user_input)
        praise_text: str = "ユアグレートパーソンなのだ"
        praise_voice: str = praise.get_praise_voice(praise_text)
        feeling: str = praise.get_feeling(praise_text)
        return {"praise_text": praise_text, "praise_voice": praise_voice, "feeling": feeling}
    except Exception as e:
        logger.error(f"Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

