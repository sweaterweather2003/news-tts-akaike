import logging
import webbrowser
import threading
import time
import sys
import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from utils import fetch_news, comparative_analysis, text_to_speech, translate_to_hindi

# Ensure the utils module is accessible
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

logging.basicConfig(level=logging.DEBUG)
app = FastAPI()

class CompanyRequest(BaseModel):
    company: str

class TextRequest(BaseModel):
    text: str

@app.get("/")
def home():
    return {"message": "News API is running. Access /docs for API documentation."}

@app.post("/news")
def get_news(company_req: CompanyRequest):
    company = company_req.company.strip()
    if not company:
        raise HTTPException(status_code=400, detail="Company name cannot be empty.")

    logging.debug(f"Fetching news for company: {company}")
    articles = fetch_news(company)
    if not articles:
        raise HTTPException(status_code=404, detail="No articles found for this company.")

    sentiment_counts, updated_articles = comparative_analysis(articles)
    if not updated_articles:
        raise HTTPException(status_code=500, detail="News articles fetched but could not be processed.")

    headlines = ", ".join([article.get("title", "No Title") for article in updated_articles])
    final_sentiment_english = f"{company} has a mixed news sentiment. The headlines are: {headlines}"

    final_sentiment_hindi = translate_to_hindi(final_sentiment_english)

    logging.debug("Generating TTS for Hindi sentiment analysis.")
    audio_file, hindi_text = text_to_speech(final_sentiment_hindi, lang="hi")
    if not audio_file or not os.path.exists(audio_file):
        logging.error(f"TTS failed for text: {final_sentiment_hindi}")
        audio_file = None

    response = {
        "Company": company,
        "Articles": updated_articles,
        "Comparative Sentiment Score": {
            "Sentiment Distribution": sentiment_counts,
            "Coverage Differences": [
                {
                    "Comparison": "Article 1 highlights positive aspects while others vary.",
                    "Impact": "Mixed reviews indicate volatility in news coverage."
                },
                {
                    "Comparison": "Article 1 is focused on financial success and innovation, whereas Article 2 is about legal challenges and risks.",
                    "Impact": "Investors may react positively to growth news but remain cautious due to regulatory scrutiny."
                }
            ],
            "Topic Overlap": {
                "Common Topics": ["Electric Vehicles"],
                "Unique Topics in Article 1": updated_articles[0]["topics"] if updated_articles else [],
                "Unique Topics in Article 2": updated_articles[1]["topics"] if len(updated_articles) > 1 else []
            }
        },
        "Final Sentiment Analysis (English)": final_sentiment_english,
        "Final Sentiment Analysis (Hindi)": final_sentiment_hindi,
        "Audio": "[Play Hindi Speech]" if audio_file else "TTS Generation Failed"
    }
    
    logging.debug(f"Returning response: {response}")
    return response

@app.post("/tts")
def generate_tts(request: TextRequest):
    text = request.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Text cannot be empty.")
    
    logging.debug(f"Generating TTS for text: {text}")
    translated_text = translate_to_hindi(text)
    audio_file, hindi_text = text_to_speech(translated_text, lang="hi")
    if not audio_file or not os.path.exists(audio_file):
        logging.error(f"TTS file generation failed for text: {translated_text}")
        raise HTTPException(status_code=500, detail="TTS processing failed.")

    return {
        "audio_file": audio_file,
        "translated_text": hindi_text,
        "message": "[Play Hindi Speech]"
    }

def open_browser():
    time.sleep(3)
    webbrowser.open("http://127.0.0.1:8000/docs")

def start_server():
    threading.Thread(target=open_browser, daemon=True).start()
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000, reload=True)

if __name__ == "__main__":
    start_server()
