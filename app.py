import gradio as gr
import requests
import webbrowser
import threading
import time

API_URL = "http://127.0.0.1:8000"

def process_company(company):
    if not company.strip():
        return {"error": "Please enter a valid company name."}, None

    try:
        response = requests.post(f"{API_URL}/news", json={"company": company})
        if response.status_code == 200:
            news_data = response.json()
        else:
            error_message = response.json().get("detail", "Failed to fetch news data")
            return {"error": error_message}, None

        # Use the Hindi sentiment analysis for TTS
        final_text_hindi = news_data.get("Final Sentiment Analysis (Hindi)", "हिंदी सारांश उपलब्ध नहीं है।")
        tts_response = requests.post(f"{API_URL}/tts", json={"text": final_text_hindi})
        audio_file = None
        if tts_response.status_code == 200:
            audio_file = tts_response.json().get("audio_file", None)
        return news_data, audio_file

    except requests.exceptions.RequestException as e:
        return {"error": f"Network error: {e}"}, None
    except Exception as e:
        return {"error": f"Unexpected error: {e}"}, None

def open_browser():
    time.sleep(3)
    webbrowser.open("http://127.0.0.1:7860")

iface = gr.Interface(
    fn=process_company,
    inputs=gr.Textbox(label="Enter Company Name"),
    outputs=[
        gr.JSON(label="News Sentiment Report"),
        gr.Audio(label="Hindi TTS Audio")
    ],
    title="News Summarization and TTS Application",
    description="Enter a company name to fetch news, analyze sentiment, and get a Hindi audio summary."
)

if __name__ == "__main__":
    threading.Thread(target=open_browser, daemon=True).start()
    iface.launch()
