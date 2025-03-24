import requests
import logging
import os
import nltk
from transformers import pipeline
from gtts import gTTS
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from deep_translator import GoogleTranslator

# Set up logging
logging.basicConfig(level=logging.INFO)

# Load and configure the sentiment analysis model
sentiment_pipeline = pipeline(
    "sentiment-analysis",
    model="distilbert/distilbert-base-uncased-finetuned-sst-2-english"
)

# Ensure necessary NLTK resources
nltk_resources = ["punkt", "stopwords"]
for resource in nltk_resources:
    try:
        nltk.data.find(f"tokenizers/{resource}") if resource == "punkt" else nltk.data.find(f"corpora/{resource}")
    except LookupError:
        nltk.download(resource)

# Secure API key (if needed for news extraction)
NEWS_API_KEY = os.getenv("NEWS_API_KEY", "6e61dc1545bd464db26fffec489629e7")

def fetch_news(company):
    """
    Fetches news articles related to the company using NewsAPI.org.
    """
    url = f"https://newsapi.org/v2/everything?q={company}&apiKey={NEWS_API_KEY}&language=en&pageSize=5"
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        return data.get("articles", [])
    except requests.exceptions.RequestException as e:
        logging.error(f"Error fetching news: {e}")
        return []

def analyze_sentiment(text):
    """
    Analyzes the sentiment of the provided text.
    Returns: 'Positive', 'Negative', or 'Neutral'.
    """
    if not text.strip():
        return "Neutral"
    try:
        result = sentiment_pipeline(text[:512])
        sentiment = result[0]["label"]
        return "Positive" if sentiment == "POSITIVE" else "Negative" if sentiment == "NEGATIVE" else "Neutral"
    except Exception as e:
        logging.error(f"Error in sentiment analysis: {e}")
        return "Neutral"

def get_topics(text):
    """
    Extracts the top 3 frequent words (excluding stopwords) as topics.
    """
    if not text.strip():
        return []
    stop_words = set(stopwords.words("english"))
    words = word_tokenize(text)
    filtered_words = [word.lower() for word in words if word.isalnum() and word.lower() not in stop_words]
    freq_dist = nltk.FreqDist(filtered_words)
    return [word for word, _ in freq_dist.most_common(3)]

def comparative_analysis(articles):
    """
    Performs sentiment analysis and topic extraction on articles.
    Returns: sentiment counts and updated articles.
    """
    sentiment_counts = {"Positive": 0, "Negative": 0, "Neutral": 0}
    updated_articles = []
    
    for article in articles:
        summary = article.get("description") or article.get("content") or article.get("title")
        if not summary:
            continue  
        sentiment = analyze_sentiment(summary)
        topics = get_topics(summary)
        article["sentiment"] = sentiment
        article["topics"] = topics
        sentiment_counts[sentiment] += 1
        updated_articles.append(article)
    
    return sentiment_counts, updated_articles

def translate_to_hindi(text):
    """
    Translates English text to Hindi using deep-translator.
    """
    if not text.strip():
        logging.error("❌ Translation failed: No text provided.")
        return text
    try:
        hindi_text = GoogleTranslator(source='en', target='hi').translate(text)
        logging.info(f"✅ Translated to Hindi: {hindi_text}")
        return hindi_text
    except Exception as e:
        logging.error(f"❌ Error in translation: {e}")
        return text  # Fallback to English if translation fails

def text_to_speech(text, lang="hi"):
    """
    Converts translated Hindi text to speech and saves output as 'output.mp3'.
    Assumes the input text is in English and forces a Hindi translation.
    Returns the filename and the final Hindi text.
    """
    if not text.strip():
        logging.error("❌ TTS failed: No text provided.")
        return None, None
    try:
        # Force translation to Hindi before TTS conversion
        hindi_text = translate_to_hindi(text)
        logging.info(f"✅ Final Text for TTS (Hindi): {hindi_text}")
        tts = gTTS(hindi_text, lang="hi", slow=False)
        filename = "output.mp3"
        tts.save(filename)
        logging.info(f"✅ Generated TTS File: {filename}")
        return filename, hindi_text
    except Exception as e:
        logging.error(f"❌ Error in TTS conversion: {e}")
        return None, None
