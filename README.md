# News Summarization and Text-to-Speech Application

## Overview
This project is a web-based application that extracts key details from multiple news articles related to a given company, performs sentiment analysis, conducts a comparative analysis, and generates a text-to-speech (TTS) output in Hindi. The application uses web scraping, sentiment analysis, and TTS conversion, and features a simple web interface built with Gradio.

## Features
- News Extraction: Scrapes at least 10 news articles related to a given company.
- Sentiment Analysis: Uses a pre-trained transformer model to analyze the sentiment of each article.
- Comparative Analysis: Aggregates and compares sentiment across articles.
- Text-to-Speech: Generates Hindi speech from the sentiment summary using gTTS.
- API Development: Provides backend API endpoints using FastAPI.
- Web Interface: User-friendly interface using Gradio.
- Deployment: Ready to be deployed on Hugging Face Spaces.

## Structure
- api.py – FastAPI endpoints.
- app.py – frontend code (Gradio).
- utils.py – Contains all the utility functions (news extraction, sentiment analysis, translation, TTS).
- requirements.txt – Python dependencies

Note: While I've not been able to generate hindi speech and text, the program is able to read news headlines from the web and is able to generate an output.mp3 reading the final sentiment analysis
