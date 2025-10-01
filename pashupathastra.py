import json
import urllib.parse
import socket
import sqlite3
from csv import excel

from selenium.webdriver.chrome.options import Options
import smtplib
from bs4 import BeautifulSoup
import subprocess
import sys
import docx2txt
from dotenv import load_dotenv
from http.client import responses
from PyPDF2 import PdfReader
import re
import cv2
import numpy as np
from werkzeug.utils import secure_filename
from config import apikey
import pandas as panda
from ipaddress import ip_address
from logging.config import listen
import requests
import wikipedia
from pyparsing import original_text_for
import torch
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from sqlalchemy.util import await_only
from pyspark.sql import SparkSession
from webdriver_manager.chrome import ChromeDriverManager
import os
import pyarrow
import socket
import requests
import geocoder
import asyncio
import webbrowser
import pyttsx3
import speech_recognition as sr
import pyaudio
import torch
import torch.nn as nn
from PIL import Image
import torchvision.transforms as transforms
import datetime
import time
from requests import get
from googletrans import Translator
from gtts import gTTS
import playsound
import glob
import os
import yt_dlp
import uuid
import mysql.connector
import smtplib
from datetime import datetime
from wikipedia import summary
from PyQt5 import QtWidgets,QtCore,QtGui
from PyQt5.QtCore import QTime,QTimer,QDate,Qt
from PyQt5.QtGui import QMovie
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.uic import loadUiType
import multiprocessing
import faulthandler
import PyPDF2
import platform
import urllib.parse
import threading
pyautogui = None
gender = "male"
if not os.environ.get('DISPLAY'):
    print("No GUI environment detected. Skipping pyautogui-related imports.")
    try:
        import pyautogui as pg
        pyautogui = pg  # Reassign properly
        import pywhatkit
    except ImportError:
        print("pyautogui not installed. Some features will be unavailable.")
system = platform.system()
faulthandler.enable()
engine = pyttsx3.init()
lock = threading.Lock()
# Labels for age, gender, and emotions
age_labels = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
gender_labels = ['Male', 'Female']
emotions = ["Neutral", "Happiness", "Surprise", "Sadness", "Anger", "Disgust", "Fear", "Contempt"]
# Transform for input image
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])
# Load models
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
age_net = cv2.dnn.readNetFromCaffe('models/deploy_age.prototxt', 'models/age_net.caffemodel')
gender_net = cv2.dnn.readNetFromCaffe('models/deploy_gender.prototxt', 'models/gender_net.caffemodel')
emotion_net = cv2.dnn.readNetFromONNX('emotion-ferplus-8.onnx')
# Email config
load_dotenv()
SENDER_EMAIL = os.getenv("SENDER_EMAIL")
SENDER_PASSWORD = os.getenv("SENDER_PASSWORD")
# Initialize the TTS engine
if system == "Windows":
    engine = pyttsx3.init('sapi5')
else:
    engine = pyttsx3.init()
voices = engine.getProperty('voices')
engine.setProperty('voice', voices[0].id)
# Replace with the actual path to the ChromeDriver
CHROMEDRIVER = "E:\APPS\chromedriver-win64"
recognizer = sr.Recognizer()
translator = Translator()
# Define a global variable so the driver persists
driver = None
# Folder where PDF files are stored
PDF_FOLDER = "./pdfs"
# Text to speech functions
def speak(text):
    with lock:
        engine.say(text)
        engine.runAndWait()
# Function to play the song
def get_voice_command():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("Say the song name:")
        recognizer.adjust_for_ambient_noise(source)
        audio = recognizer.listen(source)

    try:
        command = recognizer.recognize_google(audio)
        print(f"🗣 You said: {command}")
        return command
    except sr.UnknownValueError:
        print("Could not understand the audio.")
    except sr.RequestError as e:
        print(f"Could not request results; {e}")
    return None
def log_to_db_success(action, result, status):
    try:
        # Connect to MySQL Workbench
        conn = mysql.connector.connect(
            host="localhost",
            user="root",
            password="12345",
            database="pashupathastra_ai"
        )
        cursor = conn.cursor()
        result="Success"
        # Insert log into the logs_success table
        sql_query = "INSERT INTO logs_success (action, result, status, timestamp) VALUES (%s, %s, %s, %s)"
        data = (action, result, status, datetime.now())

        cursor.execute(sql_query, data)
        conn.commit()

        cursor.close()
        conn.close()
        print("Database logging successful.")

    except mysql.connector.Error as e:
        print(f"MySQL database logging error: {e}")
def log_to_db_error(action,result,status):
    try:
        # Connect to MySQL Workbench
        conn = mysql.connector.connect(
            host="localhost",
            user="root",
            password="12345",
            database="pashupathastra_ai"
        )
        cursor = conn.cursor()
        result = "ERROR"
        # Insert log into the logs_success table
        sql_query = "INSERT INTO logs_error (action, result, status, timestamp) VALUES (%s, %s, %s, %s)"
        data = (action, result, status, datetime.now())

        cursor.execute(sql_query, data)
        conn.commit()

        cursor.close()
        conn.close()
        print("Database logging successful.")

    except mysql.connector.Error as e:
        print(f"MySQL database logging error: {e}")
def get_youtube_url(query):
    ydl_opts = {
        'quiet': True,
        'skip_download': True,
        'extract_flat': 'in_playlist',
        'default_search': 'ytsearch1',
        'forcejson': True,
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        try:
            result = ydl.extract_info(query, download=False)
            video_url = result['entries'][0]['url']

            # If it's already a full YouTube URL, return as is
            if video_url.startswith("http"):
                return video_url
            else:
                return f"https://www.youtube.com/watch?v={video_url}"
        except Exception as e:
            print(f" Failed to fetch video: {e}")
            return None
def open_in_chrome(url):
    print(f"Opening: {url}")
    if platform.system() == "Windows":
        chrome_path = "C:/Program Files/Google/Chrome/Application/chrome.exe"
        if not os.path.exists(chrome_path):
            chrome_path = "C:/Program Files (x86)/Google/Chrome/Application/chrome.exe"
        if os.path.exists(chrome_path):
            webbrowser.register('chrome', None, webbrowser.BackgroundBrowser(chrome_path))
            webbrowser.get('chrome').open(url)
        else:
            print(" Chrome not found. Using default browser.")
            webbrowser.open(url)
    else:
        chrome_path = "/usr/bin/google-chrome"
        if os.path.exists(chrome_path):
            webbrowser.register('chrome', None, webbrowser.BackgroundBrowser(chrome_path))
            webbrowser.get('chrome').open(url)
        else:
            print("Chrome not found. Using default browser.")
            webbrowser.open(url)
# Function to take voice input
def listen_command():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        speak("What song do you like to play, Sir?")
        print(" Say the song name to play on YouTube...")
        audio = recognizer.listen(source)
    try:
        command = recognizer.recognize_google(audio)
        print(f" You said: {command}")
        return command
    except sr.UnknownValueError:
        print(" Could not understand audio.")
        return None
    except sr.RequestError as e:
        print(f" Could not request results; {e}")
        return None
# Tells us the location of the user
def location():
    try:
        res = requests.get("https://ipinfo.io")
        data = res.json()
        city = data.get("city", "Unknown")
        latitude, longitude = data.get("loc", "0,0").split(",")
        speak(f"You are currently residing in {city}. Precisely your coordinates are latitude:{latitude} and longitude:{longitude}")
    except Exception as e:
        print("Failed to get location:", e)
        speak("I couldn't determine your location.")
def listen_and_detect():

    with sr.Microphone() as source:
        print("Listening.......")
        audio = recognizer.listen(source)
        try:
            print("Recognizing........")
            query = recognizer.recognize_google(audio)
            print("You said", query)
            return query
        except sr.UnknownValueError:
            print("Sorry, I could not understand the audio")
        except sr.RequestError:
            print("Could not request results")
        return None
#date-time module
def wish():
    hour = datetime.now().hour
    if hour>=0  and hour<=12:
        speak("Good Morning Sir!")
    elif hour>=12 and hour<16:
        speak("Good Afternoon Sir!")
    elif hour>=16 and hour<20:
        speak("Good Evening Sir!")
    else:
        speak("Good Night Sir!")
# listen and translate
async def listen_and_translate():
    while True:
        print("\n---- Main Menu ----")
        speak("Say something to translate or say 'main menu' to go back")
        query = listen_and_detect()
        if query:
            if "main menu" in query.lower() or "exit" in query.lower() or "go back" in query.lower():
                speak("Returning to the main menu")
                print("Returning to the main menu")
                break
            else:
               await detect_and_translate(query)
# Fetch user Location
def get_user_location():
    try:
        res = requests.get("https://ipinfo.io")
        data = res.json()
        city = data.get("city", "Unknown")
        latitude, longitude = data.get("loc", "0,0").split(",")
        return city, latitude, longitude
    except Exception as e:
        print("Failed to get location:", e)
        speak("I couldn't determine your location.")
        return "Unknown", "0", "0"
#Get weather command
def weather_listen():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("Listening...")
        speak("Please say something...")
        recognizer.adjust_for_ambient_noise(source)
        try:
            audio = recognizer.listen(source, timeout=5, phrase_time_limit=8)
            print("Recognizing...")
            command = recognizer.recognize_google(audio)
            print(f"You said: {command}")
            return command.lower()
        except sr.WaitTimeoutError:
            print("Listening timed out.")
            speak("Sorry, I couldn't hear you. Please try again.")
        except sr.UnknownValueError:
            print("Could not understand audio.")
            speak("I didn't catch that. Could you repeat?")
        except sr.RequestError as e:
            print(f"Error: {e}")
            speak("There was an error with the voice service.")
        return None
# Get the weather of the location
def get_weather(city):
    api = "b5df6ce959c52f1149f64772a4120899"
    BASE_URL = "https://api.openweathermap.org/data/2.5/weather"
    try:
        params = {"q":city,"appid":api,"units":"metric"}
        response = requests.get(BASE_URL,params=params)
        data = response.json()

        if response.status_code == 200:
            weather_description = data["weather"][0]["description"]
            temp = data["main"]["temp"]
            humidity = data["main"]["humidity"]
            wind_speed = data["wind"]["speed"]

            weather_report = (f"The current weather in {city} is {weather_description}. "
                              f"The temperature is {temp} degrees Celsius with a humidity of {humidity}% "
                              f"and wind speed of {wind_speed} meters per second.")
            return weather_report
        else:
            return f"Could not fetch the weather report"
    except Exception as e:
        print("Error fetching details ",e)
        return "Sorry, I could not fetch the details"
# Get weather prediction
def weather_detection():
    speak("Do you want to know the today's weather?")
    print("Do you want to know the today's weather?")
    command = weather_listen()
    if command:
        city,latitude,longitude = get_user_location()
        speak(f"Fetching weather information for {city}.")
        weather_report = get_weather(city)
        print(weather_report)
        speak(weather_report)
    elif "main menu" in command:
        return
    else:
        print("Okay let me know if you want something else")
# Translation
async def detect_and_translate(text):
    detected = await translator.detect(text)
    lang_code = detected.lang
    print(f"Detected language:{lang_code}")

    # Translate to Indian english
    translation =await translator.translate(text,src = lang_code,dest='en')
    print(f"Translated:{translation.text}")
    speak(f"{translation.text}")
# Wikipedia
def listen_wikipedia_command():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        speak("Sir, what do you like to search on wikipedia?")
        print("Listening for topic......")
        recognizer.adjust_for_ambient_noise(source)
        audio = recognizer.listen(source)
    try:
        command = recognizer.recognize_google(audio)
        print(f"You said:{command}")
        return command
    except sr.UnknownValueError:
        print("Sorry, I could not understand that")
        return command
    except sr.UnknownValueError:
        print("There was a problem connecting to the speech service")
        return None
# Listen google Command
def listen_google_command():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        speak("What do you want to search for the day,Sir?")
        print("Listening........")
        audio = recognizer.listen(source)
    try:
        command = recognizer.recognize_google(audio)
        print(f"You said:{command}")
        return command
    except sr.UnknownValueError:
        print("Sorry. I didn't understand understand what you said")
        return None
    except sr.RequestError:
        print("Cannot request result")
        return  None
# search in google
def search_google():
    command = listen_google_command()
    if command:
        speak(f"Searching in google {command}")
        url = f"https://www.google.com/search?q={command.replace(' ', '+')}"
        webbrowser.open(url)
# WikiPedia Search Command by Voice Command
def search_wikipedia_info(query):
    try:
        query = query.replace("wikipedia","")
        summary = wikipedia.summary(query)
        return summary
    except wikipedia.exceptions.DisambiguationError as e:
        return f"Your query is too vague. Try one of these: {e.options[:5]}"
    except wikipedia.exceptions.PageError:
        return "Sorry! I couldn't find any results on the query"
    except Exception as e:
        return f"Ann error has occurred: {str(e)}"
EMAIL = "9875441655"
PASSWORD = "Suman@Talukdar"
# open facebook
def login_facebook():
    speak("Opening your Facebook and logging into your profile now...")
    # Open Facebook in Chrome
    chrome_path = "C:/Program Files/Google/Chrome/Application/chrome.exe %s"  # Adjust path if needed
    webbrowser.get(chrome_path).open("https://www.facebook.com")
    # Wait for browser and page to load
    time.sleep(7)
    # Simulate typing login credentials using pyautogui
    pyautogui.write(EMAIL)
    pyautogui.press('tab')  # Move to password field
    pyautogui.write(PASSWORD)
    pyautogui.press('enter')  # Press login button
    speak("You are now logged into Facebook.")
# listen to get news
def listen_command_news():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        speak("Sir.....What kind of news do you want to hear today?")
        audio = recognizer.listen(source)
        try:
            command = recognizer.recognize_google(audio)
            return command.lower()
        except sr.UnknownValueError:
            print("Sorry! I did not understand that")
        except sr.RequestError:
            print("Network error")
# Listen to auto email commands
def listen_email():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        print("Listening...")
        r.adjust_for_ambient_noise(source)
        audio = r.listen(source)
    try:
        command = r.recognize_google(audio)
        print(f"You said: {command}")
        return command
    except sr.UnknownValueError:
        speak("Sorry, I didn't catch that.")
        return None
# Getting details of the email
def voice_email_details():
    speak("Do you want to open Gmail?")
    command = listen_email()
    if command and "yes" in command.lower():
        webbrowser.open("https://mail.google.com")
        speak("Opening Gmail...")

    speak("To whom should I send the email?")
    to_email = listen()

    speak("What is the subject of your email?")
    subject = listen()

    speak("What should I say in the email?")
    body = listen()

    if to_email and subject and body:
        send_email(to_email, subject, body)
def send_email(to_email, subject, body):
    sender_email = "suman.talukdar53@gmail.com"
    sender_password = "zjvq idph sybe ftxb"  # App password, not your main Gmail password

    message = f"Subject: {subject}\n\n{body}"
    try:
        server = smtplib.SMTP("smtp.gmail.com", 587)
        server.starttls()
        server.login(sender_email, sender_password)
        server.sendmail(sender_email, to_email, message)
        server.quit()
        speak("Email has been sent successfully.")
    except Exception as e:
        speak("Sorry, I was unable to send the email.")
        print(e)
def takeCommands():
    r = sr.Recognizer()
    audio = None
    try:
        with sr.Microphone() as source:
            r.adjust_for_ambient_noise(source)
            print("Listening...")
            audio = r.listen(source, timeout=3, phrase_time_limit=5)
    except sr.WaitTimeoutError:
        print("Listening timed out.")
        return None
    except Exception as e:
        print(f"Microphone error: {e}")
        return None

    if audio is None:
        return None

    try:
        print("Recognizing...")
        query = r.recognize_google(audio, language='en-in')
        print(f"User said: {query}")
        return query
    except sr.UnknownValueError:
        print("Sorry, I didn't understand.")
        return None
    except sr.RequestError as e:
        print(f"Speech recognition error: {e}")
        return None
# MS word operations
def word_open():
    recognizer = sr.Recognizer()
    mic = sr.Microphone()
    speak("MS word has been opened for you. Please tell me what would you like me to type")
    speak("Please tell me what would like me to type......?")
    speak("Please listen to the following instruction very carefully....")
    speak("Please tell 'stop to stop typing'")
    speak("Please tell save to save the file")
    speak("Please tell paragraph to change the paragraph")
    speak("Please tell space to put whitespace between the words")
    with mic as source:
        recognizer.adjust_for_ambient_noise(source)
        print("Listening... Say 'stop typing' to exit.")
        while True:
            try:
                audio = recognizer.listen(source, timeout=5)
                user_text_cmd = recognizer.recognize_google(audio).lower()
                if user_text_cmd:
                    if "stop" in user_text_cmd:
                        speak("Stopping typing")
                        break
                    elif "save the file" in user_text_cmd:
                        speak("saving the file")
                        pyautogui.hotkey('ctrl', 's')
                        time.sleep(1)
                        speak("Please tell me the name of the file which you want to save")
                        user_filename = takeCommands().lower()
                        pyautogui.write(f"{user_filename}.docx")
                        pyautogui.press("enter")
                        speak("File has been saved successfully")
                    elif "paragraph" in user_text_cmd:
                        speak("Changing paragraph......")
                        pyautogui.typewrite('\n' + '\t \t \t \t')
                    elif "space" in user_text_cmd:
                        speak("White space")
                        pyautogui.typewrite("  ")
                    else:
                        pyautogui.typewrite(user_text_cmd + "  ")
            except sr.WaitTimeoutError:
                print("No speech detected. Retrying...")
            except sr.UnknownValueError:
                print("Sorry, I didn't catch that.")
            except sr.RequestError as e:
                print(f"Could not request results; {e}")
                break
# captures facial image right now and analyzes face
def capture_facial_image():
    # Start webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        speak("Webcam not accessible.")
        exit()

    speak("Camera is on. Please adjust your face. Press Enter or Space to proceed.")
    print("Adjust your face in front of the camera. Press Enter or Space to proceed...")

    # Show live preview window until user presses a key
    while True:
        ret, preview_frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break

        cv2.imshow("Camera Preview - Press Enter or Space to start", preview_frame)
        key = cv2.waitKey(1)
        if key == 13 or key == 32:  # Enter or Space
            break

    # Start detection
    speak("Starting face analysis.")
    start_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            speak("Failed to grab frame.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(60, 60))

        if len(faces) > 0:
            for (x, y, w, h) in faces:
                face_img = frame[y:y + h, x:x + w].copy()

                # Gender
                face_blob = cv2.dnn.blobFromImage(face_img, 1.0, (227, 227),
                                                  (78.4263377603, 87.7689143744, 114.895847746), swapRB=False)
                gender_net.setInput(face_blob)
                gender_preds = gender_net.forward()
                gender = gender_labels[gender_preds[0].argmax()]

                # Age
                age_net.setInput(face_blob)
                age_preds = age_net.forward()
                age = age_labels[age_preds[0].argmax()]

                # Emotion
                face_gray = gray[y:y + h, x:x + w]
                face_resized = cv2.resize(face_gray, (64, 64))
                emotion_blob = cv2.dnn.blobFromImage(face_resized, 1 / 255, (64, 64), (0,), swapRB=False, crop=False)
                emotion_net.setInput(emotion_blob)
                emotion_preds = emotion_net.forward()
                emotion = emotions[int(np.argmax(emotion_preds))]

                # Result
                result = f"The person is a {gender} aged around {age} and seems to be {emotion}."
                print(result)
                speak(result)

                # Draw on frame
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, f"{gender}, {age}, {emotion}", (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                break  # Detect only first face

        else:
            if time.time() - start_time > 15:
                speak("No face detected in camera. Exiting.")
                break

        cv2.imshow("Emotion, Age & Gender Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            speak("Exiting face detection.")
            break

    cap.release()
    cv2.destroyAllWindows()
# Performs operations in notepad
def notepad_open():

    time.sleep(2.5)
    speak("Please tell me what would like me to type......?")
    speak("Please listen to the following instruction very carefully....")
    speak("Please tell 'stop to stop typing'")
    speak("Please tell save to save the file")
    speak("Please tell paragraph to change the paragraph")
    speak("Please tell space to put whitespace between the words")
    recognizer = sr.Recognizer()
    mic = sr.Microphone()
    with mic as source:
        recognizer.adjust_for_ambient_noise(source)
        print("Listening... Say 'stop typing' to exit.")
        while True:
            try:
                audio = recognizer.listen(source, timeout=5)
                user_text_cmd = recognizer.recognize_google(audio).lower()
                if user_text_cmd:
                    if "stop" in user_text_cmd:
                        speak("Stopping typing")
                        break
                    elif "save" in user_text_cmd:
                        speak("saving the file")
                        pyautogui.hotkey('ctrl', 's')
                        time.sleep(1)
                        speak("Please tell me the name of the file which you want to save")
                        user_filename = takeCommands().lower()
                        pyautogui.write(f"{user_filename}.txt")
                        pyautogui.press("enter")
                        speak("File has been saved successfully")
                    elif "paragraph" in user_text_cmd:
                        speak("Changing paragraph......")
                        pyautogui.typewrite('\n' + '\t \t \t \t')
                    elif "space" in user_text_cmd:
                        speak("White space")
                        pyautogui.typewrite("  ")
                    elif "close" in user_text_cmd or "exit notepad" in user_text_cmd:
                        speak("Closing Notepad.")
                        os.system("taskkill /f /im notepad.exe")
                        break
                    else:
                        pyautogui.typewrite(user_text_cmd + "  ")
            except sr.WaitTimeoutError:
                print("No speech detected. Retrying...")
            except sr.UnknownValueError:
                print("Sorry, I didn't catch that.")
            except sr.RequestError as e:
                print(f"Could not request results; {e}")
                break
    speak("Your Task has been completed sir......")
# Tells time
def tell_time():
    now = datetime.now()
    current_time = now.strftime("%I:%M %p")
    speak(f"The time is {current_time}")
# Tells news
def get_news():
    news_url = "https://newsapi.org/v2/top-headlines?sources=techcrunch&apiKey=102791f5d781419786e77f38f16394ea"
    main_page_news = requests.get(news_url).json()
    articles = main_page_news["articles"]
    head = []
    day = ["first","second","third","fourth","fifth","sixth","seventh","eighth","ninth","tenth"]
    for article in articles:
        head.append(article["title"])
    for i in range (len(day)):
        speak(f"Today's {day[i]} news is: {head[i]}")
    speak("That's the news headline for now. Hope you have heard it all")
# Voice command capture
def takeCommand():
    r = sr.Recognizer()
    audio = None
    try:
        with sr.Microphone() as source:
            r.adjust_for_ambient_noise(source)
            print("Listening...")
            audio = r.listen(source, timeout=3, phrase_time_limit=5)
    except sr.WaitTimeoutError:
        print("Listening timed out.")
        return None
    except Exception as e:
        print(f"Microphone error: {e}")
        return None

    if audio is None:
        return None

    try:
        print("Recognizing...")
        query = r.recognize_google(audio, language='en-in')
        print(f"User said: {query}")
        return query
    except sr.UnknownValueError:
        print("Sorry, I didn't understand.")
        return None
    except sr.RequestError as e:
        print(f"Speech recognition error: {e}")
        return None
#Shutting down system
def system_shutdown():
    os_type = platform.system().lower()
    if "windows" in os_type:
        os.system("shutdown /s /t 1")
    elif "linux" in os_type or "darwin" in os_type:
        os.system("shutdown now")
# Restart now system
def system_restart():
    os_type = platform.system().lower()
    if "windows" in os_type:
        os.system("shutdown /r /t 1")
    elif "linux" in os_type or "darwin" in os_type:
        os.system("reboot")
# Sleeping the system
def system_sleep():
    os_type = platform.system().lower()
    if "windows" in os_type:
        os.system("rundll32.exe powrprof.dll,SetSuspendState 0,1,0")
    elif "linux" in os_type:
        os.system("systemctl suspend")
    elif "darwin" in os_type:  # macOS
        os.system("pmset sleepnow")
# Replace with your OpenRouter API key
OPENROUTER_API_KEY = "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"
# LLM Model to use (you can change this to another available OpenRouter model)
MODEL = "mistralai/mistral-7b-instruct"
# Main function of chat bot
def main(question, history):
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": MODEL,
        "messages": history + [{"role": "user", "content": question}],
    }
    response = requests.post(url, headers=headers, json=payload)
    if response.status_code == 200:
        message = response.json()["choices"][0]["message"]["content"]
        return message.strip()
    else:
        print(response.text)
        return "Sorry, there was an error contacting the AI."
def listen_for_filename():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("Say the name of the PDF file (without '.pdf'):")
        recognizer.adjust_for_ambient_noise(source)
        audio = recognizer.listen(source)
        try:
            spoken_text = recognizer.recognize_google(audio)
            filename = spoken_text.strip().lower().replace(" ", "_") + ".pdf"
            print(f" You said: {filename}")
            return filename
        except sr.UnknownValueError:
            print(" Could not understand the audio.")
        except sr.RequestError as e:
            print(f" Error with Google API: {e}")
    return None
def open_powerpoint():
    system_platform = platform.system()

    if system_platform == "Windows":
        speak("Searching for Microsoft PowerPoint on your system, please wait...")

        ppt_path = find_file("POWERPNT.EXE", "C:\\")  # scan entire C drive

        if ppt_path:
            subprocess.Popen([ppt_path])
            speak("Microsoft PowerPoint has been opened for you, Sir")
        else:
            speak("Sorry, I could not find Microsoft PowerPoint on your system.")

    elif system_platform == "Darwin":  # MacOS
        try:
            subprocess.Popen(["open", "-a", "Microsoft PowerPoint"])
            speak("Microsoft PowerPoint has been opened for you, Sir")
        except Exception:
            speak("Sorry, Microsoft PowerPoint is not installed on your Mac.")

    elif system_platform == "Linux":
        try:
            subprocess.Popen(["libreoffice", "--impress"])
            speak("LibreOffice Impress has been opened for you, Sir")
        except FileNotFoundError:
            speak("Sorry, LibreOffice Impress is not installed on your Linux system.")
def find_file(filename, search_path="C:\\"):
    for root, dirs, files in os.walk(search_path):
        if filename in files:
            return os.path.join(root, filename)
    return None
def open_powerpoint():
    system_platform = platform.system()

    if system_platform == "Windows":
        ppt_path = find_file("POWERPNT.EXE", "C:\\")
        if ppt_path:
            subprocess.Popen([ppt_path])
            speak("Microsoft PowerPoint has been opened for you, Sir")
        else:
            speak("Sorry, Microsoft PowerPoint could not be found on your system.")

    elif system_platform == "Darwin":  # MacOS
        try:
            subprocess.Popen(["open", "-a", "Microsoft PowerPoint"])
            speak("Microsoft PowerPoint has been opened for you, Sir")
        except Exception:
            speak("Sorry, Microsoft PowerPoint is not installed on your Mac.")

    elif system_platform == "Linux":
        try:
            subprocess.Popen(["libreoffice", "--impress"])
            speak("LibreOffice Impress has been opened for you, Sir")
        except FileNotFoundError:
            speak("Sorry, LibreOffice Impress is not installed on your Linux system.")
def extract_text_from_pdf(filepath):
    try:
        with open(filepath, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            text = ""
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
            return text.strip()
    except Exception as e:
        print(f" Error reading PDF: {e}")
        return None
def read_aloud(text):
    if not text:
        print(" No text to read.")
        return
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()
def open_excel():
    system_platform = platform.system()

    if system_platform == "Windows":
        speak("Searching for Microsoft Excel on your system, please wait...")

        excel_path = find_file("EXCEL.EXE", "C:\\")  # scan entire C drive

        if excel_path:
            subprocess.Popen([excel_path])
            speak("Microsoft Excel has been opened for you, Sir")
        else:
            speak("Sorry, I could not find Microsoft Excel on your system.")

    elif system_platform == "Darwin":  # MacOS
        try:
            subprocess.Popen(["open", "-a", "Microsoft Excel"])
            speak("Microsoft Excel has been opened for you, Sir")
        except Exception:
            speak("Sorry, Microsoft Excel is not installed on your Mac.")

    elif system_platform == "Linux":
        try:
            subprocess.Popen(["libreoffice", "--calc"])
            speak("LibreOffice Calc has been opened for you, Sir")
        except FileNotFoundError:
            speak("Sorry, LibreOffice Calc is not installed on your Linux system.")
def open_word():
    system_platform = platform.system()

    if system_platform == "Windows":
        speak("Searching for Microsoft Word on your system, please wait...")

        word_path = find_file("WINWORD.EXE", "C:\\")  # scan entire C drive

        if word_path:
            subprocess.Popen([word_path])
            speak("Microsoft Word has been opened for you, Sir")
        else:
            speak("Sorry, I could not find Microsoft Word on your system.")

    elif system_platform == "Darwin":  # MacOS
        try:
            subprocess.Popen(["open", "-a", "Microsoft Word"])
            speak("Microsoft Word has been opened for you, Sir")
        except Exception:
            speak("Sorry, Microsoft Word is not installed on your Mac.")

    elif system_platform == "Linux":
        try:
            subprocess.Popen(["libreoffice", "--writer"])
            speak("LibreOffice Writer has been opened for you, Sir")
        except FileNotFoundError:
            speak("Sorry, LibreOffice Writer is not installed on your Linux system.")
def open_spotify():
    system_platform = platform.system()

    if system_platform == "Windows":
        speak("Searching for Spotify on your system, please wait...")

        # Common installation paths
        common_paths = [
            os.path.expanduser(r"~\AppData\Roaming\Spotify\Spotify.exe"),
            r"C:\Program Files\Spotify\Spotify.exe",
            r"C:\Program Files (x86)\Spotify\Spotify.exe"
        ]

        spotify_path = None
        for path in common_paths:
            if os.path.exists(path):
                spotify_path = path
                break

        # If not found in common locations, scan the whole C: drive
        if not spotify_path:
            spotify_path = find_file("Spotify.exe", "C:\\")

        if spotify_path:
            subprocess.Popen([spotify_path])
            speak("Spotify has been opened for you, Sir")
        else:
            speak("Sorry, I could not find Spotify on your system.")

    elif system_platform == "Darwin":  # MacOS
        try:
            subprocess.Popen(["open", "-a", "Spotify"])
            speak("Spotify has been opened for you, Sir")
        except Exception:
            speak("Sorry, Spotify is not installed on your Mac.")

    elif system_platform == "Linux":
        try:
            subprocess.Popen(["spotify"])
            speak("Spotify has been opened for you, Sir")
        except FileNotFoundError:
            speak("Sorry, Spotify is not installed on your Linux system.")
def find_file_ide(filename, search_path="E:\\"):
    """
    Searches recursively for a file in the given directory.
    Returns the first match found, else None.
    """
    for root, dirs, files in os.walk(search_path):
        if filename in files:
            return os.path.join(root, filename)
    return None
def open_sts():
    system_platform = platform.system()

    if system_platform == "Windows":
        speak("Searching for Spring Tool Suite on your system, please wait...")

        # Common installation directories
        common_paths = [
            r"E:\APPS\sts-4.27.0.RELEASE\SpringToolSuite4.exe"
        ]

        sts_path = None
        for path in common_paths:
            if os.path.exists(path):
                sts_path = path
                break

        # Fallback: scan entire C: drive
        if not sts_path:
            sts_path = find_file_ide("SpringToolSuite4.exe", "E:\\")

        if sts_path:
            subprocess.Popen([sts_path])
            speak("Spring Tool Suite has been opened for you, Sir")
        else:
            speak("Sorry, I could not find Spring Tool Suite on your system.")

    elif system_platform == "Darwin":  # macOS
        try:
            subprocess.Popen(["open", "-a", "SpringToolSuite4"])
            speak("Spring Tool Suite has been opened for you, Sir")
        except Exception:
            speak("Sorry, Spring Tool Suite is not installed on your Mac.")

    elif system_platform == "Linux":
        try:
            subprocess.Popen(["sts"])  # if added to PATH
            speak("Spring Tool Suite has been opened for you, Sir")
        except FileNotFoundError:
            # Try finding the launcher manually
            sts_path = find_file("SpringToolSuite4", "/")
            if sts_path:
                subprocess.Popen([sts_path])
                speak("Spring Tool Suite has been opened for you, Sir")
            else:
                speak("Sorry, I could not find Spring Tool Suite on your Linux system.")
def train_model(dataset_dir="known_faces"):
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    faces, labels, names = [], [], {}
    label_id = 0

    for person_name in os.listdir(dataset_dir):
        person_path = os.path.join(dataset_dir, person_name)
        if not os.path.isdir(person_path):
            continue

        names[label_id] = person_name
        for img_file in os.listdir(person_path):
            img_path = os.path.join(person_path, img_file)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue

            # Resize training images to a consistent size
            resized_img = cv2.resize(img, (200, 200))
            faces.append(resized_img)
            labels.append(label_id)

        label_id += 1

    if not faces:
        raise RuntimeError("No training data found!")

    recognizer.train(faces, np.array(labels))
    return recognizer, names
def unlock_with_face(recognizer, names):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return None

    face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    authorized = None
    speak("Please look at the camera for face recognition.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to grab frame.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            # Draw rectangle around the detected face
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

            face_img = gray[y:y + h, x:x + w]
            face_resized = cv2.resize(face_img, (200, 200))

            # Prediction
            label, confidence = recognizer.predict(face_resized)

            print(f"Detected face. Predicted label: {label}, Confidence: {confidence}")

            # Confidence check: Lower confidence is better match
            if confidence < 80:  # Adjust threshold as needed
                authorized = names.get(label)
                if authorized:
                    speak(f"Access granted. Welcome {authorized}.")
                    # Release resources and exit
                    cap.release()
                    cv2.destroyAllWindows()
                    return authorized
            # Display text for a better user experience
            if confidence < 95:
                text = f"Authorized: {names.get(label)}"
                color = (0, 255, 0)
            else:
                text = "Unauthorized"
                color = (0, 0, 255)

            cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

        cv2.imshow("Face Unlock", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    speak("Access denied.")
    return None
def main_voice_assistant():
    wish()
    log_to_db_success("Wish()", f"Wish() has been executed", "SUCCESS")
    tell_time()
    log_to_db_success("tell_time()", f"tell_time() has been executed", "SUCCESS")
    speak("This is your personal voice assistant AI, Pashupathasthra. How can I help you?")
    while True:
        query = takeCommand()

        if query is None or query == "none":
            print("No command recognized. Listening again...")
            continue
        query = query.lower()
        log_to_db_success("Query", query, "SUCCESS")
        # logic building for tasks
        if "open notepad" in query:
            speak("Opening the notepad for you, Sir. Please wait for some time")
            system_platform = platform.system()

            if system_platform == "Windows":
                try:
                    subprocess.Popen(["notepad.exe"])
                    speak("Notepad has been opened for you, Sir")
                    log_to_db_success("Open Notepad", f"Notepad has been opened and text is written and file has been saved", "SUCCESS")
                except FileNotFoundError as e:
                    speak("Sorry, I could not open Notepad on your system.")
                    log_to_db_error("Open Notepad", str(e), "FAILURE")
            elif system_platform == "Darwin":  # MacOS
                # macOS doesn't have Notepad, so we open TextEdit instead
                try:
                    subprocess.Popen(["open", "-a", "TextEdit"])
                    speak("TextEdit has been opened for you, Sir")
                    log_to_db_success("Open Notepad",
                                      f"Notepad has been opened and text is written and file has been saved", "SUCCESS")
                except Exception:
                    speak("Sorry, I could not open TextEdit on your Mac.")
                    log_to_db_error("Open Notepad", str(e), "FAILURE")
            elif system_platform == "Linux":
                # Linux systems may have different editors; we try gedit first
                try:
                    subprocess.Popen(["gedit"])
                    speak("Gedit has been opened for you, Sir")
                    log_to_db_success("Open Notepad",
                                      f"Notepad has been opened and text is written and file has been saved", "SUCCESS")
                except FileNotFoundError:
                    try:
                        subprocess.Popen(["nano"])
                        speak("Nano editor has been opened in terminal, Sir")
                        log_to_db_success("Open Notepad",
                                          f"Notepad has been opened and text is written and file has been saved",
                                          "SUCCESS")
                    except FileNotFoundError:
                        speak("Sorry, I could not find a text editor on your Linux system.")
                        log_to_db_error("Open Notepad", str(e), "FAILURE")
            notepad_open()
        elif "open spring" in query:
            try:
                speak("Opening Spring tool suite for you sir. Kindly wait for a few minutes")
                open_sts()
                log_to_db_success("Open Spring Tool Suite",
                                  f"Spring Tool Suite has been opened and you can write your code",
                                  "SUCCESS")
                time.sleep(1.5)
            except Exception as e:
                speak("Could not understand what you said ")
                log_to_db_error("Open Spring Tool Suite", str(e), "FAILURE")
        elif "cmd" in query:
            try:
                speak("Opening command prompt terminal for you sir")
                os.system("start cmd")
                time.sleep(1.5)
                log_to_db_success("Open Command Prompt",
                                  f"Commmand Prompt has been opened",
                                  "SUCCESS")
                speak("Command Prompt Terminal is successfully opened. Would you like to have anything else, Sir?")
            except Exception as e:
                speak("Oops some error has occured")
                log_to_db_error("Open Command Prompt", str(e), "FAILURE")

        elif "open spotify" in query:
            try:
                speak("Opening spotify for you sir")
                open_spotify()
                time.sleep(1.5)
                log_to_db_success("Open Spotify",
                                  f"Spotify has been opened",
                                  "SUCCESS")
                speak("Spotify is opened for you sir. Would you like to have anything else, Sir?")
            except Exception as e:
                speak("Oops some error has occured")
                log_to_db_error("Open Spotify", str(e), "FAILURE")
        elif "open youtube" in query:
            try:
                speak("Youtube has been opened for you sir. What song would you like to play?")
                song_name = get_voice_command()
                if song_name:
                    video_url = get_youtube_url(song_name)
                    if video_url:
                        open_in_chrome(video_url)
                        log_to_db_success("Open youtube",
                                          f"Youtube has been opened",
                                          "SUCCESS")
                    else:
                        print("Could not find any video.")
                        log_to_db_error("Open Spotify", "Could not find any video. ", "FAILURE")
            except Exception as e:
                speak("Oops some error has occured")
                log_to_db_error("Open Youtube", str(e), "FAILURE")
        elif "open translator" in query:
            try:
                asyncio.run(listen_and_translate())
                log_to_db_success("Open Translator",
                                  f"Translator has been opened",
                                  "SUCCESS")
            except Exception as e:
                speak("Oops some error has occured")
                log_to_db_error("Open Translator", str(e), "FAILURE")
        elif "open weather" in query:
            try:
                weather_detection()
                log_to_db_success("Open Weather Detection",
                                  f"Weather Detection has been opened",
                                  "SUCCESS")
            except Exception as e:
                speak("Oops some error has occured")
                log_to_db_error("Open Translator", str(e), "FAILURE")
        elif "open ip address" in query:
            try:
                hostname = socket.gethostname()
                ip = socket.gethostbyname(hostname)
                log_to_db_success("Open IP Address",
                                  f"Hostname: {hostname}, IP: {ip}",
                                  "SUCCESS")
                speak(f"Your Ip address is {ip}")
            except Exception as e:
                print("Error", e)
                speak("Oops some error has occured")
                log_to_db_error("Open Ip Address", str(e), "FAILURE")
        elif "open word" in query:
            try:
                speak("Opening MS word for you, Sir......")
                open_word()
                time.sleep(1.5)
                word_open()
                log_to_db_success("Open Microsoft Word",
                                  f"Microsoft Word is opened for you",
                                  "SUCCESS")
            except Exception as e:
                speak("Oops some error has occured")
                log_to_db_error("Open Microsoft Word", str(e), "FAILURE")
        elif "open powerpoint" in query:
            try:
                speak("Opening Microsoft Powerpoint")
                open_powerpoint()
                time.sleep(1.5)
                log_to_db_success("Open Microsoft Powerpoint",
                                  f"Microsoft Powerpoint is opened for you",
                                  "SUCCESS")
            except Exception as e:
                speak("Oops some error has occured")
                log_to_db_error("Open Powerpoint", str(e), "FAILURE")
        elif "open excel" in query:
            try:
                speak("Opening Microsoft Excel for you")
                open_excel()
                speak("Microsoft Excel has been opened for you sir")
                log_to_db_success("Open Microsoft Excel",
                                  f"Microsoft Excel is opened for you",
                                  "SUCCESS")
            except Exception as e:
                speak("Oops some error has occured")
                log_to_db_error("Open Microsoft Excel", str(e), "FAILURE")
        elif "open wikipedia" in query:
            try:
                topic = listen_wikipedia_command()
                if topic:
                    result = search_wikipedia_info(topic)
                    speak(f"According to wikipedia, {result}")
                speak("Hope I have told you everything you have asked for Sir")
                log_to_db_success("Open Wikipedia",
                                  f"Wikipedia is opened for you",
                                  "SUCCESS")
            except Exception as e:
                speak("Oops some error has occured")
                log_to_db_error("Open Wikipedia", str(e), "FAILURE")
        elif "open facebook" in query:
            try:
                login_facebook()
                log_to_db_success("Open Facebook",
                                  f"Facebook is opened for you",
                                  "SUCCESS")
            except Exception as e:
                speak("Oops some error has occured")
                log_to_db_error("Open Facebook", str(e), "FAILURE")
        elif "location" in query:
            try:
                location()
                log_to_db_success("Open Location",
                                  f"Location is opened for you",
                                  "SUCCESS")
            except Exception as e:
                speak("Oops some error has occured")
                log_to_db_error("Open Location", str(e), "FAILURE")
        elif "open instagram" in query:
            try:
                speak("Opening your instagram profile for you Sir")
                webbrowser.open("https://www.instagram.com/suman.talukdar53/")
                speak("I have opened your instagram profile for you Sir")
                log_to_db_success("Open Instagram",
                                  f"Instagram is opened for you",
                                  "SUCCESS")
            except Exception as e:
                speak("Oops some error has occured")
                log_to_db_error("Open Instagram", str(e), "FAILURE")
        elif "open linkedin" in query:
            try:
                speak("Opening your Linkedin profile for you Sir")
                webbrowser.open("https://www.linkedin.com/in/suman-talukdar-29b3352b6")
                speak("I have opened your linkedin profile for you, Sir")
                log_to_db_success("Open Linkedin",
                                  f"Linkedin is opened for you",
                                  "SUCCESS")
            except Exception as e:
                speak("Oops some error has occured")
                log_to_db_error("Open Linkedin", str(e), "FAILURE")
        elif "open github" in query:
            try:
                speak("Opening your Github profile for you")
                webbrowser.open("https://github.com/jiraiyasuman")
                speak("I have opened the github profile for you")
                log_to_db_success("Open Github",
                                  f"Github is opened for you",
                                  "SUCCESS")
            except Exception as e:
                speak("Oops some error has occured")
                log_to_db_error("Open Github", str(e), "FAILURE")
        elif "open google" in query:
            try:
                search_google()
                log_to_db_success("Open Google",
                              f"Google is opened for you",
                              "SUCCESS")
            except Exception as e:
                speak("Oops some error has occured")
                log_to_db_error("Open Google", str(e), "FAILURE")
        elif "open library genesis" in query:
            try:
                speak("Opening Library Genesis for you sir")
                webbrowser.open("https://libgen.gs/")
                speak("I have opened Library Genesis for you")
                log_to_db_success("Open Library Genesis",
                                  f"Library Genesis is opened for you",
                                  "SUCCESS")
            except Exception as e:
                speak("Oops some error has occured")
                log_to_db_error("Open Library Genesis", str(e), "FAILURE")
        elif "open developer doc" in query:
           try:
               speak("Opening Developer docs for you sir")
               webbrowser.open("https://devdocs.io/")
               speak("I have opened developer docs for you sir")
               log_to_db_success("Open Developer Docs",
                                 f"Developer Docs is opened for you",
                                 "SUCCESS")
           except Exception as e:
               speak("Oops some error has occured")
               log_to_db_error("Open Developer Docs", str(e), "FAILURE")
        elif "open email" in query:
            try:
                voice_email_details()
                log_to_db_success("Open Email",
                                   f"GMail is opened for you",
                                  "SUCCESS")
            except Exception as e:
                speak("Oops some error has occured")
                log_to_db_error("Open Email", str(e), "FAILURE")
        elif "age" in query or "gender" in query or "facial emotion" in query:
            try:
                capture_facial_image()
                log_to_db_success("Open Emotion Detection",
                                  f"Emotion Detection is opened for you",
                                  "SUCCESS")
            except Exception as e:
                speak("Oops some error has occured")
                log_to_db_error("Open Emotion Detection", str(e), "FAILURE")
        elif "open news" in query:
            try:
                speak("Please wait for a few minutes . Fetching and telling news for you")
                get_news()
                log_to_db_success("Open News",
                                  f"News is opened for you",
                                  "SUCCESS")
            except Exception as e:
                speak("Oops some error has occured")
                log_to_db_error("Open News", str(e), "FAILURE")
        elif "switch tabs" in query:
            try:
                speak("Switching tabs for you sir!")
                pyautogui.keyDown("alt")
                pyautogui.press("tab")
                time.sleep(2)
                pyautogui.keyUp("alt")
                speak("Tabs have been switched for you sir")
                log_to_db_success("Switch Tabs",
                                  f"Tabs switch has been done for you",
                                  "SUCCESS")
            except Exception as e:
                speak("Oops some error has occured")
                log_to_db_error("Switch Tabs", str(e), "FAILURE")
        elif "shutdown" in query:
            try:
                speak("Shutting down your system")
                log_to_db_success("System shut down",
                                  f"System shut down has been done for you",
                                  "SUCCESS")
                system_shutdown()
            except Exception as e:
                speak("Oops some error has occured")
                log_to_db_error("Shutting down", str(e), "FAILURE")
        elif "restart" in query:
            try:
                speak("Restarting down your system")
                log_to_db_success("System restart",
                                  f"System restart has been done for you",
                                  "SUCCESS")
                system_restart()
            except Exception as e:
                speak("Oops some error has occured")
                log_to_db_error("System restart", str(e), "FAILURE")
        elif "sleep" in query:
            try:
                speak("Putting the system to sleep")
                log_to_db_success("System sleep",
                                  f"System sleep has been done for you",
                                  "SUCCESS")
                system_sleep()
            except Exception as e:
                speak("Oops some error has occured")
                log_to_db_error("System restart", str(e), "FAILURE")
        elif "chat" in query:
            try:
                question = takeCommand().lower()
                if "hello" in question or "hi" in question:
                    if "male" in gender:
                        speak("Hello sir! My name is Pashupathasthra. I am your personal voice assistant AI")
                    else:
                        speak("Hello maam! My name is Pashupathasthra. I am your personal voice assistant AI")
                elif "how are you" in question:
                    if "male" in gender:
                        speak("I am fine, sir")
                    else:
                        speak("I am fine maam")
                else:
                    history = [{"role": "system", "content": "You are a helpful AI assistant."}]
                    response = main(question, history)
                    history.append({"role": "user", "content": query})
                    history.append({"role": "assistant", "content": response})
                    speak(response)
                log_to_db_success("Chat",
                                  f"chat has been done for you",
                                  "SUCCESS")
            except Exception as e:
                speak("Oops some error has occured")
                log_to_db_error("Chat Bot", str(e), "FAILURE")
        elif "goodbye" in query:
            try:
                speak("Goodbye ..... Nice interacting with you.....")
                log_to_db_success("GoodBye",
                                  f"Exit has been done for you",
                                  "SUCCESS")
                exit()
            except Exception as e:
                speak("Oops some error has occured")
                log_to_db_error("Goodbye", str(e), "FAILURE")
        elif "pdf reader" in query:
            try:
                speak("Please tell the name of the file you want to read?")
                filename = listen_for_filename()
                if not filename:
                    return

                filepath = os.path.join(PDF_FOLDER, filename)
                if not os.path.exists(filepath):
                    print(f" File not found: {filepath}")
                    return

                print(" Reading the file...")
                text = extract_text_from_pdf(filepath)
                read_aloud(text)
                log_to_db_success("File Reader",
                                  f"File Reader has been done for you",
                                  "SUCCESS")
            except Exception as e:
                speak("Oops some error has occured")
                log_to_db_error("File Reader", str(e), "FAILURE")
        elif "live updates" in query:
            try:
                speak("Which news site do you want me to open for you...")
                name = takeCommand().lower()
                if "bbc" in name:
                    speak("Opening BBC news for you")
                    webbrowser.open("https://www.bbc.com/news")
                elif "cnn" in name:
                    speak("Opening CNN for you")
                    webbrowser.open("https://www.cnn.com/")
                elif "new york times" in name:
                    speak("Opening new york times for you")
                    webbrowser.open("https://www.nytimes.com/")
                elif "npr" in name:
                    speak(" Opening npr news for you")
                    webbrowser.open("https://www.npr.org/")
                elif "al jazeera" in name:
                    speak("Opening Al jazeera news for you")
                    webbrowser.open("https://www.aljazeera.com/")
                else:
                    speak("Sorry sir .. could not open this news site for you")
                log_to_db_success("Live Updates",
                                  f"Live Updates has been done for you",
                                  "SUCCESS")
            except Exception as e:
                speak("Oops some error has occured")
                log_to_db_error("Live Update", str(e), "FAILURE")
        else:
            continue
# Main program
if __name__ == "__main__":
      try:
         recognizer, names = train_model("known_faces")
         user = unlock_with_face(recognizer, names)
         if user:
             log_to_db_success("Face Recognition",
                               f"Face Recognition is being executed successfully", "SUCCESS")
             main_voice_assistant()
      except Exception as e:
         speak("Setup error. Please check your dataset.")
         log_to_db_error("Face Recognition", str(e), "FAILURE")
    #main_voice_assistant()
