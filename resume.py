"""
voice_scheduler.py
Simple voice-driven appointment scheduler storing to MySQL and reading today's appointments.

Requirements:
  pip install SpeechRecognition pyttsx3 dateparser mysql-connector-python
"""

import speech_recognition as sr
import pyttsx3
import dateparser
import mysql.connector
from datetime import datetime, date, timedelta
import re
import sys
import time

# ---------- Configuration: update these ----------
DB_CONFIG = {
    "host": "localhost",
    "user": "root",
    "password": "12345",
    "database": "schedule_app",
}
# -------------------------------------------------

# Initialize recognizer and TTS engine
recognizer = sr.Recognizer()
tts = pyttsx3.init()
tts.setProperty("rate", 170)  # speaking rate

def speak(text):
    print("TTS:", text)
    tts.say(text)
    tts.runAndWait()

def listen(timeout=6, phrase_time_limit=8):
    """Listen from microphone and return recognized text (lowercase)."""
    with sr.Microphone() as source:
        recognizer.adjust_for_ambient_noise(source, duration=0.6)
        print("Listening...")
        try:
            audio = recognizer.listen(source, timeout=timeout, phrase_time_limit=phrase_time_limit)
        except sr.WaitTimeoutError:
            print("No speech detected (timeout).")
            return ""
    try:
        # This uses Google Web Speech API by default (requires internet).
        text = recognizer.recognize_google(audio)
        print("Heard:", text)
        return text.lower()
    except sr.UnknownValueError:
        print("Could not understand audio.")
        return ""
    except sr.RequestError as e:
        print("Speech service error:", e)
        speak("Sorry, speech recognition service is unavailable.")
        return ""

# ---------- Database helpers ----------
def get_db_connection():
    return mysql.connector.connect(**DB_CONFIG)

def save_appointment(title, description, start_dt, end_dt=None, location=None):
    conn = get_db_connection()
    try:
        cur = conn.cursor()
        sql = ("INSERT INTO appointments (title, description, location, start_time, end_time) "
               "VALUES (%s, %s, %s, %s, %s)")
        cur.execute(sql, (title, description, location, start_dt, end_dt))
        conn.commit()
        print("Saved appointment:", title, start_dt)
    finally:
        cur.close()
        conn.close()

def fetch_appointments_between(start_dt, end_dt):
    conn = get_db_connection()
    try:
        cur = conn.cursor(dictionary=True)
        cur.execute("SELECT * FROM appointments WHERE start_time >= %s AND start_time < %s ORDER BY start_time",
                    (start_dt, end_dt))
        rows = cur.fetchall()
        return rows
    finally:
        cur.close()
        conn.close()

# ---------- Parsing helpers ----------
def parse_datetime_from_text(text, prefer_future=True, base_dt=None):
    """
    Uses dateparser to find a datetime in text.
    Returns a datetime object or None.
    """
    settings = {"PREFER_DATES_FROM": "future" if prefer_future else "past",
                "RETURN_AS_TIMEZONE_AWARE": False}
    if base_dt:
        settings["RELATIVE_BASE"] = base_dt
    dt = dateparser.parse(text, settings=settings)
    return dt

def extract_title(text):
    """
    Try to extract a short title from the phrase.
    We'll attempt heuristics: after 'schedule' or 'set' up to 'on'/'at'/'for'.
    """
    # remove scheduling verbs
    text = re.sub(r"\b(schedule|set up|set|create|add|book|make)\b", "", text)
    # split by on/at/for/from/to/at time
    parts = re.split(r"\b(on|at|for|from|to|next|this)\b", text)
    if parts:
        title = parts[0].strip(" ,.")
        if len(title) > 0:
            return title[:200]
    return text.strip()[:200]

def extract_time_text(text):
    """
    Try to extract the time-related fragment for parsing.
    This is a naive approach: look for 'on', 'at', 'tomorrow', weekdays, dates, times.
    """
    # common time words
    time_words = r"(today|tomorrow|tonight|next|monday|tuesday|wednesday|thursday|friday|saturday|sunday|am|pm|[0-9]{1,2}[:.][0-9]{2}|[0-9]{1,2} (am|pm)|[0-9]{1,2} o'clock|on|at|this|coming|next|in the morning|in the evening|noon|midnight|afternoon)"
    matches = re.findall(time_words, text, flags=re.IGNORECASE)
    if matches:
        # return the whole text as fallback
        return text
    # fallback: return whole text to let dateparser search
    return text

# ---------- High-level intents ----------
def intent_schedule(text):
    """Return True if user intends to schedule something."""
    triggers = ["schedule", "set up", "set", "create", "add", "book", "make appointment", "schedule an appointment", "meeting", "meet with"]
    return any(t in text for t in triggers)

def intent_read_today(text):
    """Return True if user wants today's appointments."""
    triggers = ["today", "todays", "today's", "what's my schedule", "what is my schedule", "appointments today", "today appointments", "read today's appointments", "read my schedule"]
    return any(t in text for t in triggers)

# ---------- Interaction flows ----------
def schedule_flow(spoken_text):
    # Heuristic parsing
    title = extract_title(spoken_text)
    time_text = extract_time_text(spoken_text)
    dt = parse_datetime_from_text(time_text)
    if dt is None:
        # ask the user for date/time
        speak("I didn't catch the time. When is the appointment?")
        reply = listen()
        if not reply:
            speak("Sorry, couldn't get the time. Try again.")
            return
        dt = parse_datetime_from_text(reply)
        if dt is None:
            speak("I couldn't parse the time. Please enter the date and time manually.")
            return

    # ask for duration or end time
    speak("How long is the appointment in minutes, or say end time?")
    reply = listen()
    end_dt = None
    duration_minutes = None
    if reply:
        # try to parse simple "X minutes" or parse datetime
        m = re.search(r"(\d{1,4})\s*(minutes|minute|mins|min)", reply)
        if m:
            duration_minutes = int(m.group(1))
            end_dt = dt + timedelta(minutes=duration_minutes)
        else:
            # try parse end time phrase
            parsed_end = parse_datetime_from_text(reply, base_dt=dt)
            if parsed_end:
                end_dt = parsed_end

    # ask for location/description
    speak("Do you want to add a location or short description? If yes, say it now or say 'no'.")
    reply = listen()
    location = None
    description = None
    if reply and reply.strip().lower() not in ("no", "none"):
        description = reply
        # try to extract a location phrase if the user mentions 'at <place>'
        m = re.search(r"\bat\s+(.*)", reply)
        if m:
            location = m.group(1)

    # final confirmation
    start_str = dt.strftime("%Y-%m-%d %H:%M")
    if end_dt:
        end_str = end_dt.strftime("%Y-%m-%d %H:%M")
        speak(f"Scheduling {title} on {start_str} until {end_str}. Should I save it?")
    else:
        speak(f"Scheduling {title} on {start_str}. Should I save it?")

    confirm = listen()
    if "yes" in confirm or "save" in confirm or "sure" in confirm:
        save_appointment(title=title or "Appointment",
                         description=description,
                         start_dt=dt,
                         end_dt=end_dt,
                         location=location)
        speak("Appointment saved.")
    else:
        speak("Okay, not saved.")

def read_todays_appointments():
    today = date.today()
    start_dt = datetime.combine(today, datetime.min.time())
    end_dt = start_dt + timedelta(days=1)
    rows = fetch_appointments_between(start_dt, end_dt)
    if not rows:
        speak("You have no appointments scheduled for today.")
        return
    speak(f"You have {len(rows)} appointment{'s' if len(rows)>1 else ''} today.")
    for r in rows:
        st = r["start_time"]
        title = r["title"]
        loc = r.get("location") or ""
        # format time nicely
        tstr = st.strftime("%I:%M %p") if isinstance(st, datetime) else str(st)
        summary = f"At {tstr}, {title}"
        if loc:
            summary += f" at {loc}"
        speak(summary)
        time.sleep(0.5)

# ---------- Main loop ----------
def main_loop():
    speak("Voice scheduler started. Say 'schedule' to add an appointment, or say 'what's my schedule today' to hear today's appointments. Say 'exit' to quit.")
    while True:
        text = listen()
        if not text:
            continue

        if "exit" in text or "quit" in text or "stop" in text:
            speak("Goodbye.")
            break

        if intent_read_today(text):
            read_todays_appointments()
            continue

        if intent_schedule(text):
            schedule_flow(text)
            continue

        # fallback small talk commands
        if "help" in text:
            speak("Say schedule to create an appointment, or say what's my schedule today.")
            continue

        speak("I didn't understand. Please say 'schedule' or 'what's my schedule today' or 'exit'.")

if __name__ == "__main__":
    try:
        main_loop()
    except KeyboardInterrupt:
        print("Exiting.")
        sys.exit(0)
