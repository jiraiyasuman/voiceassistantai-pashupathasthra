import os
import speech_recognition as sr
import pyttsx3
import PyPDF2

# Folder where PDF files are stored
PDF_FOLDER = "./pdfs"  # make sure this folder exists and contains your files

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

def main():
    pyttsx3.speak("Please tell the name of the file you want to read?")
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

if __name__ == "__main__":
    main()
