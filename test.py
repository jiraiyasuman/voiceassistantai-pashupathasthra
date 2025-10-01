import sqlite3
from datetime import datetime
import time


# Placeholder functions (assuming these exist elsewhere in your code)
def speak(text):
    print(text)


def system_sleep():
    print("System is going to sleep...")
    time.sleep(3)  # Simulate system sleep
    print("System is awake.")


import mysql.connector
from datetime import datetime


def log_to_db_success(action, result, status):
    try:
        # Connect to MySQL Workbench
        conn = mysql.connector.connect(
            host="localhost",  # e.g., "localhost"
            user="root",  # e.g., "root"
            password="12345",  # Your MySQL password
            database="pashupathastra_ai"
        )
        cursor = conn.cursor()

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


# Example Usage:
# log_to_db_success("System sleep", "System sleep has been done for you", "SUCCESS")
def log_to_db_error(action, result, status):
    # This function would be similar but for logging errors.
    print(f"ERROR: {action}, {result}, {status}")


# Example usage (your original code snippet)
query = "sleep"  # Simulate user command

if "sleep" in query:
    try:
        speak("Putting the system to sleep")
        # Now, this call will correctly execute the function
        log_to_db_success("System sleep",
                          "System sleep has been done for you",
                          "SUCCESS")
        system_sleep()
    except Exception as e:
        speak("Oops some error has occured")
        log_to_db_error("System sleep", str(e), "FAILURE")