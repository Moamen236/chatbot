import tkinter as tk
from tkinter import scrolledtext
import nltk
from nltk.stem.lancaster import LancasterStemmer
import requests
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense, Dropout
import random
import json
import mysql.connector
from datetime import date

# Load the intents data from the file
with open('intents.json') as file:
    data = json.load(file)

# Create the stemmer
stemmer = LancasterStemmer()

# Load the trained model
model = keras.models.load_model('model.h5')

# Connect to the database
database = mysql.connector.connect(
    host="localhost",
    user="root",
    password="",
    database="gym_system"
)

# Create a cursor object to execute SQL queries
databaseCursor = database.cursor(dictionary=True)

# Function to insert user data
def userData(name, phone):
    insertUser = "INSERT INTO users (name, phone, created_at, updated_at) VALUES (%s, %s, NOW(), NOW())"
    val = (name, phone, )
    databaseCursor.execute(insertUser, val)
    database.commit()
    return databaseCursor.lastrowid

# Function to book a class
def book_class(chosenType, chosenClass):
    # Execute a SELECT query available classes in this type
    classes = "SELECT * FROM classes WHERE type_id = %s"
    val = (int(chosenType), )
    databaseCursor.execute(classes, val)
    classesResult = databaseCursor.fetchall()

    # Execute a SELECT query to check if the class is available
    availableClass = "SELECT * FROM classes WHERE id = %s"
    val = (int(chosenClass), )
    databaseCursor.execute(availableClass, val)
    classResult = databaseCursor.fetchone()

    # If the class is full, return a message saying it's unavailable
    if classResult['max_capacity'] == 0:
        return "Sorry, that class is already full. Please choose a different time or class type."
    else:
        # Otherwise, decrement the class's capacity and insert a new booking record
        updateCapacity = "UPDATE classes SET max_capacity = max_capacity - 1 WHERE id = %s"
        updateCapacityValues = (int(chosenClass), )
        databaseCursor.execute(updateCapacity, updateCapacityValues)

        createBooking = "INSERT INTO bookings (class_id, user_id, created_at, updated_at) VALUES (%s, %s, NOW(), NOW())"
        createBookingValues = (int(chosenClass), int(user_id), )
        databaseCursor.execute(createBooking, createBookingValues)

        database.commit()

        # Return a confirmation message
        return "You're booked for a {} class on {}!".format(classResult['name'], classResult['date'])

# Function to determine diet
def determine_diet(dietary_restrictions, goals):
    # Set up the API endpoint and headers
    url = "https://trackapi.nutritionix.com/v2/natural/nutrients"
    headers = {"x-app-id": "acf12833", "x-app-key": "6e7b86d785c31d0a67ba243a543d8e26", "Content-Type": "application/json"}

    # Set up the request body
    query = dietary_restrictions
    data = {"query": query, "timezone": "US/Eastern"}

    # Make the API call and retrieve the results
    try:
        response = requests.post(url, headers=headers, json=data)
        results = response.json()["foods"]
    except (requests.exceptions.HTTPError, KeyError):
        return "Sorry, there was a problem with your request. Please try again later."

    # Filter the results based on the user's goals
    filtered_results = []
    for result in results:
        if int(result["nf_calories"]) <= int(goals):
            filtered_results.append(result)

    # Format the results as a string
    meal_plans = ""
    for result in filtered_results:
        name = result["food_name"]
        calories = result["nf_calories"]
        meal_plans += "{}: {} calories\n".format(name, calories)

    # Return the meal plans
    return "Here are some meal plans that might work for you:\n{}".format(meal_plans)

# Function to get an answer to a frequently asked question
def get_faq_answer(question):
    # Find the faq intent and retrieve the corresponding response
    faq_intent = next((intent for intent in data['intents'] if intent['tag'] == 'faq'), None)
    if faq_intent is not None:
        for i in range(len(faq_intent['patterns'])):
            if question.lower() in faq_intent['patterns'][i].lower():
                return faq_intent['responses'][i]

    # If no match is found, display a message to the user
    unrecognized_message = "I'm sorry, I didn't understand your question. Here are some common questions you can ask:\n"
    unrecognized_message += "\n".join(faq_intent['patterns'])
    return unrecognized_message

# Function to generate a response to the user's message
def generate_response(model, words, classes, message):
    # Tokenize and stem the message
    message_words = nltk.word_tokenize(message)
    message_words = [stemmer.stem(w.lower()) for w in message_words]

    # Create a bag of words representation of the message
    bag = [1 if w in message_words else 0 for w in words]

    # Use the model to predict the class of the message
    prediction = model.predict(np.array([bag]))[0]
    threshold = 0.25
    prediction = [[i, p] for i, p in enumerate(prediction) if p > threshold]

    prediction.sort(key=lambda x: x[1], reverse=True)

    response = ""
    if not prediction:
        return "I'm sorry, I didn't understand your message. Please try again."

    # Loop through the predictions and find the most likely class
    for pred in prediction:
        pred_class = classes[pred[0]]
        if pred_class == "greeting":
            response = random.choice(data['intents'][0]['responses'])
        elif pred_class == "goodbye":
            response = random.choice(data['intents'][1]['responses'])
        elif pred_class == "book_class":
            response = book_class(chosenType=pred[1], chosenClass=pred[2])
        elif pred_class == "determine_diet":
            response = determine_diet(dietary_restrictions=pred[1], goals=pred[2])
        elif pred_class == "get_faq_answer":
            response = get_faq_answer(question=message)
        else:
            response = "I'm sorry, I didn't understand your message. Please try again."

    return response

# Create the GUI
root = tk.Tk()
root.title("Chatbot")

# Create the chat window
chat_window = tk.Frame(root, bg="white")
chat_window.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

# Create the chat history window
chat_history = scrolledtext.ScrolledText(chat_window, state=tk.DISABLED)
chat_history.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

# Function to send a message to the chat history window
def send_to_chat_history(message):
    chat_history.configure(state=tk.NORMAL)
    chat_history.insert(tk.END, message + '\n\n')
    chat_history.configure(state=tk.DISABLED)
    chat_history.see(tk.END)

# Create the message input field
message_input = tk.Entry(root, width=80, bg="white")
message_input.pack(side=tk.LEFT, padx=10, pady=10)

# Function to handle sending a message
def send_message(event=None):
    message = message_input.get()
    message_input.delete(0, tk.END)
    send_to_chat_history("You: {}".format(message))
    response = generate_response(model=model, words=words, classes=classes, message=message)
    send_to_chat_history("Chatbot: {}".format(response))

# Create the send button
send_button = tk.Button(root, text="Send", command=send_message, bg="white")
send_button.pack(side=tk.LEFT, padx=10, pady=10)

# Bind the return key to the send message function
root.bind('<Return>', send_message)

# Load the words and classes from the training data
with open('words.json') as file:
    words = json.load(file)

with open('classes.json') as file:
    classes = json.load(file)

# Run the GUI
root.mainloop()