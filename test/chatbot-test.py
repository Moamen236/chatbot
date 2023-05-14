Sure, here's the continued code with the GUI implementation:

```
import nltk
import tkinter as tk
from tkinter import scrolledtext
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()
import requests

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense, Dropout

import random
import json
import mysql.connector
from datetime import date

# Create the main window
window = tk.Tk()
window.title("Chatbot")

# Create the chat history text area
chat_history = scrolledtext.ScrolledText(window, width=50, height=20)
chat_history.configure(state='disabled')
chat_history.pack(padx=10, pady=10)

# Create the user input text box
user_input = tk.Entry(window, width=50)
user_input.pack(padx=10, pady=10)

with open('intents.json') as file:
    data = json.load(file)

# Modify the data to include gym-related intents

words = []
classes = []
documents = []
ignore_words = ['?', '!']

for intent in data['intents']:
    for pattern in intent['patterns']:
        # Tokenize the pattern and add the words to the list of words
        w = nltk.word_tokenize(pattern)
        words.extend(w)
        # Add the pattern and its associated class to the list of documents
        documents.append((w, intent['tag']))
        # Add the class to the list of classes
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

# Stem and lowercase the words, and remove duplicates
words = [stemmer.stem(w.lower()) for w in words if w not in ignore_words]
words = sorted(list(set(words)))

# Sort the classes
classes = sorted(list(set(classes)))

# Create training data
training = []
output = []
output_empty = [0] * len(classes)

for doc in documents:
    # Tokenize and stem the words in the document
    pattern_words = [stemmer.stem(w.lower()) for w in doc[0]]
    # Create a bag of words representation of the pattern
    bag = [1 if w in pattern_words else 0 for w in words]
    # Create the output for the document
    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1
    # Add the bag of words representation and output to the training data
    training.append(bag)
    output.append(output_row)

# Convert the training data and output to numpy arrays
training = np.array(training)
output = np.array(output)

model = keras.Sequential()
model.add(Dense(128, input_shape=(len(training[0]),), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(output[0]), activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(training, output, epochs=1000, batch_size=8)

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
def userData():
    chat_history.configure(state='normal')
    chat_history.insert(tk.END, "Bot : Please Enter your name :\n")
    chat_history.configure(state='disabled')
    name = user_input.get()
    user_input.delete(0, tk.END)
    chat_history.configure(state='normal')
    chat_history.insert(tk.END, "You: " + name + "\n")
    chat_history.configure(state='disabled')
    
    chat_history.configure(state='normal')
    chat_history.insert(tk.END, "Bot : Please Enter your phone :\n")
    chat_history.configure(state='disabled')
    phone = user_input.get()
    user_input.delete(0, tk.END)
    chat_history.configure(state='normal')
    chat_history.insert(tk.END, "You: " + phone + "\n")
    chat_history.configure(state='disabled')
    
    insertUser = "INSERT INTO users (name, phone, created_at, updated_at) VALUES (%s, %s, NOW(), NOW())"
    userValues = (name, phone)
    databaseCursor.execute(insertUser, userValues)
    database.commit()
    chat_history.configure(state='normal')
    chat_history.insert(tk.END, "Bot : User data has been added to the system\n")
    chat_history.configure(state='disabled')

# Function to predict the class for a given sentence
def predict_class(sentence):
    # Tokenize and stem the words in the sentence
    sentence_words = [stemmer.stem(w.lower()) for w in nltk.word_tokenize(sentence)]
    # Create a bag of words representation of the sentence
    bag = [1 if w in sentence_words else 0 for w in words]
    # Use the model to predict the class for the sentence
    res = model.predict(np.array([bag]))[0]
    # Return the class with the highest probability
    return classes[np.argmax(res)]

# Function to get a response for a given sentence
def get_response(sentence):
    # Predict the class for the sentence
    tag = predict_class(sentence)
    # Search for the intent with the predicted class
    for intent in data['intents']:
        if intent['tag'] == tag:
            # Generate a random response from the list of responses
            response = random.choice(intent['responses'])
            # If the intent is to make a reservation, save the reservation to the database
            if intent['tag'] == 'make_reservation':
                # Get the user's name and phone number
                selectUser = "SELECT * FROM users ORDER BY id DESC LIMIT 1"
                databaseCursor.execute(selectUser)
                user = databaseCursor.fetchone()
                name = user['name']
                phone = user['phone']
                # Get the date and time for the reservation
                date = intent['patterns'][0].split()[2]
                time = intent['patterns'][0].split()[4]
                # Save the reservation to the database
                insertReservation = "INSERT INTO reservations (user_id, date, time, created_at, updated_at) VALUES (%s, %s, %s, NOW(), NOW())"
                reservationValues = (user['id'], date, time)
                databaseCursor.execute(insertReservation, reservationValues)
                database.commit()
                # Send a confirmation message to the user
                url = "https://api.twilio.com/2010-04-01/Accounts/ACXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX/Messages.json"
                payload = {'From': '+1415XXXXXXX', 'To': phone, 'Body': 'Hi ' + name + ', your reservation on ' + date + ' at ' + time + ' has been confirmed. Thanks for choosing our gym!'}
                requests.post(url, auth=('ACXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX', 'your_auth_token'), data=payload)
            # Return the response
            return response

# Function to handle user input
def handle_input():
    # Get the user's input
    user_text = user_input.get()
    # Clear the input box
    user_input.delete(0, tk.END)
    # Add the user's input to the chat history
    chat_history.configure(state='normal')
    chat_history.insert(tk.END, "You: " + user_text + "\n")
    chat_history.configure(state='disabled')
    # Get a response to the user's input
    bot_text = get_response(user_text)
    # Add the bot's response to the chat history
    chat_history.configure(state='normal')
    chat_history.insert(tk.END, "Bot : " + bot_text + "\n")
    chat_history.configure(state='disabled')
    
    if bot_text == "Please Enter your name :":
        userData()
    
# Create the send button
send_button = tk.Button(window, text="Send", command=handle_input)
send_button.pack(padx=10, pady=10)

# Start the main loop
window.mainloop()