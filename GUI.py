# pip3 install nltk
# pip install mysql.connector
# pip3 install tensorflow
#pip3 install tkinter
# nltk.download() ## download All

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

with open('intents.json') as file:
    data = json.load(file)

# Create the main window
window = tk.Tk()
window.title("Chatbot")

# Create the chat history text area
chat_history = scrolledtext.ScrolledText(window, width=60, height=40)
chat_history.configure(state='disabled')
chat_history.pack(padx=10, pady=10)

# Create the user input text box
user_input = tk.Entry(window, width=80)
user_input.pack(padx=10, pady=10)

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
database_cursor = database.cursor(dictionary=True)

chat_end = []
diet = []
booking = {}

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

    chat_end.clear()
    # Return the meal plans
    return "Here are some meal plans that might work for you:\n{}".format(meal_plans)

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

def handle_diet_dietary(message):
    diet.append(message)
    chat_end.clear()
    chat_end.append('goals')
    return "What are your fitness goals?"

def handle_diet_goals(message):
    diet.append(message)
    return determine_diet(diet[0], diet[1])

# Function to check if the value is number or not
def is_number(value):
    try:
        float(value)
        return True
    except ValueError:
        return False

# Function to return all type from database  
def handle_book_choose_type():
    # return all classes types to user to choose which class he need
    types_query = "SELECT * FROM types"
    database_cursor.execute(types_query)
    types_result = database_cursor.fetchall()
    chat_end.clear()
    chat_end.append('chosen_type')
    result = ""
    for type in types_result:
        result += str(type['id'])+ ": "+type['name'] + "\n"
    return "Choose which type you need (answer by number) :- \n " + result

# Function to return all classes depends on the chosen type
def handle_book_choose_class(chosen_type):
    if(is_number(chosen_type)):
        booking['type'] = chosen_type
        #Execute a SELECT query available classes in this type
        classes = "SELECT * FROM classes WHERE type_id = %s"
        val = (int(chosen_type), )
        database_cursor.execute(classes, val)
        classes_result = database_cursor.fetchall()
        chat_end.clear()
        chat_end.append('chosen_class')
        result = ""
        for oneClass in classes_result:
            result += str(oneClass['id'])+": "+ oneClass['name']+"=> Start Date: "+ str(oneClass['date']) + "\n"
        return "Choose which class you need (answer by number) :- \n " + result
    else:
        return "Please Type a valid type number"
    
#Function to check the max capacity in the class and take the agreement from the user
def handle_book_check_capacity(chosen_class):
    if(is_number(chosen_class)):
        booking['class'] = chosen_class
        # Execute a SELECT query to check if the class is available
        available_class = "SELECT * FROM classes WHERE id = %s"
        val = (int(chosen_class), )
        database_cursor.execute(available_class, val)
        class_result = database_cursor.fetchone()

        if class_result['max_capacity'] == 0:
            return "Sorry, that class is already full. Please choose a different class type."
        else:
            chat_end.clear()
            chat_end.append('agreement')
            return "There are empty places. If you want to confirm the reservation, Type Yes"

# Function to get user name
def handle_book_user_name(agreement):
    agreement = agreement.lower()
    if(agreement in ['yes', 'no']):
        if(agreement == 'yes'):
            chat_end.clear()
            chat_end.append('chosen_userName')
            return "Please Enter your name :"
        else:
            chat_end.clear()
            return "Thank you with spend time with us!"
    else:
        return "Please enter valid value ( yes or no )"

# Function to get user phone number
def handle_book_user_phone(name):
    booking['name'] = name
    chat_end.clear()
    chat_end.append('chosen_userPhone')
    return "Please Enter your phone :"

# Function to insert user date and book the class
def handle_book_user_data(phone):
    # return "user data"
    insert_user = "INSERT INTO users (name, phone, created_at, updated_at) VALUES (%s, %s, NOW(), NOW())"
    val = (booking['name'], phone, )
    database_cursor.execute(insert_user, val)
    database.commit()
    user_id = database_cursor.lastrowid

    # Execute a SELECT query to check if the class is available
    available_class = "SELECT * FROM classes WHERE id = %s"
    val = (int(booking['class']), )
    database_cursor.execute(available_class, val)
    class_result = database_cursor.fetchone()

    # Otherwise, decrement the class's capacity and insert a new booking record
    update_capacity = "UPDATE classes SET max_capacity = max_capacity - 1 WHERE id = %s"
    update_capacity_values = (int(booking['class']), )
    database_cursor.execute(update_capacity, update_capacity_values)

    create_booking = "INSERT INTO bookings (class_id, user_id, created_at, updated_at) VALUES (%s, %s, NOW(), NOW())"
    create_booking_values = (int(booking['class']), int(user_id), )
    database_cursor.execute(create_booking, create_booking_values)

    database.commit()
    chat_end.clear()
    # Return a confirmation message
    book_class_intent = next((intent for intent in data['intents'] if intent['tag'] == 'book_class'), None)
    return random.choice(book_class_intent['responses']).format(class_type=class_result['name'], date=class_result['date'])

# Function to predict the class for a given sentence
def predict_class(sentence):
    # Tokenize and stem the message
    message_words = nltk.word_tokenize(sentence)
    message_words = [stemmer.stem(w.lower()) for w in message_words]
    
    # Create a bag of words representation of the message
    bag = [1 if w in message_words else 0 for w in words]
    
    # Use the model to predict the class of the message
    prediction = model.predict(np.array([bag]))[0]
    max_index = np.argmax(prediction)
    return classes[max_index]

# Function to return random response from tag
def random_responses(tag):
    responses = next((intent for intent in data['intents'] if intent['tag'] == tag), None)
    return random.choice(responses['responses'])

# Function to generate the bot response
def generate_response(message):
    class_label = predict_class(message)

    # Choose a random response from the appropriate class
    for intent in data['intents']:
        if intent['tag'] == class_label:
            if intent['tag'] == 'greeting':
                return random_responses('greeting')
            elif intent['tag'] == 'goodbye':
                return random_responses('goodbye')
            elif intent['tag'] == 'book_class':
                return handle_book_choose_type()
            elif intent['tag'] == 'determine_diet':
                chat_end.append('dietary')
                return "What is your dietary restrictions?"
            elif intent['tag'] == 'faq':
                return get_faq_answer(message)
            elif intent['tag'] == 'thanks':
                return random_responses('thanks')
            else:
                faq_intent = next((intent for intent in data['intents'] if intent['tag'] == 'faq'), None)
                # If no match is found, display a message to the user
                unrecognized_message = "I'm sorry, I didn't understand your question. Here are some common questions you can ask:\n"
                unrecognized_message += "\n".join(faq_intent['patterns'])
                return unrecognized_message

# Function to handle bot response with the GUI
def handle_input():
    # Get the user's input
    message = user_input.get()
    if message.strip():
        # Generate a response from the chatbot
        if('dietary' in chat_end):
            response = handle_diet_dietary(message)
        elif('goals' in chat_end):
            response = handle_diet_goals(message)
        elif('chosen_type' in chat_end):
            response = handle_book_choose_class(message)
        elif('chosen_class' in chat_end):
            response = handle_book_check_capacity(message)
        elif('agreement' in chat_end):
            response = handle_book_user_name(message)
        elif('chosen_userName' in chat_end):
            response = handle_book_user_phone(message)
        elif('chosen_userPhone' in chat_end):
            response = handle_book_user_data(message)
        else:
            response = generate_response(message)

        # Display the response in the chat history
        chat_history.configure(state='normal')
        chat_history.insert(tk.END, "You: " + message + "\n")
        chat_history.insert(tk.END, "Bot: " + str(response) + "\n")
        chat_history.configure(state='disabled')

        # Clear the user's input
        user_input.delete(0, tk.END)

# Create the send button
send_button = tk.Button(window, text="Send", command=handle_input)
send_button.pack(padx=10, pady=10)

# Start the main event loop
window.mainloop()