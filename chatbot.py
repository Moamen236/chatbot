import nltk
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
    print("Bot : Please Enter your name :")
    name = input("You: ")
    print("Bot : Please Enter your phone :")
    phone = input("You: ")
    
    insertUser = "INSERT INTO users (name, phone, created_at, updated_at) VALUES (%s, %s, NOW(), NOW())"
    val = (name, phone, )
    databaseCursor.execute(insertUser, val)
    database.commit()
    return databaseCursor.lastrowid

#Function to book a class
def book_class():
    # return all classes types to user to choose which class he need
    typesQuery = "SELECT * FROM types"
    databaseCursor.execute(typesQuery)
    typesResult = databaseCursor.fetchall()
    print("Bot : Choose which type you need (answer by number) :-")
    for type in typesResult:
        print(type['id'], ": ",type['name'])
    chosenType = input("You: ")

    #Execute a SELECT query available classes in this type
    classes = "SELECT * FROM classes WHERE type_id = %s"
    val = (int(chosenType), )
    databaseCursor.execute(classes, val)
    classesResult = databaseCursor.fetchall()
    print("Bot : Choose which class you need (answer by number) :-")
    for oneClass in classesResult:
        print(oneClass['id'], ": ", oneClass['name'], "=> Start Date: ", oneClass['date'])
    chosenClass = input("You: ")

    # Execute a SELECT query to check if the class is available
    availableClass = "SELECT * FROM classes WHERE id = %s"
    val = (int(chosenClass), )
    databaseCursor.execute(availableClass, val)
    classResult = databaseCursor.fetchone()
    
    #If the class is full, return a message saying it's unavailable
    if classResult['max_capacity'] == 0:
        return "Sorry, that class is already full. Please choose a different time or class type."
    else:
        print("Bot : There are empty places. If you want to confirm the reservation, Type Yes")
        userChoice = input("You: ")
        if(userChoice.lower() == 'yes'):
            # Create new user and get the id
            user_id = userData()
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
        else:
            return "Thank you for dealing with us"

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

# def get_faq_answer(question):
#     # Retrieve the answer to the user's question from intents.json file from responses array instead of user input by nltk.word_tokenize of the patterns
#     # Return the answer as a string
#     return "Here's the answer to your question: ..."
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

def generate_response(model, words, classes, message):
    # Tokenize and stem the message
    message_words = nltk.word_tokenize(message)
    message_words = [stemmer.stem(w.lower()) for w in message_words]
    
    # Create a bag of words representation of the message
    bag = [1 if w in message_words else 0 for w in words]
    
    # Use the model to predict the class of the message
    prediction = model.predict(np.array([bag]))[0]
    max_index = np.argmax(prediction)
    class_label = classes[max_index]
    # Choose a random response from the appropriate class
    for intent in data['intents']:
        if intent['tag'] == class_label:
            if intent['tag'] == 'book_class':
                # class_type = intent['parameters'][1]
                # date = intent['parameters']['date']
                # time = intent['parameters']['time']
                return book_class()
            elif intent['tag'] == 'determine_diet':
                # dietary_prompts = [param["prompts"] for param in intent['parameters'] if param["name"] == "dietary_restrictions"][0]
                print("Bot:", "Do you have any dietary restrictions?")
                dietary_restrictions = input("You: ")

                # goals_prompts = [param["prompts"] for param in intent['parameters'] if param["name"] == "goals"][0]
                print("Bot:", "What are your fitness goals?")
                goals = input("You: ")
                return determine_diet(dietary_restrictions, goals)
            elif intent['tag'] == 'faq':
                return get_faq_answer(message)
            else:
                responses = intent['responses']
                return random.choice(responses)

print("Chatbot is running!")

while True:
    message = input("You: ")
    response = generate_response(model, words, classes, message)
    print("Bot:", response)