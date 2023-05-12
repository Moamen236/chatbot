import tkinter as tk
from tkinter import scrolledtext
from chatbot import generate_response
from chatbot import model
from chatbot import words
from chatbot import classes

# Define the chatbot model, words, and classes
# This assumes that you've already loaded and trained the model, and defined the words and classes
# model = model
# words = ...
# classes = ...

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

# Define the function to handle user input
def handle_input():
    # Get the user's input
    message = user_input.get()

    # Generate a response from the chatbot
    response = generate_response(model, words, classes, message)

    # Display the response in the chat history
    chat_history.configure(state='normal')
    chat_history.insert(tk.END, "You: " + message + "\n")
    chat_history.insert(tk.END, "Bot: " + response + "\n")
    chat_history.configure(state='disabled')

    # Clear the user's input
    user_input.delete(0, tk.END)

# Create the send button
send_button = tk.Button(window, text="Send", command=handle_input)
send_button.pack(padx=10, pady=10)

# Start the main event loop
window.mainloop()