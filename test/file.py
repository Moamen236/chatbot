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