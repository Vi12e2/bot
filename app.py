from flask import Flask, render_template, request, jsonify
import logging
import torch
from chatbot_model import ChatbotModel

app = Flask(__name__)

chatbot = ChatbotModel()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/get', methods=['POST'])
def chatbot_response():
    user_text = request.form['msg']
    
    # Создание нового контекста для каждого запроса
    context = [chatbot.initial_context]

    # Генерация ответа
    response, _ = chatbot.generate_response(user_text, context)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return jsonify({"response": response})

if __name__ == "__main__":
    app.run(host="localhost", port=8080, debug=True)

logging.basicConfig(level=logging.INFO)
