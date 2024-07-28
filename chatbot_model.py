import logging
import re
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer, StoppingCriteria, StoppingCriteriaList

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ChatbotModel:
    def __init__(self, model_path=r'C:\pyt\ \c5\checkpoint-6000'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_path)
        self.model = GPT2LMHeadModel.from_pretrained(model_path).to(self.device)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.initial_context = "Привет, я - бот, вы можете задавать мне любые вопросы"  # Начальный контекст
        self.is_first_response = True   # Флаг на первый ответ

    def generate_response(self, user_input, context):
        if self.is_first_response:
            context.append(f"[USER] {user_input}")
            input_text = f"[CONTEXT] {' '.join(context)} [QUESTION] {user_input} [ANSWER]"
        else:
            context.append(f"[USER] {user_input}")
            # input_text = f"[CONTEXT][QUESTION] {user_input} [ANSWER]"
            input_text = f"[QUESTION] {user_input} [ANSWER]"

        inputs = self.tokenizer.encode(input_text, return_tensors='pt').to(self.device)
        if self.is_first_response:
            max_length = inputs.shape[1] + 50   # Ограничение первого ответа
        else:
            max_length = inputs.shape[1] + 120  # Ограничение последующих ответов

        with torch.no_grad():
            outputs = self.model.generate(
                inputs,
                max_length=max_length,           
                min_length=inputs.shape[1] + 20,  
                num_return_sequences=1,
                pad_token_id=self.tokenizer.eos_token_id,
                do_sample=True,
                top_p=0.95,
                top_k=30,
                repetition_penalty=2.0,
                no_repeat_ngram_size=2,  # Предотвращение повторения биграмм
                temperature=0.7,
                num_beams=2,
            )
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Очистка ответа
        response = self._clean_response(user_input, response)

        # Обновление контекста с ответом бота
        context.append(f"[BOT] {response}")

        # Сброс флага после первого ответа
        self.is_first_response = False

        return response, context

    def _clean_response(self, user_input, response):
        response = re.sub(r'\[.*?\]', '', response).strip()  
        if not user_input in self.initial_context and user_input in response:
            response = response.replace(user_input, '').strip()
        response = self._remove_invalid_characters(response)
        
        return response
    
    def _remove_invalid_characters(self, text):
        return text.replace('�', '')

# Пример использования ChatbotModel
if __name__ == "__main__":
    chatbot = ChatbotModel()

