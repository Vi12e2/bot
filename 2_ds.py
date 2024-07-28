from bs4 import BeautifulSoup 
from tqdm.auto import tqdm
from transformers.integrations import TensorBoardCallback
from datasets import load_dataset, concatenate_datasets
from transformers import GPT2LMHeadModel, GPT2Tokenizer, Trainer, TrainingArguments, pipeline
import sys
import logging
import os


# Отключение буферизации для stdout и stderr
sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', buffering=1)
sys.stderr = os.fdopen(sys.stderr.fileno(), 'w', buffering=1)

# Настройка логирования
output_dir = '/home/u12/working/c21'
log_file = os.path.join(output_dir, 'training.log')
eval_log_file = os.path.join(output_dir, 'evaluation.log')

# Проверка директории для сохранения
os.makedirs(output_dir, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler(sys.stdout)  
    ]
)
logger = logging.getLogger(__name__)

# Настройка логгера для оценки
eval_logger = logging.getLogger('evaluation_logger')
eval_logger.setLevel(logging.INFO)
eval_logger.addHandler(logging.FileHandler(eval_log_file))
eval_logger.addHandler(logging.StreamHandler(sys.stdout))

# Логирование для Transformers
transformers_logger = logging.getLogger('transformers')
transformers_logger.setLevel(logging.INFO)
transformers_logger.addHandler(logging.FileHandler(log_file))
transformers_logger.addHandler(logging.StreamHandler(sys.stdout))

trust_remote_code = True

# Загрузка датасетов
dataset_habr = load_dataset('IlyaGusev/habr', split='train[85%:100%]', trust_remote_code=True)
dataset_sberquad = load_dataset('kuznetsoffandrey/sberquad', split='train[:100%]', trust_remote_code=True)

# Разделение датасетов на тренировочные и валидационные наборы
train_test_split_habr = dataset_habr.train_test_split(test_size=0.1)
train_dataset_habr = train_test_split_habr['train']
eval_dataset_habr = train_test_split_habr['test']

train_test_split_sberquad = dataset_sberquad.train_test_split(test_size=0.1)
train_dataset_sberquad = train_test_split_sberquad['train']
eval_dataset_sberquad = train_test_split_sberquad['test']

# Инициализация токенизатора и модели
model_name_or_path = '/home/u12/working/checkpoint-4305'
tokenizer = GPT2Tokenizer.from_pretrained(model_name_or_path)
model = GPT2LMHeadModel.from_pretrained(model_name_or_path)

# Добавление токена [PAD]
tokenizer.pad_token = tokenizer.eos_token


def clean_html_and_remove_links(text, replace_with="[LINK_REMOVED]"):
    # Использование BeautifulSoup для очистки HTML и замены ссылок
    soup = BeautifulSoup(text, 'html.parser')
    for a_tag in soup.find_all('a'):
        a_tag.replace_with(replace_with)

    return soup.get_text()


def flatten_comments(examples):
    flattened_comments = []
    for comments in examples['comments']:
        if 'message_html' in comments:
            # Очистка HTML и замена ссылок меткой
            cleaned_comments = [clean_html_and_remove_links(comment) for comment in comments['message_html']]
            # Удаление фразы
            cleaned_comments = [comment.replace("UFO just landed and posted this here", "").replace("НЛО прилетело и опубликовало эту надпись здесь.", "").replace("Спасибо за статью", "").replace("спасибо за статью", "").replace("Хабр", "[этот сайт]").replace("Хабрахабр", "[этот сайт]").replace("хабрахабр", "[этот сайт]").replace("хабр", "[этот сайт]").replace("ХабраХабр", "[этот сайт]").replace("ХабрХабр", "[этот сайт]").replace("хабрхабр", "[этот сайт]").replace("хабрахабре", "[этом сайте]").replace("Хабрахабре", "[этом сайте]").replace("Хабре", "[этом сайте]").replace("хабре", "[этом сайте]").replace("Хабра", "[этого сайта]").replace("Хабрахабра", "[этого сайта]").replace("хабрахабра", "[этого сайта]").replace("хабра", "[этого сайта]").replace("Хабрхабра", "[этого сайта]").replace("хабрхабра", "[этого сайта]") for comment in cleaned_comments]
            # Добавление маркеров начала и конца комментария
            marked_comments = [f"[ANSWER_START] {comment} [ANSWER_END]" for comment in cleaned_comments]
            flattened_comments.append(" ".join(marked_comments))
        else:
            flattened_comments.append("")
    examples['comments'] = flattened_comments
    return examples


def tokenize_function_habr(examples):
    combined_texts = []
    for title, comments in zip(examples['title'], examples['comments']):
        combined_texts.append(f"[CONTEXT] {title} [QUESTION] [ANSWER] {comments}")

    # for text_markdown, title, comments in zip(examples['text_markdown'], examples['title'], examples['comments']):
       
        # combined_texts.append(f"[CONTEXT] {title} [QUESTION] {title} [ANSWER] {text_markdown}")
        
        # combined_texts.append(f"[CONTEXT] {title} [QUESTION] [ANSWER] {text_markdown}\n[COMMENTS] {comments}")

    tokenized_inputs = tokenizer(combined_texts, padding='max_length', truncation=True, max_length=162)
    tokenized_inputs["labels"] = tokenized_inputs["input_ids"].copy()
    return tokenized_inputs


def tokenize_function_sberquad(examples):
    combined_texts = []
    for context, question, answers in zip(examples['context'], examples['question'], examples['answers']):
        if isinstance(answers, dict) and 'text' in answers:
            answer_text = " ".join(answers['text'])
        elif isinstance(answers, list):
            answer_text = " ".join([ans['text'] for ans in answers])
        else:
            answer_text = answers
        combined_texts.append(f"[CONTEXT] {context} [QUESTION] {question} [ANSWER] {answer_text}")

    tokenized_inputs = tokenizer(combined_texts, padding='max_length', truncation=True, max_length=162)
    tokenized_inputs["labels"] = tokenized_inputs["input_ids"].copy()
    return tokenized_inputs


train_dataset_habr = train_dataset_habr.map(flatten_comments, batched=True)
eval_dataset_habr = eval_dataset_habr.map(flatten_comments, batched=True)

# Токенизация датасетов
logger.info("Tokenizing datasets...")
tokenized_datasets_habr = train_dataset_habr.map(tokenize_function_habr, batched=True, remove_columns=["id", "language", "url", "text_markdown", "text_html", "lead_markdown", "lead_html", "type", "labels", "original_author", "original_url", "time_published", "author", "title", "statistics", "hubs", "flows", "tags", "reading_time", "format", "complexity", "comments"])
tokenized_eval_dataset_habr = eval_dataset_habr.map(tokenize_function_habr, batched=True, remove_columns=["id", "language", "url", "text_markdown", "text_html", "lead_markdown", "lead_html", "type", "labels", "original_author", "original_url", "time_published", "author", "title", "statistics", "hubs", "flows", "tags", "reading_time", "format", "complexity", "comments"])

tokenized_datasets_sberquad = dataset_sberquad.map(tokenize_function_sberquad, batched=True, remove_columns=["id", "title", "context", "question", "answers"])
tokenized_eval_dataset_sberquad = eval_dataset_sberquad.map(tokenize_function_sberquad, batched=True, remove_columns=["id", "title", "context", "question", "answers"])

# Объединение тренировочных и валидационных датасетов
train_dataset = concatenate_datasets([tokenized_datasets_sberquad, tokenized_datasets_habr])
eval_dataset = concatenate_datasets([tokenized_eval_dataset_sberquad, tokenized_eval_dataset_habr])

training_args = TrainingArguments(
    output_dir=output_dir,
    learning_rate=2e-5,
    per_device_train_batch_size=12,
    per_device_eval_batch_size=12,
    num_train_epochs=3,
    weight_decay=0.03,
    save_steps=1200,           
    eval_strategy="steps",
    eval_steps=600,
    logging_steps=25,
    disable_tqdm=False,  
    report_to="all",
    gradient_accumulation_steps=8,  # Количество шагов для накопления градиентов
    # dataloader_num_workers=5,     # Число потоков загрузки данных
    # fp16=True
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    callbacks=[TensorBoardCallback()],
)

# Тренировка модели
trainer.train()

# Сохранение модели и токенизатора
trainer.save_model(output_dir)
tokenizer.save_pretrained(output_dir)

trainer.save_state()

