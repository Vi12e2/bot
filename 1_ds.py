import logging
import os
import sys
from transformers import GPT2LMHeadModel, GPT2Tokenizer, GPT2Config, Trainer, TrainingArguments, pipeline
from datasets import load_dataset
from transformers.integrations import TensorBoardCallback
from tqdm.auto import tqdm

# Отключение буферизации
sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', buffering=1)
sys.stderr = os.fdopen(sys.stderr.fileno(), 'w', buffering=1)


output_dir = '/home/u12/working/c5'
log_file = os.path.join(output_dir, 'training.log')
eval_log_file = os.path.join(output_dir, 'evaluation.log')
os.makedirs(output_dir, exist_ok=True)

# Настройки логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


eval_logger = logging.getLogger('evaluation_logger')
eval_logger.setLevel(logging.INFO)
eval_logger.addHandler(logging.FileHandler(eval_log_file))
eval_logger.addHandler(logging.StreamHandler(sys.stdout))


transformers_logger = logging.getLogger('transformers')
transformers_logger.setLevel(logging.INFO)
transformers_logger.addHandler(logging.FileHandler(log_file))
transformers_logger.addHandler(logging.StreamHandler(sys.stdout))

# Загрузка датасета
logger.info("Loading dataset...")
dataset = load_dataset('kuznetsoffandrey/sberquad', split='train[:100%]', trust_remote_code=True)

# Разделение датасета
logger.info("Splitting dataset...")
train_test_split = dataset.train_test_split(test_size=0.1)
train_dataset = train_test_split['train']
eval_dataset = train_test_split['test']

# Инициализация токенайзера и модели
model_name_or_path = '/home/u12/working/c5/checkpoint-11335'
tokenizer = GPT2Tokenizer.from_pretrained(model_name_or_path)
config = GPT2Config.from_pretrained(model_name_or_path)

# Dropout настройки
config.attn_pdrop = 0.1      
config.resid_pdrop = 0.1    
config.embd_pdrop = 0.1     

model = GPT2LMHeadModel.from_pretrained(model_name_or_path, config=config)

# Добавление PAD токена
tokenizer.pad_token = tokenizer.eos_token

def tokenize_function(examples):
    combined_texts = []
    for context, question, answers in zip(examples['context'], examples['question'], examples['answers']):
        if isinstance(answers, dict) and 'text' in answers:
            answer_text = " ".join(answers['text'])
        elif isinstance(answers, list):
            answer_text = " ".join([ans['text'] for ans in answers])
        else:
            answer_text = answers
        combined_texts.append(f"[CONTEXT] {context} [QUESTION] {question} [ANSWER] {answer_text}")

    tokenized_inputs = tokenizer(combined_texts, padding='max_length', truncation=True, max_length=177)
    tokenized_inputs["labels"] = tokenized_inputs["input_ids"].copy()
    return tokenized_inputs

# Токенизация датасета
tokenized_datasets = dataset.map(tokenize_function, batched=True, remove_columns=["id", "title", "context", "question", "answers"])
tokenized_eval_dataset = eval_dataset.map(tokenize_function, batched=True, remove_columns=["id", "title", "context", "question", "answers"])

# Подготовка данных для тренировки
train_dataset = tokenized_datasets
eval_dataset = tokenized_eval_dataset

training_args = TrainingArguments(
    output_dir=output_dir,
    learning_rate=1e-5,
    per_device_train_batch_size=12,
    per_device_eval_batch_size=12,
    num_train_epochs=5,
    weight_decay=0.01,            
    save_steps=2000,              
    eval_strategy="steps",
    eval_steps=1000,
    logging_steps=250,
    disable_tqdm=False,  
    report_to="all",
    # gradient_accumulation_steps=2,  
    # dataloader_num_workers=2,  
    # fp16=True
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    callbacks=[TensorBoardCallback()],  
)

trainer.train()

# Сохранение модели и токенизатора после тренировки
trainer.save_model(output_dir)
tokenizer.save_pretrained(output_dir)

# Сохранение состояния оптимизатора и планировщика
trainer.save_state()

