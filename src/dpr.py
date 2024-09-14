from transformers import AutoTokenizer, AutoModel, Trainer, TrainingArguments
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader
from transformers import DataCollatorWithPadding
from sklearn.metrics import accuracy_score
from transformers import EarlyStoppingCallback

import torch
import pandas as pd
import wandb

wandb.login(key='') # personal key
project_name = 'dpr_training'
wandb.init(project=project_name)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
print("Device cnt:", torch.cuda.device_count())

class QADataset(Dataset):
    def __init__(self, data_path, split='train'):
        self.data = pd.read_csv(data_path)
        self.data = self.data.sample(frac=1).reset_index(drop=True)
        if split == 'train':
            self.data = self.data[:-100]
        elif split == 'val':
            self.data = self.data[-100:]
        self.data.reset_index(drop=True, inplace=True)
        self.tokenizer = AutoTokenizer.from_pretrained('kakaobank/kf-deberta-base')
        self.questions = self.data['Question']
        self.passages = self.data['Passage']  # passage를 title로 대체
    
    def __len__(self):
        return len(self.questions)
    
    def __getitem__(self, idx):
        question_encodings = self.tokenizer(self.questions[idx], truncation=True, padding='max_length', max_length=512, return_tensors="pt")
        passage_encodings = self.tokenizer(self.passages[idx], truncation=True, padding='max_length', max_length=512, return_tensors="pt")
        return {
            "input_ids_q": question_encodings['input_ids'].squeeze(),  # Question input ids
            "attention_mask_q": question_encodings['attention_mask'].squeeze(),  # Question attention mask
            "input_ids_p": passage_encodings['input_ids'].squeeze(),  # Passage input ids
            "attention_mask_p": passage_encodings['attention_mask'].squeeze()  # Passage attention mask
        }

train_dataset = QADataset('/workspace/Legal_Specific_KoLLM/dataset/final_long_form_qa_dataset/train.csv', split='train')
val_dataset = QADataset('/workspace/Legal_Specific_KoLLM/dataset/final_long_form_qa_dataset/train.csv', split='val')

class DPRQuestionEncoder(torch.nn.Module):
    def __init__(self, model_name):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
    
    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids, attention_mask)
        return outputs.last_hidden_state[:, 0]  # [CLS] token

class DPRPassageEncoder(torch.nn.Module):
    def __init__(self, model_name):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
    
    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids, attention_mask)
        return outputs.last_hidden_state[:, 0]  # [CLS] token

class DPRModel(torch.nn.Module):
    def __init__(self, question_model_name, passage_model_name):
        super().__init__()
        self.question_encoder = DPRQuestionEncoder(question_model_name)
        self.passage_encoder = DPRPassageEncoder(passage_model_name)
    
    def forward(self, input_ids_q, attention_mask_q, input_ids_p, attention_mask_p, **kwargs):
        batch_size = input_ids_q.size(0)
        question_embeddings = self.question_encoder(input_ids_q, attention_mask_q)
        passage_embeddings = self.passage_encoder(input_ids_p, attention_mask_p)
        similarities = torch.matmul(question_embeddings, passage_embeddings.T)  # [batch_size, batch_size]

        labels = torch.arange(batch_size).to(similarities.device)
        loss_fct = torch.nn.CrossEntropyLoss()
        loss = loss_fct(similarities, labels)

        return {"loss": loss, "logits": similarities}
    
    def save_model(self, save_path):
        torch.save(self.state_dict(), save_path)

    @classmethod
    def load_model(cls, save_path, question_model_name, passage_model_name):
        model = cls(question_model_name, passage_model_name)
        model.load_state_dict(torch.load(save_path))
        return model

class CustomDataCollator:
    def __call__(self, features):
        input_ids_q = torch.stack([f['input_ids_q'] for f in features])
        attention_mask_q = torch.stack([f['attention_mask_q'] for f in features])
        input_ids_p = torch.stack([f['input_ids_p'] for f in features])
        attention_mask_p = torch.stack([f['attention_mask_p'] for f in features])
        
        labels = torch.arange(len(features))
        
        return {
            "input_ids_q": input_ids_q,
            "attention_mask_q": attention_mask_q,
            "input_ids_p": input_ids_p,
            "attention_mask_p": attention_mask_p,
            "labels": labels
        }

model = DPRModel('kakaobank/kf-deberta-base', 'kakaobank/kf-deberta-base').to(device)
tokenizer = AutoTokenizer.from_pretrained('kakaobank/kf-deberta-base')

data_collator = CustomDataCollator()

training_args = TrainingArguments(
    output_dir='./results',
    evaluation_strategy="steps",
    eval_steps=10,  # 매 50 스텝마다 평가
    save_strategy="steps",  # 에폭마다 체크포인트 저장
    num_train_epochs=10,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    logging_dir='./logs',
    logging_steps=10,
    load_best_model_at_end=True,  # 훈련이 끝난 후 최적의 모델을 로드
    metric_for_best_model='eval_accuracy',
    label_names=["labels"],
    learning_rate=1e-5,
    save_total_limit=5  # 저장할 체크포인트의 최대 개수
)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = logits.argmax(axis=1)
    accuracy = accuracy_score(labels, predictions) # 실제 정답 인덱스와 예측 인덱스 비교
    return {'eval_accuracy': accuracy} # 'eval_accuracy' 키로 정확도 반환

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3, early_stopping_threshold=0.001)]  # 여기에 콜백 추가
)

trainer.train()

save_path = './results/dpr_model.pth'
model.save_model(save_path)

print("Final model and configuration saved to './results/checkpoint-final'")