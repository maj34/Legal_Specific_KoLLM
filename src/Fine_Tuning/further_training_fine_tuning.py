from tqdm.auto import tqdm
from sklearn.model_selection import train_test_split
from datasets import Dataset
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    DataCollatorForLanguageModeling, TrainingArguments, BitsAndBytesConfig)
from trl import SFTTrainer
from peft import PeftModel, prepare_model_for_kbit_training, LoraConfig, get_peft_model
from sklearn.model_selection import train_test_split
from omegaconf import OmegaConf

import pandas as pd
import wandb
import torch
import os
import json
import re
import argparse

def run(cfg):
    # import qa dataset
    with open(cfg.dataset.path, 'r', encoding="utf-8") as file:
        train_dataset = json.load(file)

    # wandb 초기화
    # wandb api key 입력
    wandb.login(key='your api key')
    project_name = cfg.model.split('/')[1] + '-' + 'further_training' + '-' 'fine_Tuning'
    wandb.init(project=project_name, config=OmegaConf.to_container(cfg, resolve=True))

    checkpoint_path = "../checkpoints" + '/' + cfg.model.split('/')[1] + '/' + 'further_training' + '/' + 'fine_tuning'

    os.makedirs(checkpoint_path, exist_ok=True)

    dataset = pd.DataFrame()
    
    further_trained_checkpoint_path = '/workspace/Legal_Specific_KoLLM/src/LLM_Fine_Tuning/checkpoints/EEVE-Korean-10.8B-v1.0/further_training'

    tokenizer = AutoTokenizer.from_pretrained(further_trained_checkpoint_path)

    for key, value in tqdm(enumerate(train_dataset.items()), total=len(train_dataset)):
        answer = value[1]['answer']
        stop_sentences = ['질문하신 내용에 대하여 아래와 같이 답변드리오니 참고하시기 바랍니다.', '질문하신 내용에 대하여 아래와 같이 답변 드립니다.']
        for stop_sentence in stop_sentences:
                    if stop_sentence in answer:
                        clean_answer = answer.replace(stop_sentence, '')
                        break
                    else:
                        clean_answer = answer
        clean_text = clean_answer.strip()
        
        clean_text = clean_text.replace('\'', '')
        pattern = r'(감사|이성재|추천|혹시|이상).*?\.'
        clean_text = re.sub(pattern, '', clean_text, flags=re.DOTALL)
        
        message = [{"role": "system", "content": "You are a helpful legal chatbot."}, {"role": "user", "content": value[1]['question']}, {"role": "assistant", "content": clean_text}]
        input = tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=False, return_tensors="pt")
        
        dataset.loc[key, 'text'] = input

    # train, valid split
    train_dataset, valid_dataset = train_test_split(dataset, test_size=0.1, random_state=8421)

    train_dataset = Dataset.from_pandas(train_dataset)
    valid_dataset = Dataset.from_pandas(valid_dataset)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit = cfg.bnb_config.load_in_4bit,
        bnb_4bit_use_double_quant = cfg.bnb_config.bnb_4bit_use_double_quant,
        bnb_4bit_quant_type = cfg.bnb_config.bnb_4bit_quant_type,
        bnb_4bit_compute_dtype = torch.float16
    )

    # monkey_patch_packing_llama() # Monkey-patch LlamaForCausalLM
    model = AutoModelForCausalLM.from_pretrained(cfg.model, quantization_config=bnb_config)
    model = prepare_model_for_kbit_training(model)
    model.resize_token_embeddings(len(tokenizer))

    print(model.config)

    model.gradient_checkpointing_enable()
    model = prepare_model_for_kbit_training(model)

    model = PeftModel.from_pretrained(
        model=model,
        model_id =further_trained_checkpoint_path,
        peft_config=bnb_config
    )
    
    def print_trainable_parameters(model):
        """
        Prints the number of trainable parameters in the model.
        """
        trainable_params = 0
        all_param = 0
        for _, param in model.named_parameters():
            all_param += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()
        print(
            f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
        )

    config = LoraConfig(
        r = cfg.lora_config.r, 
        lora_alpha = cfg.lora_config.lora_alpha, 
        target_modules = list(cfg.lora_config.target_modules),
        lora_dropout = cfg.lora_config.lora_dropout, 
        bias = cfg.lora_config.bias, 
        task_type = cfg.lora_config.task_type
    )

    model = get_peft_model(model, config)
    model.resize_token_embeddings(len(tokenizer))

    print_trainable_parameters(model)

    training_args = TrainingArguments(
        output_dir = checkpoint_path,
        num_train_epochs = cfg.training_args.num_train_epochs,
        per_device_train_batch_size = cfg.training_args.per_device_train_batch_size,
        per_device_eval_batch_size = cfg.training_args.per_device_eval_batch_size,
        gradient_accumulation_steps = cfg.training_args.gradient_accumulation_steps,
        evaluation_strategy = cfg.training_args.evaluation_strategy,
        save_strategy = cfg.training_args.save_strategy,
        save_total_limit = cfg.training_args.save_total_limit,
        logging_steps = cfg.training_args.logging_steps,
        learning_rate = cfg.training_args.learning_rate,
        weight_decay = cfg.training_args.weight_decay,
        fp16 = cfg.training_args.fp16,
        seed = cfg.training_args.seed,
        load_best_model_at_end = cfg.training_args.load_best_model_at_end,
        report_to = cfg.training_args.report_to
    )

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        tokenizer=tokenizer,
        dataset_text_field="text"
    )

    trainer.train()

    results = trainer.evaluate()
    print(f"Perplexity: {torch.exp(torch.tensor(results['eval_loss']))}")

if __name__=='__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file', type=str, default='eeve.yaml')
    
    args = parser.parse_args()
    
    # load default config
    cfg = OmegaConf.load(args.config_file)
    
    run(cfg)