import numpy as np
import pandas as pd
from sklearn.metrics import mutual_info_score
import torch
from transformers import GPT2Model, GPT2Tokenizer
from transformers import AutoTokenizer, AutoModelForCausalLM, BloomForCausalLM, LlamaForCausalLM
from typing import Union, List, Dict, Tuple, Optional, Any
import pickle
import torch.optim as optim
from tqdm import tqdm
import os
from multidomain_dataset import MultiDomainDataset
from torch.utils.data import Dataset, Sampler, DataLoader, DistributedSampler
from torch.utils.tensorboard import SummaryWriter

MODEL_NAME = "llm/Qwen2.5-7B-Instruct"

global_devices = [i for i in range(torch.cuda.device_count())] if torch.cuda.device_count() >= 1 else ["cpu"]
max_memory = {k: '40GB' for k in global_devices}

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, device_map='balanced', torch_dtype=torch.bfloat16, max_memory=max_memory)

tokenizer.pad_token = tokenizer.eos_token

src = "de"
tgt = "en"

mi_scores_file = "neurons/mi_scores_multi_domain_qwen2_5.pkl"

with open(mi_scores_file, 'rb') as f:
    data = pickle.load(f)
    importance_matrix_dict = data["mutual_info_dict"]


all_values = []
all_keys = []

for key, value in importance_matrix_dict.items():
    if "mlp" in key:
        all_values.extend(value)
        all_keys.extend([(key, i) for i in range(len(value))])

total_neurons = len(all_values)
num_selected = int(total_neurons * 0.01)
sorted_indices = np.argsort(all_values)[::-1][:num_selected]

layer_indices = {}
for idx in sorted_indices:
    layer, neuron_idx = all_keys[idx]
    if layer not in layer_indices:
        layer_indices[layer] = []
    layer_indices[layer].append(neuron_idx)


optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-5)

train_dataset = MultiDomainDataset(
    tokenizer=tokenizer, 
    src=src, 
    tgt=tgt, 
    split="train"
)
valid_dataset = MultiDomainDataset(
    tokenizer=tokenizer,
    src=src,
    tgt=tgt,
    split="dev"
)

train_dataloader = DataLoader(train_dataset, batch_size=1)
valid_dataloader = DataLoader(valid_dataset, batch_size=1)

update_frequency = 5
num_epochs = 2
update_counter = 0
path_to_save_model = "saves"


for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    dataloader_len = len(train_dataloader)
    pbar = tqdm(total=dataloader_len, desc=f'train_{epoch}')
    writer = SummaryWriter(os.path.join(path_to_save_model, "model_all_domain_mlp_llama2_logs", "logs"))
    for batch_idx, batch in enumerate(train_dataloader):
        input_ids = batch['input_ids'].to("cuda")
        attention_mask = batch['attention_mask'].to("cuda")
        labels = batch['labels'].to("cuda")

        outputs = model(input_ids=input_ids[0], attention_mask=attention_mask[0], labels=labels[0])
        loss = outputs.loss

        loss.backward()
        
        if (batch_idx + 1) % update_frequency == 0:
            for name, param in model.named_parameters():
                module_name = ('.').join(name.split('.')[:-1])
                if module_name in layer_indices:
                    important_indices = layer_indices[module_name]
                    mask = torch.zeros_like(param)
                    for idx in important_indices:
                        mask[idx] = 1
                    if 'weight' in name:
                        param.grad *= mask

                else:
                    param.requires_grad = False

            optimizer.step()
            optimizer.zero_grad()
            total_loss += loss.item()

            update_counter += 1

            print(f"Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx+1}/{dataloader_len}], Update {update_counter}, Loss: {loss.item(): 4f}")
            writer.add_scalar("loss", loss.item(), update_counter)

        pbar.update(1)

    if (batch_idx + 1) % update_frequency != 0:
        for name, param in model.named_parameters():
            module_name = ('.').join(name.split('.')[:-1])
            if module_name in layer_indices:
                important_indices = layer_indices[module_name]
                mask = torch.zeros_like(param)
                for idx in important_indices:
                    mask[idx] = 1
                if 'weight' in name:
                    param.grad *= mask

            else:
                param.requires_grad = False

        optimizer.step()
        optimizer.zero_grad()
        total_loss += loss.item()
        avg_train_loss = total_loss / dataloader_len
        print(f"Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx+1}/{dataloader_len}], Update {update_counter}, Loss: {avg_train_loss: 4f}")
        writer.add_scalar('loss', loss.item(), update_counter + 1)

    
    pbar.close()
    writer.close()
    avg_train_loss = total_loss / dataloader_len
    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {avg_train_loss: 4f}")

    model.eval()
    val_loss = 0
    with torch.no_grad():
        for batch_idx, batch in enumerate(valid_dataloader):
            input_ids = batch['input_ids'].to("cuda")
            attention_mask = batch['attention_mask'].to("cuda")
            labels = batch['labels'].to("cuda")

            outputs = model(input_ids=input_ids[0], attention_mask=attention_mask[0], labels=labels[0])
            val_loss += outputs.loss.item()
        
    avg_val_loss = val_loss / len(valid_dataloader)

    print(f"Epoch {epoch+1}/{num_epochs}")
    print(f"Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

    model.save_pretrained(os.path.join(path_to_save_model, f"model_all_domain_mlp_{epoch}"))
    tokenizer.save_pretrained(os.path.join(path_to_save_model, f"model_all_domain_mlp_{epoch}"))
    
