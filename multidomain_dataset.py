from torch.utils.data import Dataset, Sampler, DataLoader
from torch.utils.data.sampler import SequentialSampler, BatchSampler
from typing import Union, List, Dict, Tuple, Optional, Any
import random
import torch
import json
import os


class MultiDomainDataset(Dataset):
    def construct_translation_prompt(
        self,
        train_example: Dict[str, Union[str, int]],
        prompt_template: str,
        instruction: str = "",
    ) -> Tuple[str, str]:
        messages = []
        if instruction != "":
            instruction = instruction.format(**train_example)
            messages.append({"role": "system", "content": instruction})

        prompt_template = random.choice(prompt_template)
        prompt_input = (
            prompt_template[0].format(lang1= train_example["src"], lang2= train_example["tgt"], sent1= train_example["sent1"].strip("\n"), domain=train_example["domain"])
            )
        messages.append({"role": "user", "content": prompt_input})
        
        prompt_input = messages
        tgt_lang = train_example["tgt"]
        label = f"{tgt_lang}: " + train_example["sent2"].strip("\n")

        return prompt_input, label
    
    def __init__(self, tokenizer, src, tgt, split="train", source_file=None, target_file=None, file_path=None, max_length=1024, is_random=False, random_num=0):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.lang_code_dict = {"en": "English", "de": "German", "zh": "Chinese"}
        self.domain_code_list = ['it', 'law', 'med', 'sub']
        # self.domain_code_list = ["edu", "spok", "thes"]
        self.domain_code_dict = {"it": "IT", "med": "medical", "koran": "Koran", "law": "law", "thes": "thesis", "sub": "subtitles", "edu": "education", "spok": "spoken"}

        self.translate_instruction = 'You are a translation specialist who specializes in translating texts from {src} into {tgt} in {domain} domain. Please translate the following content into {tgt} and only reply the translated sentence starting with "{tgt}:" without line breaks or any special tokens.'
        
        self.domain_translation_patterns_with_domain_tgt = [
            ("{lang1}: {sent1}", "{lang2}: {sent2}")
        ]

        self.new_sent_list = []
        shuffle_train = []

        if split == "train":
            few_data_path = "data/high_quality/shuf.all-domain.de-en.json"
            # few_data_path = "data/high_quality_zhen/shuf.all-domain.zh-en.json"
            if os.path.exists(few_data_path):
                with open(few_data_path, 'r', encoding='utf-8') as f:
                    content = json.load(f)
                
                if is_random:
                    random.shuffle(content)
                    content = content[:random_num]
                    with open(f"data/high_quality/shuf.all-domain.de-en.{random_num}.json", 'w', encoding="utf-8") as f:
                        json.dump(content, f, ensure_ascii=False, indent=4)

                for line in content:
                    shuffle_train.append({"src": self.lang_code_dict[src], "tgt": self.lang_code_dict[tgt], "sent1": line["input"], "sent2": line["output"], "domain": line["instruction"].split()[15]})
            else:
                for cur_domain_code in self.domain_code_list:
                    # file_path = f"data/high_quality_zhen/shuf.{cur_domain_code}.train.{src}-{tgt}.jsonl"
                    file_path = f"data/high_quality/shuf.{cur_domain_code}.train.{src}-{tgt}.jsonl"
                    
                    with open(file_path, 'r', encoding='utf-8') as f:
                        lines = f.readlines()

                    self.sent_list = []
                    for line in lines:
                        line = json.loads(line)
                        self.sent_list.append((line["translation"][src].replace("\n", ""), line["translation"][tgt].replace("\n", "")))                    

                    for line in self.sent_list:
                        shuffle_train.append({"src": self.lang_code_dict[src], "tgt": self.lang_code_dict[tgt], "sent1": line[0], "sent2": line[1], "domain": self.domain_code_dict[cur_domain_code]})    
                
                random.shuffle(shuffle_train) 
                shuffle_train = shuffle_train[:10000] #for 4 domains de-en
                # shuffle_train = shuffle_train[:6000] #for 3 domains zh-en

                shuffle_train_save = []
                for line in shuffle_train:
                    content = {
                        "instruction": self.translate_instruction.format(domain=line["domain"], tgt=line["tgt"]),
                        "input": line["sent1"],
                        "output": line["sent2"]
                        }
                    shuffle_train_save.append(content)
                    
                with open("data/high_quality/shuf.all-domain.de-en.json", "w", encoding='utf-8') as f:
                # with open("data/high_quality_zhen/shuf.all-domain.zh-en.json", "w", encoding='utf-8') as f:
                    json.dump(shuffle_train_save, f, ensure_ascii=False, indent=4)
        elif split == "dev":
            for cur_domain_code in self.domain_code_list:
                # file_path = f"data/high_quality_zhen/{cur_domain_code}.dev.{src}-{tgt}.jsonl"
                file_path = f"data/high_quality/{cur_domain_code}.dev.{src}-{tgt}.jsonl"
                with open(file_path, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                
                self.sent_list = []
                
                for line in lines:
                    line = json.loads(line)
                    self.sent_list.append((line["translation"][src].replace("\n", ""), line["translation"][tgt].replace("\n", "")))
                
                for line in self.sent_list:
                    shuffle_train.append({"src": self.lang_code_dict[src], "tgt": self.lang_code_dict[tgt], "sent1": line[0], "sent2": line[1], "domain": self.domain_code_dict[cur_domain_code]})    
                
        
        for example in shuffle_train:
            prompt_input, label = self.construct_translation_prompt(train_example=example, prompt_template=self.domain_translation_patterns_with_domain_tgt, instruction=self.translate_instruction)
            result = tokenizer.apply_chat_template(prompt_input, tokenize=False, add_generation_prompt=False)
            bos_token = ""
            eos_token = ""
            self.new_sent_list.append({"input": result, "label": label})

    def __len__(self):
        return len(self.new_sent_list)

    def __getitem__(self, idx):
        source_text = self.new_sent_list[idx]["input"]
        target_text = self.new_sent_list[idx]["label"]

        source_encoding = self.tokenizer(
            source_text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        ).to("cuda")

        target_encoding = self.tokenizer(
            target_text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        ).to("cuda")

        input_ids = source_encoding["input_ids"]
        label_ids = target_encoding["input_ids"]
       
        labels = torch.cat([torch.full_like(input_ids, -100), label_ids], dim=1)
        input_ids = torch.cat((input_ids, label_ids), dim=1)
        attention_mask = torch.cat((source_encoding["attention_mask"], target_encoding["attention_mask"]), dim=1)

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }
    

