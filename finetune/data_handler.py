import csv
import torch
import json
from loguru import logger
from torch.utils.data import Dataset
from utils import Recorder
from torch.nn.utils.rnn import pad_sequence
from transformers import T5Tokenizer, AutoTokenizer

tokenizer = None

@Recorder
def get_data(args):
    global tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    data = {
        'train': FinetuneDataset(args.train_path, args.task),
        'valid': FinetuneDataset(args.valid_path, args.task),
        'test': FinetuneDataset(args.test_path, args.task)
    }
    return data

@Recorder
def get_test_data(args):
    global tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    data = {
        'test': FinetuneDataset(args.test_path, args.task)
    }
    return data

@Recorder
def get_train_data(args):
    global tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    data = {
        'test': FinetuneDataset(args.train_path, args.task)
    }
    return data
    
class FinetuneDataset(Dataset):
    def __init__(self, filepath, task='smiles2cap'):

        with open(filepath, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        self.data = [[item['smiles'], item['description']] for item in self.data]
        logger.info(f"Load data from {filepath}")
        
        # for debug only
        # self.data = self.data[:32]

        self.tokenizer = tokenizer
        self.task = task
        logger.info(f"Task: {task}")

        max_source_length = 512
        max_target_length = 512
        logger.info(f"Max source length: {max_source_length}")
        logger.info(f"Max target length: {max_target_length}")

        if self.task == 'smiles2cap':
            self.smiles = [tokenizer(item[0], padding=False, return_tensors='pt', max_length=max_source_length, truncation=True) for item in self.data]
            self.descriptions = [tokenizer(item[1], padding=False, return_tensors='pt', max_length=max_target_length, truncation=True) for item in self.data]
        elif self.task == 'cap2smiles':
            self.smiles = [tokenizer(item[0], padding=False, return_tensors='pt', max_length=max_target_length, truncation=True) for item in self.data]
            self.descriptions = [tokenizer(item[1], padding=False, return_tensors='pt', max_length=max_source_length, truncation=True) for item in self.data]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if self.task == 'smiles2cap':
            smiles = self.smiles[idx]
            description = self.descriptions[idx]
            return {
                'input_text': smiles,
                'output_text': description.input_ids
            }
        elif self.task == 'cap2smiles':
            smiles = self.smiles[idx]
            description = self.descriptions[idx]
            return {
                'input_text': description,
                'output_text': smiles.input_ids
            }

def collate_fn(batch):
    padding_value = tokenizer.pad_token_id

    input_ids = [item['input_text']['input_ids'].squeeze(0) for item in batch]
    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=padding_value)

    attention_mask = [item['input_text']['attention_mask'].squeeze(0) for item in batch]
    attention_mask = pad_sequence(attention_mask, batch_first=True, padding_value=0)

    labels = [item['output_text'].squeeze(0) for item in batch]
    labels = pad_sequence(labels, batch_first=True, padding_value=padding_value)
    labels[labels == padding_value] = -100
    
    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'labels': labels
    }