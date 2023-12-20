import json
import os
from torch.utils.data import Dataset
from utils import Recorder
from loguru import logger
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoTokenizer

tokenizer = None

@Recorder
def get_data(args):
    global tokenizer
    os.environ["TOKENIZERS_PARALLELISM"] = "true"
    tokenizer = AutoTokenizer.from_pretrained(args.model, max_length=512)
    data = {
        'train': PretrainDataset(args.train_path),
    }
    return data
    
class PretrainDataset(Dataset):
    def __init__(self, filepath):

        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        logger.info(f"Loaded {len(data)} data from {filepath}")
        self.data = [[item['smiles'], item['description']] for item in data]
        logger.info(f"Finish parsing data")

        self.tokenizer = tokenizer

        max_source_length = 512
        max_target_length = 512
        logger.info(f"max_source_length: {max_source_length}, max_target_length: {max_target_length}")
        self.smiles = [tokenizer(item[0], padding=False, return_tensors='pt', max_length=max_source_length, truncation=True) for item in self.data]
        logger.info(f"Finish tokenizing smiles")
        self.descriptions = [tokenizer(item[1], padding=False, return_tensors='pt', max_length=max_target_length, truncation=True) for item in self.data]
        logger.info(f"Finish tokenizing descriptions")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        smiles = self.smiles[idx]
        description = self.descriptions[idx]
        return {
            'smiles': smiles,
            'description': description
        }


def collate_fn(batch):
    padding_value = tokenizer.pad_token_id

    # split batch into two parts: smiles -> description and description -> smiles
    smile2cap_batch = batch[:len(batch)//2]
    cap2smile_batch = batch[len(batch)//2:]

    # concatenate input_ids from two parts and pad into a batch
    smile2cap_input_ids = [item['smiles']['input_ids'].squeeze(0) for item in smile2cap_batch]
    cap2smile_input_ids = [item['description']['input_ids'].squeeze(0) for item in cap2smile_batch]
    input_ids = pad_sequence(smile2cap_input_ids + cap2smile_input_ids, batch_first=True, padding_value=padding_value)

    # concatenate attention_mask from two parts and pad into a batch
    smile2cap_attention_mask = [item['smiles']['attention_mask'].squeeze(0) for item in smile2cap_batch]
    cap2smile_attention_mask = [item['description']['attention_mask'].squeeze(0) for item in cap2smile_batch]
    attention_mask = pad_sequence(smile2cap_attention_mask + cap2smile_attention_mask, batch_first=True, padding_value=0)

    # concatenate labels from two parts and pad into a batch
    smile2cap_labels = [item['description']['input_ids'].squeeze(0) for item in smile2cap_batch]
    cap2smile_labels = [item['smiles']['input_ids'].squeeze(0) for item in cap2smile_batch]
    labels = pad_sequence(smile2cap_labels + cap2smile_labels, batch_first=True, padding_value=padding_value)
    labels[labels == padding_value] = -100

    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'labels': labels
    }
