#dataset to handle loading and batching 

import json
import torch
from torch.utils.data import Dataset, DataLoader
from .tokenizer import GarmentTokenizer
from .config import ModelConfig

class TechPackDataset(Dataset):
    #loads flattened training examples

    def __init__(self, data_file, tokenizer, max_seq_length = 128):


        self.tokenizer = tokenizer
        self.max_length = max_seq_length

        #load data
        with open(data_file, 'r', encoding='utf-8') as f:
            self.data = json.load(f)

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        #single training example, return source ids and target ids

        example = self.data[idx]

        #get input and output text
        input_text = example['input']
        output_text = example['output']

        #tokenize
        src_ids = self.tokenizer.encode(input_text, max_length=self.max_length)
        tgt_ids = self.tokenizer.encode(output_text, max_length=self.max_length)

        #convert to tensors
        src_ids = torch.tensor(src_ids, dtype=torch.long)
        tgt_ids = torch.tensor(tgt_ids, dtype=torch.long)

        return src_ids, tgt_ids
    
def create_dataLoaders(config):
        #create train and validation dataloaders, return train loader, val loader and tokenizer

        #load tokenizer
        tokenizer = GarmentTokenizer()
        tokenizer.load(config.tokenizer_file)

        #create datasets
        train_dataset = TechPackDataset(config.train_file, tokenizer, config.max_seq_length)
        val_dataset = TechPackDataset(config.val_file, tokenizer, config.max_seq_length)

        #create dataloaders
        train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=config.batch_size)

        print (f"Train dataset size: {len(train_dataset)} examples")
        print (f"Validation dataset size: {len(val_dataset)} examples")
        print (f"Batch size: {config.batch_size}")

        return train_loader, val_loader, tokenizer
    
if __name__ == "__main__":
    #test dataset loading and batching

    config = ModelConfig()
    train_loader, val_loader, tokenizer = create_dataLoaders(config)

     # Test batch
    print(f"\n Testing baatch")
    for src, tgt in train_loader:
        print(f"\n   Batch shapes:")
        print(f"   Source: {src.shape}")
        print(f"   Target: {tgt.shape}")
        
        # Decode first example
        print(f"\n   First example:")
        src_text = tokenizer.decode(src[0].tolist())
        tgt_text = tokenizer.decode(tgt[0].tolist())
        
        print(f"   Input:  {src_text[:80]}...")
        print(f"   Output: {tgt_text[:80]}...")
        
        break





