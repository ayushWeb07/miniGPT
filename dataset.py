# import packages
from datasets import load_dataset
import os
import tiktoken
import torch
from torch.utils.data import Dataset, DataLoader



# constants
dataset_file_path = "tiny_stories.txt"
PIN_MEMORY= torch.cuda.is_available()
DEVICE= torch.device("cuda" if torch.cuda.is_available() else "cpu")

GPT_CONFIG_124M = {
    "vocab_size": 50257,    # Vocabulary size
    "context_length": 128, # Context length
    "emb_dim": 384,         # Embedding dimension
    "n_heads": 6,          # Number of attention heads
    "n_layers": 6,         # Number of layers
    "drop_rate": 0.1,       # Dropout rate
    "qkv_bias": False       # Query-Key-Value bias
}
            
            
# get the input data
with open(dataset_file_path, "r") as f:
    raw_data= f.read()
    
    
    
# custom dataset
class GPTDatasetV1(Dataset):
    
    def __init__(self, tokenizer, raw_data: str, context_size: int, stride: int):
        
        self.tokenizer= tokenizer
        self.context_size= context_size
        self.stride= stride
        
        # tokenize the raw text
        self.token_ids= tokenizer.encode(raw_data, allowed_special= {"<|endoftext|>"})
        
        
        # sliding window approach -> get the inputs and labels
        self.input_ids= []
        self.label_ids= []
        
        for i in range(0, len(self.token_ids) - self.context_size, self.stride):
            
            x= self.token_ids[i:i + self.context_size]
            y= self.token_ids[i+1:i + self.context_size + 1]
            
            self.input_ids.append(torch.tensor(x))
            self.label_ids.append(torch.tensor(y))
            
    
    def __len__(self):
        return len(self.input_ids)
    
    def __getitem__(self, index):
        x= self.input_ids[index]
        y= self.label_ids[index]
        return x, y        
    
    
    
    
# load the dataset -> data loader
def get_dataloader(raw_data: str, pin_memory: bool, context_size: int= 128, stride: int= 128, batch_size: int= 32, shuffle: bool= True):
    
    # get the tokenizer
    tokenizer= tiktoken.get_encoding("gpt2")
    
    # get the dataset
    dataset= GPTDatasetV1(tokenizer, raw_data, context_size, stride)
    
    # get the dataloader
    dataloader= DataLoader(dataset, batch_size= batch_size, shuffle= shuffle, pin_memory= pin_memory)
    
    return dataloader
    
    
    
    
if __name__ == "__main__":
    # store all the data rows -> txt file
    if not os.path.exists(dataset_file_path):
        
        ds = load_dataset("roneneldan/TinyStories") # load the ds from hugging face
        
        # write all the train rows -> file
        with open(dataset_file_path, "a") as f:
            for item in ds["train"]:
                f.write(f"{item['text']}\n")
                
        # write all the valid rows -> file
        with open(dataset_file_path, "a") as f:
            for item in ds["validation"]:
                f.write(f"{item['text']}\n")
                