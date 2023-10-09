from torch.utils.data import Dataset
from transformers import AutoTokenizer
import torch

class CustomDataset(Dataset):
    def __init__(self, data, max_sent_len):
        self.data = data 
        self.max_sent_len = max_sent_len#
        self.max_seq_len = 0
        # print(data)
        self.tokenizer = AutoTokenizer.from_pretrained('bkai-foundation-models/vietnamese-bi-encoder')

        for idx in range(len(self.data)):
            padding_sent = ["" for ii in range(self.max_sent_len - len(self.data[idx]["context"]))]
        
            self.data[idx]["context_sentence_mask"] = torch.zeros(self.max_sent_len, dtype=int)
            self.data[idx]["context_sentence_mask"][:len(self.data[idx]["context"])] = torch.ones(len(self.data[idx]["context"]), dtype=int)

            self.data[idx]["context"] = self.data[idx]["context"] + padding_sent

            self.data[idx]["context"] = self.tokenizer(self.data[idx]["context"], padding=True, truncation=True, return_tensors='pt')

            self.data[idx]["claim"] = self.tokenizer(self.data[idx]["claim"], padding=True, truncation=True, return_tensors='pt')
            self.max_seq_len = max(self.max_seq_len, self.data[idx]["context"]["input_ids"].size()[-1], self.data[idx]["claim"]["input_ids"].size()[-1])
            # print(self.max_seq_len)
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
 
   
        return self.data[idx]