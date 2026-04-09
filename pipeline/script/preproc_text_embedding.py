import torch
import torch.nn.functional as F

from torch import Tensor
from transformers import AutoTokenizer, AutoModel

from tqdm import tqdm
import numpy as np

import os
import joblib
import argparse

device="cpu"

class PREPROC_EMBEDDING:
    def __init__(self, df):
        self.tokenizer = AutoTokenizer.from_pretrained('Alibaba-NLP/gte-Qwen2-7B-instruct', trust_remote_code=True)
        self.model = AutoModel.from_pretrained('Alibaba-NLP/gte-Qwen2-7B-instruct', trust_remote_code=True).to(device)

        self.max_length = 8192

        self.df = df

    def make_embed(self, text):
        with torch.no_grad():
            batch_dict = self.tokenizer(
                text,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            batch_dict.to(device)
            outputs = self.model(**batch_dict, use_cache=False)
            embeddings = self.last_token_pool(outputs.last_hidden_state, batch_dict['attention_mask'])
            
            # normalize embeddings
            embeddings = F.normalize(embeddings, p=2, dim=1)
            return f"{list(embeddings.cpu().detach().numpy()[0].tolist())}"
        
    def last_token_pool(self, last_hidden_states: Tensor,
                     attention_mask: Tensor) -> Tensor:
        left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
        if left_padding:
            return last_hidden_states[:, -1]
        else:
            sequence_lengths = attention_mask.sum(dim=1) - 1
            batch_size = last_hidden_states.shape[0]
            return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]
    
    
    def get_detailed_instruct(self, task_description: str, query: str) -> str:
        return f'Instruct: {task_description}\nQuery: {query}'

    def get_token_length(self, text):
        with torch.no_grad():
            batch_dict = self.tokenizer(
                text,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                add_special_tokens=False,
                return_tensors="pt",
                )
            return batch_dict['input_ids'].shape[-1]
    
    def get_min_token_text(self, text, num):
        return self.tokenizer.decode( 
            self.tokenizer(
                    text,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    add_special_tokens=False,
                    return_tensors="pt",
                    )['input_ids'][0],
            skip_special_tokens=True
            )[:num]


    def preproc_embedding(self):
        data = self.df.groupby('scene')[['지문/대사']].agg(lambda x: '\n'.join(x)).reset_index()

        st_text = data.iloc[0,-1]
        end_text = data.iloc[-1,-1]
        st_tkn = self.get_token_length(st_text)
        end_tkn = self.get_token_length(end_text)

        min_data = min(st_tkn, end_tkn)
        st_text_ = self.get_min_token_text(st_text,min_data)
        end_text_ = self.get_min_token_text(end_text,min_data)
        data['embedding']=None

        for i in tqdm(range(len(data))):
            row = data.iloc[i]
        
            if i == 0:
                text = st_text_
            elif i == len(data)-1:
                text = end_text_
            else:
                text = row['지문/대사']
        
            embedding = self.make_embed(text)
        
            data.loc[i, 'embedding']=embedding

        return data

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-I","--input_path", required=True, 
        help="The output file is saved in the same directory as the input file."
    )
    args = parser.parse_args()

    df = joblib.load(args.input_path)
    
    embeddings = PREPROC_EMBEDDING(df).preproc_embedding()
    mean_data = np.mean(
        embeddings['embedding'].map(lambda x: np.array(eval(x)))
    )

    output_path = os.path.dirname(os.path.abspath(args.input_path))
    joblib.dump(mean_data, os.path.join(output_path, 'script_embedding'))