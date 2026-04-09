import pandas as pd
import pytorch_lightning as pl
import torch.nn as nn
from transformers import ElectraModel, AutoTokenizer
import torch

import numpy as np

from tqdm import tqdm
import re
import os

import joblib
import argparse

device="cpu"
LABELS = [
    "text_complaint",     "text_welcome",    "text_admiration",    "text_fed_up",    "text_gratitude",
    "text_sadness",    "text_anger",    "text_respect",    "text_anticipation",    "text_condescending",
    "text_disappointment",    "text_resolute",    "text_distrust",    "text_proud",    "text_comfort",
    "text_fascination",    "text_caring",    "text_embarrassment",    "text_fear",    "text_despair",
    "text_pathetic",    "text_disgust",    "text_annoyance",    "text_dumbfounded",    "text_neutral",
    "text_self_hatred",    "text_bothersome",    "text_exhaustion",    "text_excitement",    "text_realization",
    "text_guilt",    "text_hatred",    "text_fondness",    "text_flustered",    "text_shock",
    "text_reluctance",    "text_sorrow",    "text_boredom",    "text_compassion",    "text_surprise",
    "text_happiness",    "text_anxiety",    "text_joy",    "text_relief"
]

class KOTEtagger(pl.LightningModule):
        def __init__(self):
            super().__init__()
            self.electra = ElectraModel.from_pretrained("beomi/KcELECTRA-base", revision="v2021").to(device)
            self.tokenizer = AutoTokenizer.from_pretrained("beomi/KcELECTRA-base", revision="v2021")
            self.classifier = nn.Linear(self.electra.config.hidden_size, 44).to(device)
            
        def forward(self, text:str):
            encoding = self.tokenizer.encode_plus(
              text, 
              add_special_tokens=True, 
              max_length=512, 
              return_token_type_ids=False, 
              padding="max_length", 
              return_attention_mask=True, 
              return_tensors="pt", 
            ).to(device)
            output = self.electra(encoding["input_ids"][:,:512], attention_mask=encoding["attention_mask"][:,:512])
            output = output.last_hidden_state[:,0,:]
            output = self.classifier(output)
            output = torch.sigmoid(output)
            torch.cuda.empty_cache()
            
            return output

class PREPROC_EMOTION:
    def __init__(self, data):
        self.data = self.refine_texts(data)

    def refine_texts(self, df):
        refine_text_df = df[df['종류'].isin(['지문','대사'])].reset_index()
        refine_text_df = refine_text_df[refine_text_df['지문/대사'].notna()]
        refine_text_df['지문/대사'] = refine_text_df['지문/대사'].map(lambda x: re.sub('[\x08\x02\x0c\x1d\x1f\u2028\t]','',x))
        refine_text_df['지문/대사'] = refine_text_df['지문/대사'].map(lambda x: re.sub('\r\n','\n',x))
        refine_text_df['지문/대사'] = refine_text_df['지문/대사'].map(lambda x: x.strip('\r'))
        refine_text_df['지문/대사'] = refine_text_df['지문/대사'].map(lambda x: x.replace('\r','\n'))
        refine_text_df = refine_text_df.sort_values(['scene','index']).reset_index(drop=True)
        refine_text_df.scene = refine_text_df.scene.ffill()

        return refine_text_df
        
    def refine_group(self, df):
        new_group = (
            df["scene"].ne(df["scene"].shift()) | 
            df["종류"].ne(df["종류"].shift()) | 
            (df["종류"]=="지문")
        )

        df["group_id"] = new_group.cumsum()

        refined = df.groupby(['scene','group_id'], sort=False).agg(
            type=('종류', 'first'),
            start_cell_id=('지문/대사', lambda x: x.index.min()),
            end_cell_id=('지문/대사', lambda x: x.index.max()),
            text=('지문/대사', '\n'.join)
        ).reset_index(drop=False)

        return refined.rename(
            columns={"scene": "scene_id"}
        )[['scene_id', 'start_cell_id','end_cell_id', 'type', 'text']]

    def preproc_emotion(self):
        # emotion model load
        trained_model = KOTEtagger()
        trained_model.load_state_dict(torch.load("kote_pytorch_lightning.bin"), strict=False)

        emotion_df = pd.DataFrame(self.refine_group(self.data))

        for label in LABELS:
            emotion_df[label]=None

        for i in tqdm(range(len(emotion_df))):
            row = emotion_df.iloc[i]
            text = row['text']
        
            preds = trained_model(text)[0]
            preds = preds.detach().cpu().numpy()
        
            emotion_df.loc[i, LABELS] = preds

        return emotion_df

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-I","--input_path", required=True, 
        help="The output file is saved in the same directory as the input file."
    )
    args = parser.parse_args()

    df = joblib.load(args.input_path)
    
    emotions = PREPROC_EMOTION(df).preproc_emotion()
    mean_data = {label: float(np.mean(emotions[label])) for label in LABELS}

    output_path = os.path.dirname(os.path.abspath(args.input_path))
    joblib.dump(mean_data, os.path.join(output_path,'script_emotion'))