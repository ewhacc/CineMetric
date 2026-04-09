import pandas as pd
import numpy as np
import re
import os

import joblib
import argparse


class PREPROC_STATIC:
    def __init__(self, data):
        self.data = data
        self.data["length"] = self.data["지문/대사"].map(len)

    def preproc_static(self):
        category = {
            "text" : self.data.종류.unique(),
            "action" : ["지문"],
            "dialog" : ["대사"],
        }
        
        statics = {
            "script_scene_num": self.data["scene"].nunique(),
            "script_cell_num" : len(self.data),
            "script_action_num" : len(self.data[self.data["종류"]=="지문"]),
            "script_dialog_num" : len(self.data[self.data["종류"]=="대사"]),
        }

        statics["script_action_num_per_cell_num"] = statics["script_action_num"]/statics["script_cell_num"]
        statics["script_dialog_num_per_cell_num"] = statics["script_dialog_num"]/statics["script_cell_num"]

        statics.update(
            {
                "script_"+cat+"_length"+ method: self.static_length(
                    self.data.loc[self.data["종류"].isin(category[cat]),"length"],
                    method=method[1:]
                )
                for cat in ["text","action","dialog"]
                for method in ["" "_median", "_avg","_std"]
            }
        )

        statics['script_action_num_per_cell_num'] = statics['script_action_num']/statics['script_cell_num']
        statics['script_dialog_num_per_cell_num'] = statics['script_dialog_num']/statics['script_cell_num']
        
        statics.update(
            self.static_char(self.data["인물"])
        )
        return statics

    def static_length(self, text_length, method="median"):
        methods = {
            "sum" : sum,
            "median" : np.median,
            "avg" : np.mean,
            "std" : np.std,
        }

        if method=="sum":
            return methods[method](text_length)
        return float(methods[method](text_length))

    def static_char(self, chars):
        chars = chars.map(lambda x: re.sub("\([^\}]+\)","",re.sub("[\n ]","",x)))
        chars = [ c for c in chars  if c not in ["", "nan"] ]
        appearances = len(chars)
        
        uni_chars, counts = np.unique(chars, return_counts=True)        
        uni_chars_list = sorted(
            [
                {
                    "name" : str(name), 
                    "count" : int(c), 
                    "ratio": float(int(c)/appearances)
                }
                for name, c in zip(uni_chars, counts)
            ],
            key = lambda x: x["count"],
            reverse=True
        )

        return {
            "charactor_num" : len(uni_chars_list),
            "charactor" : uni_chars_list
        } 

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-I","--input_path", required=True, 
        help="The output file is saved in the same directory as the input file."
    )
    args = parser.parse_args()

    df = joblib.load(args.input_path)
    
    statics = PREPROC_STATIC(df).preproc_static()
    
    output_path = os.path.dirname(os.path.abspath(args.input_path))
    joblib.dump(statics, os.path.join(output_path,'script_static'))

    