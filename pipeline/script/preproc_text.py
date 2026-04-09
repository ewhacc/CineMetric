import pandas as pd
import os

import argparse
import joblib

class PREPROC_SCENE:
    def __init__(self, filename, encoding="utf-8"):
        _, ext = os.path.splitext(filename)
        if ext == ".csv":
            self.data = pd.read_csv(filename, sep=",", encoding=encoding)
        elif ext ==".xlsx":
            self.data = pd.read_excel(filename)
        else:
            print("Check your DATA. Only xlsx and csv work")

    def preproc_scene(self):
        self.data["인물"] = self.data["인물"].fillna("")        # 이부분이 무시됨
        self.data["인물"] = self.data["인물"].map(str.strip)
        
        self.data["지문/대사"] = self.data["지문/대사"].fillna("")     # 이 부분이 무시됨
        self.data["지문/대사"] = self.data["지문/대사"].map(str.strip)

        return self.data


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-I","--input_path", required=True, 
        help=""
    )
    args = parser.parse_args()

    data = PREPROC_SCENE(args.input_path).preproc_scene()

    output_path = os.path.splitext(os.path.abspath(args.input_path))[0]
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    joblib.dump(data, os.path.join(output_path, "script_prerpoc"))
    