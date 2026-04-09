from models.params import models_params
from models.metrics import (calc_ccc, 
                     calc_aphr_away, 
                     calc_accuracy_at_std)
from models.models import (get_models,
                    StackingEnsembleRegressor,
                    FeatureScaler,
                    TargetScaler)
from models.variables import feature_columns
from joblib import dump, load
import pandas as pd
import numpy as np
import itertools
import os

base_features = list(itertools.chain.from_iterable(feature_columns.values()))
feature_space = pd.DataFrame([], columns=base_features)

fold = 5
fss = [load(f'save_models/{fold_n}/fs') for fold_n in range(fold)]
tss = [load(f'save_models/{fold_n}/ts') for fold_n in range(fold)]
models = [load(f'save_models/{fold_n}/model') for fold_n in range(fold)]

input_dir = 'data/generated_sample_data'

for feature_name in ['script_static', 'script_emotion', 'script_embedding']:
    features_path = os.path.join('data/generated_sample_data', feature_name)
    features = load(features_path)
    if 'script_embedding' == feature_name:
        feature_space.loc[0, [f'feature_{i}' for i in range(len(features))]] = features       
    else:
        for column, value in features.items():
            if column in feature_space.columns:
                feature_space.loc[0, column] = value

no_features_columns = feature_space.columns[np.where(feature_space.isna())[1]]

total = []
for (fs, ts, model) in zip(fss, tss, models):
    feature_space[no_features_columns] = pd.DataFrame(fs.X_mean_dataset).T[no_features_columns]
    feature_space['lead_actor_ids'] = [[]]
    feature_space['director_ids'] = [[]]
    features = fs.transform(feature_space)
    predict = model.predict(features)
    predict = ts.inverse_transform(predict[np.newaxis,:]).squeeze()
    total.append(predict)

print('Estimated audience')
print(np.mean(total).astype(int))
