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

seed=42

dataset = load('data/dataset')
target = '전국관객수'

all_models = get_models()

base_features = list(itertools.chain.from_iterable(feature_columns.values()))
for fold_n in range(5):
    
    train_features_fold = dataset[dataset['fold'] != fold_n]
    test_features_fold = dataset[dataset['fold'] == fold_n]
    
    train_features = train_features_fold[base_features + ['lead_actor_ids', 'director_ids']]
    train_labels = train_features_fold[target]

    test_features = test_features_fold[base_features + ['lead_actor_ids', 'director_ids']]
    test_labels = test_features_fold[target]

    test_features = test_features.fillna(train_features.mean(0))
    train_features = train_features.fillna(train_features.mean(0))
    
    ts = TargetScaler()
    ts.fit(train_labels)
    
    train_labels = ts.transform(train_labels)
    test_labels = ts.transform(test_labels)

    fs = FeatureScaler(
        use_columns=test_features.columns.tolist() + ['actor_score', 'director_score'],  
        pls_args=[
            {'columns':feature_columns['text_emotion'], 'n_components':4, 'prefix': 'text_emotion'},
            {'columns':feature_columns['embedding'], 'n_components':2, 'prefix': 'embedding'},
            {'columns':feature_columns['video_emotion'], 'n_components':2, 'prefix': 'video_emotion'},
            {'columns':feature_columns['music_mood'], 'n_components':2, 'prefix': 'music_mood'},
        ],
        target=target
    )

    fs.fit(train_features, train_labels)
    train_features = fs.transform(train_features)
    test_features = fs.transform(test_features)

    model = StackingEnsembleRegressor(
        base_models=[
            all_models[model_id]['class_def'](**models_params[model_id]) 
                for model_id in models_params.keys()
        ],
        meta_model=all_models['gbr']['class_def'](**{
            'random_state': seed,
            'subsample': 0.7, 
            'n_estimators': 270, 
            'min_samples_split': 10, 
            'min_samples_leaf': 4, 
            'min_impurity_decrease': 0.3, 
            'max_features': 'sqrt', 
            'max_depth': 6, 
            'learning_rate': 0.05
        })
    )

    model.fit(train_features, train_labels)
    os.makedirs(f'save_models/{fold_n}', exist_ok=True)
    dump(fs, f'save_models/{fold_n}/fs')
    dump(ts, f'save_models/{fold_n}/ts')
    dump(model, f'save_models/{fold_n}/model')

    ## Calculate metrics
    train_predicts = model.predict(train_features)
    test_predicts = model.predict(test_features)

    train_labels = train_labels.squeeze()
    test_labels = test_labels.squeeze()
    
    train_acc_std = calc_accuracy_at_std(train_predicts, train_labels)
    train_ccc = calc_ccc(train_predicts, train_labels)
    train_aphr_1_away = calc_aphr_away(
        ts.inverse_transform(train_predicts[np.newaxis, :]).squeeze(), 
        ts.inverse_transform(train_labels[np.newaxis, :]).squeeze()
    )

    test_acc_std = calc_accuracy_at_std(test_predicts, test_labels)
    test_ccc = calc_ccc(test_predicts, test_labels)
    test_aphr_1_away = calc_aphr_away(
        ts.inverse_transform(test_predicts[np.newaxis, :]).squeeze(), 
        ts.inverse_transform(test_labels[np.newaxis, :]).squeeze()
    )

    print('train metrics')
    print(f'\taccuracy at 1std : {train_acc_std:0.4f}')
    print(f'\t             ccc : {train_ccc:0.4f}')
    print(f'\t     aphr 1 away : {train_aphr_1_away:0.4f}')

    print('test metrics')
    print(f'\taccuracy at 1std : {test_acc_std:0.4f}')
    print(f'\t             ccc : {test_ccc:0.4f}')
    print(f'\t     aphr 1 away : {test_aphr_1_away:0.4f}')

