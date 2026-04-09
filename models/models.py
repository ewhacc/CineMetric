from pycaret.containers.models.regression import (
    get_all_model_containers as get_all_reg_model_containers,
)
from pycaret.containers.models.classification import (
    get_all_model_containers as get_all_cls_model_containers,
)
from pycaret.regression import setup
import pandas as pd
from copy import deepcopy

from sklearn.cross_decomposition import PLSRegression
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.preprocessing import StandardScaler
from scipy.stats import boxcox
from scipy.special import inv_boxcox
import numpy as np

def get_models():
    exp = setup(
        data=pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6], "c": [7, 8, 9]}),
        verbose=False,
    )
    all_reg_containers = get_all_reg_model_containers(exp)
    all_cls_containers = get_all_cls_model_containers(exp)
    
    all_models = {
        model_id: {
            "class_def": container.class_def,
            # "tune_grid": container.tune_grid,
            # "tune_distribution": container.tune_distribution,
        }
        for model_id, container in all_reg_containers.items()
    }
    all_models.pop("dummy")

    return all_models




def get_person_score(id_list, scores_map):
    scores = [scores_map[actor] for actor in id_list if actor in scores_map]
    # return np.mean(scores) if scores else np.nan
    return np.mean(scores) if scores else 0

class FeatureScaler(BaseEstimator, TransformerMixin):
    def __init__(self, use_columns=[], pls_args=[], target='전국관객수'):
        self.after_scaler = StandardScaler()
        self.pls_args = pls_args
        self.columns = use_columns
        self.target = target
        
    def fit(self, X_input, y_input):
        X = deepcopy(X_input)
        y = deepcopy(y_input)

        self.X_mean_dataset = X.mean(0)
        
        for pls_arg in self.pls_args:
            pls_arg['pls'] = PLSRegression(n_components=pls_arg['n_components'])
            pls_arg['pls'].fit(X[pls_arg['columns']], y)
            X_trans = pls_arg['pls'].transform(X[pls_arg['columns']])
            
            for i in range(pls_arg['n_components']):
                column_name = f"{pls_arg['prefix']}_{i}"
                self.columns.append(column_name)
                X[column_name] = X_trans[:, i]

            for column in pls_arg['columns']:
                try:
                    self.columns.remove(column)
                except ValueError:
                    pass

        X[self.target] = y
        if 'actor_score' in self.columns:
            column = 'lead_actor_ids'
            self.actor_score_map = X.explode([column]).groupby(by=[column])[self.target].apply(lambda x: x.nlargest(3).mean()).to_dict()
            X['actor_score'] = X['lead_actor_ids'].apply(lambda x : get_person_score(x, self.actor_score_map))

        if 'director_score' in self.columns:
            column = 'director_ids'
            self.director_score_map = X.explode([column]).groupby(by=[column])[self.target].mean().to_dict()
            X['director_score'] = X['director_ids'].apply(lambda x : get_person_score(x, self.director_score_map))

        for column in ['lead_actor_ids', 'director_ids']:
            try:
                self.columns.remove(column)
            except ValueError:
                pass

        self.X_mean = X[self.columns].mean(0)
        
        self.after_scaler.fit(X[self.columns])

        

    def transform(self, X_input):
        X = deepcopy(X_input)
        
        for pls_arg in self.pls_args:
            X_trans = pls_arg['pls'].transform(X[pls_arg['columns']])
            for i in range(pls_arg['n_components']):
                column_name = f"{pls_arg['prefix']}_{i}"
                X[column_name] = X_trans[:, i]

        if 'actor_score' in self.columns:                
            X['actor_score'] = X['lead_actor_ids'].apply(lambda x : get_person_score(x, self.actor_score_map))

        if 'director_score' in self.columns:
            X['director_score'] = X['director_ids'].apply(lambda x : get_person_score(x, self.director_score_map))
            
        X[self.columns] = X[self.columns].fillna(self.X_mean[self.columns])

        return self.after_scaler.transform(X[self.columns])




class TargetScaler(BaseEstimator, TransformerMixin):
    def __init__(self, lmbda=None):
        self.lmbda = lmbda
        self.after_scaler = StandardScaler()
        pass
        
    def fit(self, y=None):
        y = np.asarray(y)
        y_t, lmbda = boxcox(y)
        self.lmbda = lmbda
        self.after_scaler.fit(y_t.reshape(-1, 1))
        
    def transform(self, y):
        if self.lmbda is not None:
            y_t = boxcox(y, lmbda=self.lmbda).reshape(-1, 1)
            y_t = self.after_scaler.transform(y_t)
        else:
            raise ''
        
        return y_t
        
    def inverse_transform(self, y_t):
        if self.lmbda is not None:
            y_inv = self.after_scaler.inverse_transform(y_t)
            y_inv = inv_boxcox(y_inv, self.lmbda)
        else:
            raise ''
        return y_inv




class StackingEnsembleRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, base_models=None, meta_model=None):
        self.base_models = base_models
        self.meta_model = meta_model

    def fit(self, X, y):
        X, y = check_X_y(X, y)
        
        self.base_models_ = []
        base_predictions = []
        
        for model in self.base_models:
            instance = clone(model)
            instance.fit(X, y)
            self.base_models_.append(instance)
            
            preds = instance.predict(X).reshape(-1, 1)
            base_predictions.append(preds)
        
        meta_features = np.hstack(base_predictions)
        if not np.isfinite(meta_features).all():
            meta_features = np.nan_to_num(meta_features, nan=0.0, posinf=1e30, neginf=-1e30)
        meta_features = np.clip(meta_features, a_min=-1e30, a_max=1e30)
                                
        combined_features = np.hstack([X, meta_features])
        
        self.meta_model_ = clone(self.meta_model)
        self.meta_model_.fit(combined_features, y)
        
        self.is_fitted_ = True
        return self

    def predict(self, X):
        check_is_fitted(self, 'is_fitted_')
        X = check_array(X)
        
        base_predictions = []
        for model in self.base_models_:
            preds = model.predict(X).reshape(-1, 1)
            base_predictions.append(preds)
        
        meta_features = np.hstack(base_predictions)
        if not np.isfinite(meta_features).all():
            meta_features = np.nan_to_num(meta_features, nan=0.0, posinf=1e30, neginf=-1e30)
        meta_features = np.clip(meta_features, a_min=-1e30, a_max=1e30)
                                
        combined_features = np.hstack([X, meta_features])
        
        return self.meta_model_.predict(combined_features)