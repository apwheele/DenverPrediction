'''
Hyptertuning models
'''

from src import models
import numpy as np

from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor


##############################################
# define fit stat functions

# define max possible future offense involvements
def best_possible_topk(y_true, k=1000):
    y_true = np.asarray(y_true)
    order = (-1 * y_true).argsort()
    return float(y_true[order][:k].sum())

# create single metric to optimize to
def fold_objective(metric_rows, pei_denoms):
    wauc = float(np.mean([m['WeightedAUC'] for m in metric_rows]))

    pei_vals = []
    for m, d in zip(metric_rows, pei_denoms):
        if d > 0:
            pei_vals.append(float(m['Top1000_Weight']) / d)
        else:
            pei_vals.append(0.0)

    pei = float(np.mean(pei_vals))

    # 50/50 weights, can adjust if needed
    return 0.5 * wauc + 0.5 * pei

# produce metric for model ranking
def cv_eval(rm, data, ki, y_name, k_top=1000):
    metric_rows = []
    pei_denoms = []

    kvals = np.unique(ki)
    kvals.sort()

    for fold in kvals:
        train = data[ki != fold].reset_index(drop=True)
        test = data[ki == fold].reset_index(drop=True)

        rm.fit(train)
        test_eval = test.copy()
        test_eval['pred'] = rm.predict(test_eval)

        met = models.metrics(y_name, 'pred', test_eval)
        metric_rows.append(met)

        pei_denoms.append(best_possible_topk(test[y_name], k=k_top))

    return fold_objective(metric_rows, pei_denoms)

##############################################
# Tuning functions
##############################################
# CatBoost hyperparameter

def objective_cat(x_vars, y, train_data, k_folds, k_top=1000):
    def _objective(trial):
        param = {
            "n_estimators": trial.suggest_int("n_estimators", 50, 1000),
            "max_depth": trial.suggest_int("max_depth", 2, 10),
            "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 1, 100),
            "loss_function": trial.suggest_categorical("loss_function", ["RMSE", "Poisson"])}
        rm = models.Mod(
            ide_vars=x_vars,
            y=y,
            bin_y=False,
            mod=CatBoostRegressor(
                iterations=param['n_estimators'],
                depth=param['max_depth'],
                min_data_in_leaf=param['min_data_in_leaf'],
                loss_function=param['loss_function'],
                allow_writing_files=False,
                verbose=False))
        return cv_eval(rm, data=train_data, ki=k_folds, y_name=y, k_top=k_top)
    return _objective


##############################################
# LightBoost hyperparameter

def objective_lgb(x_vars, y, train_data, k_folds, k_top=1000):
    def _objective(trial):
        param = {
            "n_estimators": trial.suggest_int("n_estimators", 50, 1000),
            "max_depth": trial.suggest_int("max_depth", 2, 10),
            "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 1, 100)}
        rm = models.Mod(
            ide_vars=x_vars,
            y=y,
            bin_y=False,
            mod=LGBMRegressor(
                n_estimators=param['n_estimators'],
                max_depth=param['max_depth'],
                min_data_in_leaf=param['min_data_in_leaf'],
                verbose=-1))
        return cv_eval(rm, data=train_data, ki=k_folds, y_name=y, k_top=k_top)
    return _objective



##############################################
# XGBoost hyperparameter

def objective_xgb(x_vars, y, train_data, k_folds, k_top=1000):
    def _objective(trial):
        param = {
            "n_estimators": trial.suggest_int("n_estimators", 50, 1000),
            "max_depth": trial.suggest_int("max_depth", 2, 10)}
        rm = models.Mod(
            ide_vars=x_vars,
            y=y,
            bin_y=False,
            mod=XGBRegressor(
                n_estimators=param['n_estimators'],
                max_depth=param['max_depth'],
                verbosity=0))
        return cv_eval(rm, data=train_data, ki=k_folds, y_name=y, k_top=k_top)
    return _objective