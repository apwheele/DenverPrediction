'''
Hyptertuning models
'''

from src import models
from src.prep import train_data, x_vars
import optuna
import numpy as np

from sklearn.linear_model import LinearRegression
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor


res_results = {}

##############################################
# brief setup
y = 'violent_vicoffy'

# k-folds split by pin
k = 5
k_folds = models.kfold_split(train_data, k, split='pin')

##############################################


##############################################
# Helper functions

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
def cv_eval(rm, data=train_data, ki=k_folds, y_name=y, k_top=1000):
    metric_rows = []
    pei_denoms = []

    kvals = np.unique(ki)
    kvals.sort()

    for fold in kvals:
        train = data[ki != fold].reset_index(drop=True)
        test = data[ki == fold].reset_index(drop=True)

        rm.fit(train)
        test['pred'] = rm.predict(test)

        met = models.metrics(y_name, 'pred', test)
        metric_rows.append(met)

        pei_denoms.append(best_possible_topk(test[y_name], k=k_top))

    return fold_objective(metric_rows, pei_denoms)

##############################################


##############################################
# OLS baseline

rm_ols = models.Mod(
    ide_vars=x_vars,
    y=y,
    bin_y=False,
    mod=LinearRegression())

res_results['ols'] = {'value': cv_eval(rm_ols), 'params': {}}

##############################################


##############################################
# CatBoost hyperparameter

def objective_cat(trial):
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

    score = cv_eval(rm)
    return score


study_cat = optuna.create_study(direction="maximize")
study_cat.optimize(objective_cat, n_trials=60)
trial_cat = study_cat.best_trial
res_results['cat'] = trial_cat

##############################################


##############################################
# LightBoost hyperparameter

def objective_lgb(trial):
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
            min_data_in_leaf=param['min_data_in_leaf']))

    score = cv_eval(rm)
    return score


study_lgb = optuna.create_study(direction="maximize")
study_lgb.optimize(objective_lgb, n_trials=300)
trial_lgb = study_lgb.best_trial
res_results['lgb'] = trial_lgb

##############################################


##############################################
# XGBoost hyperparameter

def objective_xgb(trial):
    param = {
        "n_estimators": trial.suggest_int("n_estimators", 50, 1000),
        "max_depth": trial.suggest_int("max_depth", 2, 10)}
    rm = models.Mod(
        ide_vars=x_vars,
        y=y,
        bin_y=False,
        mod=XGBRegressor(
            n_estimators=param['n_estimators'],
            max_depth=param['max_depth']))

    score = cv_eval(rm)
    return score


study_xgb = optuna.create_study(direction="maximize")
study_xgb.optimize(objective_xgb, n_trials=60)
trial_xgb = study_xgb.best_trial
res_results['xgb'] = trial_xgb


##############################################


# Printing Results
print('\n\nTRIAL RESULTS\n\n')

print(f"Best Score ols {res_results['ols']['value']}")
print("Best Params")
print(res_results['ols']['params'])


for m, t in res_results.items():
    if m == 'ols':
        continue
    print(f"Best Score {m} {t.value}")
    print("Best Params")
    print(t.params)