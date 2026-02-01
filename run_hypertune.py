"""
Run hyperparameter tuning
"""

import optuna
import pandas as pd

from sklearn.linear_model import LinearRegression
from src import models
from src.prep import train_data, x_vars
from src.hypertune import cv_eval, objective_cat, objective_lgb, objective_xgb


res_results = {}

#####################################
# setup

y = 'violent_vicoffy'
k = 5
k_folds = models.kfold_split(train_data, k, split='pin')

#####################################
# OLS baseline
rm_ols = models.Mod(
    ide_vars=x_vars,
    y=y,
    bin_y=False,
    mod=LinearRegression(),
)

res_results['ols'] = {'value':cv_eval(rm_ols, data=train_data, ki=k_folds, y_name=y),'params': {}}

#####################################
# CatBoost
study_cat = optuna.create_study(direction='maximize')
study_cat.optimize(objective_cat(x_vars, y, train_data, k_folds), n_trials=60)
res_results['cat'] = {'value':study_cat.best_trial.value,'params': study_cat.best_trial.params}


#####################################
# LightBoost
study_lgb = optuna.create_study(direction='maximize')
study_lgb.optimize(objective_lgb(x_vars, y, train_data, k_folds), n_trials=60)
res_results['lgb'] = {'value':study_lgb.best_trial.value,'params': study_lgb.best_trial.params}


#####################################
# XGBReg
study_xgb = optuna.create_study(direction='maximize')
study_xgb.optimize(objective_xgb(x_vars, y, train_data, k_folds), n_trials=60)
res_results['xgb'] = {'value':study_xgb.best_trial.value,'params': study_xgb.best_trial.params}


####################################
# print + save
print('\n\nTRIAL RESULTS\n\n')
print(f"Best Score ols {res_results['ols']['value']}")
print("Best Params")
print(res_results['ols']['params'])

output = []
output.append({
    "model": "ols",
    "score": res_results["ols"]["value"],
    "params": str(res_results["ols"]["params"])
})

for m, t in res_results.items():
    if m == 'ols':
        continue
    
    score = t["value"]
    params = t["params"]
    print(f"Best Score {m} {score}")
    print("Best Params")
    print(params)

    output.append({
        "model": m,
        "score": score,
        "params": str(params)
    })

df = pd.DataFrame(output)
df.to_csv("./output/tuning_results.csv", index=False)