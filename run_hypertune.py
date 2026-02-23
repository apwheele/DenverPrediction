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

# y = 'property_vicoffy'

# alternate y vals for single prop crime models
# y = 'theft_vicoffy'
# y = 'burglary_vicoffy'
y = 'mvtheft_vicoffy'
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

score, wauc, pei, top1k = cv_eval(rm_ols, data=train_data, ki=k_folds, y_name=y)

res_results['ols'] = {
    'value': score,
    'params': {},
    'wauc': wauc,
    'pei': pei,
    'top1k': top1k
}

#####################################
# CatBoost
study_cat = optuna.create_study(direction='maximize')
study_cat.optimize(objective_cat(x_vars, y, train_data, k_folds), n_trials=60)
res_results['cat'] = {
    'value': study_cat.best_trial.value,
    'params': study_cat.best_trial.params,
    'wauc': study_cat.best_trial.user_attrs.get("WeightedAUC"),
    'pei': study_cat.best_trial.user_attrs.get("PEI"),
    'top1k': study_cat.best_trial.user_attrs.get("Top1000_Weight"),
   }



#####################################
# LightBoost
study_lgb = optuna.create_study(direction='maximize')
study_lgb.optimize(objective_lgb(x_vars, y, train_data, k_folds), n_trials=60)
res_results['lgb'] = {
    'value': study_lgb.best_trial.value,
    'params': study_lgb.best_trial.params,
    'wauc': study_lgb.best_trial.user_attrs.get("WeightedAUC"),
    'pei': study_lgb.best_trial.user_attrs.get("PEI"),
    'top1k': study_lgb.best_trial.user_attrs.get("Top1000_Weight"),
   }


#####################################
# XGBReg
study_xgb = optuna.create_study(direction='maximize')
study_xgb.optimize(objective_xgb(x_vars, y, train_data, k_folds), n_trials=60)
res_results['xgb'] = {
    'value': study_xgb.best_trial.value,
    'params': study_xgb.best_trial.params,
    'wauc': study_xgb.best_trial.user_attrs.get("WeightedAUC"),
    'pei': study_xgb.best_trial.user_attrs.get("PEI"),
    'top1k': study_xgb.best_trial.user_attrs.get("Top1000_Weight"),
   }



####################################
# print + save
print('\n\nTRIAL RESULTS\n\n')

for m, t in res_results.items():
    print(f"Best Score {m}: {t['value']}")
    print("WeightedAUC:", t.get("wauc"))
    print("PEI:", t.get("pei"))
    print("Top1000:", t.get("top1k"))
    print("Best Params:")
    print(t["params"])
    print("")

output = []
for m, t in res_results.items():
    output.append({
        "model": m,
        "score": t["value"],
        "weighted_auc": t.get("wauc"),
        "pei": t.get("pei"),
        "top1000_weight": t.get("top1k"),
        "params": str(t["params"])
    })

df = pd.DataFrame(output)
df.to_csv("./output/mvt_tuning_results.csv", index=False)