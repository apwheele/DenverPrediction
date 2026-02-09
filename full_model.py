import os
import numpy as np
import pandas as pd

from catboost import CatBoostRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from src import models
from src.prep import train_data, holdout_data, x_vars

Y = "violent_vicoffy"

BEST_PARAMS = {
    "n_estimators": 562,
    "max_depth": 3,
    "min_data_in_leaf": 88,
    "loss_function": "RMSE",
}

OUT_DIR = "./output/final_model"
os.makedirs(OUT_DIR, exist_ok=True)

def weighted_auc_like_models_py(y_count, score):
    """Match models.metrics(): weights = clip(count, 1), label = count>0."""
    y_count = np.asarray(y_count).astype(float)
    y_bin = (y_count > 0).astype(int)
    sw = np.clip(y_count, 1, None)   # exactly like .clip(1)
    return float(roc_auc_score(y_bin, score, sample_weight=sw))

def make_rm(params):
    return models.Mod(
        ide_vars=x_vars,
        y=Y,
        bin_y=False,
        mod=CatBoostRegressor(
            iterations=params["n_estimators"],
            depth=params["max_depth"],
            min_data_in_leaf=params["min_data_in_leaf"],
            loss_function=params["loss_function"],
            allow_writing_files=False,
            verbose=False,
            random_seed=10,
        ),
    )

# --------------------
# Fit on full training
# --------------------
rm_full = make_rm(BEST_PARAMS)
rm_full.fit(train_data)

holdout = holdout_data.copy().reset_index(drop=True)
holdout["score"] = rm_full.predict(holdout)

wauc_holdout = weighted_auc_like_models_py(holdout[Y].values, holdout["score"].values)

# --------------------
# Platt scaling (OOF)
# --------------------
k = 5
k_folds = models.kfold_split(train_data, k, split="pin")

oof = train_data.copy().reset_index(drop=True)
oof["score_oof"] = np.nan

for fold in np.unique(k_folds):
    tr = train_data[k_folds != fold].reset_index(drop=True)
    te_idx = np.where(k_folds == fold)[0]
    te = train_data.iloc[te_idx].reset_index(drop=True)

    rm_fold = make_rm(BEST_PARAMS)
    rm_fold.fit(tr)
    oof.loc[te_idx, "score_oof"] = rm_fold.predict(te).values

y_bin = (oof[Y].values > 0).astype(int)

# Calibration weights (optional): match your WeightedAUC weighting
sw_cal = np.clip(oof[Y].values.astype(float), 1, None)

cal = LogisticRegression(solver="lbfgs", max_iter=2000)
cal.fit(oof[["score_oof"]].values, y_bin, sample_weight=sw_cal)

holdout["prob"] = cal.predict_proba(holdout[["score"]].values)[:, 1]
holdout["y_bin"] = (holdout[Y] > 0).astype(int)
holdout["sw"] = np.clip(holdout[Y].astype(float), 1, None)

# Save
pred_path = os.path.join(OUT_DIR, "holdout_predictions.csv")
cols = [c for c in ["pin", "YEAR", Y, "y_bin", "score", "prob", "sw",
                    "theft_vicoffy", "burglary_vicoffy", "mvtheft_vicoffy", "mischief_vicoffy"] if c in holdout.columns]
holdout[cols].to_csv(pred_path, index=False)

metrics_df = pd.DataFrame([{
    "model": "catboost_regressor",
    "y": Y,
    "best_params": str(BEST_PARAMS),
    "holdout_weighted_auc": wauc_holdout,
    "holdout_top1000_weight": float(
        holdout.sort_values("score", ascending=False).head(1000)[Y].sum()
    ),
    "holdout_n": int(len(holdout)),
    "holdout_positive_rate": float((holdout[Y] > 0).mean()),
}])

metrics_path = os.path.join(OUT_DIR, "metrics.csv")
metrics_df.to_csv(metrics_path, index=False)

print("Saved:", pred_path)
print("Saved:", metrics_path)
print("Holdout WeightedAUC:", wauc_holdout)
# Holdout WeightedAUC: 0.7999248874942961