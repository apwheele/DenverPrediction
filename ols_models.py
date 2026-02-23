import os
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import roc_auc_score

from src import models
from src.prep import train_data, holdout_data, x_vars


DVS = [
    "property_vicoffy",
    "theft_vicoffy",
    "burglary_vicoffy",
    "mvtheft_vicoffy",
]

LABEL_MAP = {
    "property_vicoffy": "property",
    "theft_vicoffy":    "theft",
    "burglary_vicoffy": "burglary",
    "mvtheft_vicoffy":  "mvt",
}

OUT_DIR = "./output/final_model/ols_outputs"
os.makedirs(OUT_DIR, exist_ok=True)


def weighted_auc(y_count, score):
    """Victim-weighted AUC matching models.metrics() convention."""
    y_count = np.asarray(y_count).astype(float)
    y_bin   = (y_count > 0).astype(int)
    sw      = np.clip(y_count, 1, None)
    return float(roc_auc_score(y_bin, score, sample_weight=sw))


metrics_rows  = []
holdout_base  = holdout_data.copy().reset_index(drop=True)

for y in DVS:
    if y not in train_data.columns:
        print(f"[SKIP] {y} not found in train_data")
        continue
    if y not in holdout_base.columns:
        print(f"[SKIP] {y} not found in holdout_data")
        continue

    # ------------------------------------------------------------------
    # Fit OLS on full training data → holdout scores
    # ------------------------------------------------------------------
    ols_full = LinearRegression()
    ols_full.fit(train_data[x_vars], train_data[y])

    holdout          = holdout_base.copy()
    holdout["score"] = ols_full.predict(holdout[x_vars])

    wauc_holdout = weighted_auc(holdout[y].values, holdout["score"].values)

    # ------------------------------------------------------------------
    # 5-fold OOF scores for Platt calibration (same split as full_model.py)
    # ------------------------------------------------------------------
    k       = 5
    k_folds = models.kfold_split(train_data, k, split="pin")

    oof               = train_data.copy().reset_index(drop=True)
    oof["score_oof"]  = np.nan

    for fold in np.unique(k_folds):
        tr     = train_data[k_folds != fold].reset_index(drop=True)
        te_idx = np.where(k_folds == fold)[0]
        te     = train_data.iloc[te_idx].reset_index(drop=True)

        ols_fold = LinearRegression()
        ols_fold.fit(tr[x_vars], tr[y])
        oof.loc[te_idx, "score_oof"] = ols_fold.predict(te[x_vars])

    # ------------------------------------------------------------------
    # Victim-weighted Platt scaling
    # ------------------------------------------------------------------
    y_bin_oof = (oof[y].values > 0).astype(int)
    sw_cal    = np.clip(oof[y].values.astype(float), 1, None)

    cal = LogisticRegression(solver="lbfgs", max_iter=2000)
    cal.fit(oof[["score_oof"]].values, y_bin_oof, sample_weight=sw_cal)

    holdout["prob"]  = cal.predict_proba(holdout[["score"]].values)[:, 1]
    holdout["y_bin"] = (holdout[y] > 0).astype(int)
    holdout["sw"]    = np.clip(holdout[y].astype(float), 1, None)

    # ------------------------------------------------------------------
    # Save predictions
    # ------------------------------------------------------------------
    base_cols = ["pin", y, "y_bin", "score", "prob", "sw"]
    keep_cols = [c for c in base_cols if c in holdout.columns]

    out_path  = os.path.join(OUT_DIR, f"{LABEL_MAP[y]}_ols_holdout_predictions.csv")
    holdout[keep_cols].to_csv(out_path, index=False)

    metrics_rows.append({
        "model":                "ols_regressor",
        "y":                    y,
        "holdout_weighted_auc": wauc_holdout,
        "holdout_n":            int(len(holdout)),
        "holdout_positive_rate": float((holdout[y] > 0).mean()),
        "pred_path":            out_path,
    })

    print(f"[DONE] {y} | Holdout WeightedAUC: {wauc_holdout:.6f} | Saved: {out_path}")


# ------------------------------------------------------------------
# Save combined metrics
# ------------------------------------------------------------------
metrics_df   = pd.DataFrame(metrics_rows)
metrics_path = os.path.join(OUT_DIR, "ols_metrics.csv")
metrics_df.to_csv(metrics_path, index=False)
print("Saved OLS metrics:", metrics_path)