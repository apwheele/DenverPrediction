import os
import numpy as np
import pandas as pd

from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from src import models
from src.prep import train_data, holdout_data, x_vars


# ------------------------------------------------------------
# Hypertune outputs -> specs (edit y names here if needed)
# ------------------------------------------------------------
DV_SPECS = {
    # total property: LGB
    "property_vicoffy": {
        "model_type": "lgb",
        "best_params": {"n_estimators": 209, "max_depth": 9, "min_data_in_leaf": 74},
    },
    # burglary: CAT Poisson
    "burglary_vicoffy": {
        "model_type": "cat",
        "best_params": {
            "n_estimators": 681,
            "max_depth": 10,
            "min_data_in_leaf": 83,
            "loss_function": "Poisson",
        },
    },
    # MVT: CAT RMSE
    "mvtheft_vicoffy": {
        "model_type": "cat",
        "best_params": {
            "n_estimators": 579,
            "max_depth": 5,
            "min_data_in_leaf": 80,
            "loss_function": "RMSE",
        },
    },
    # theft: CAT RMSE
    "theft_vicoffy": {
        "model_type": "cat",
        "best_params": {
            "n_estimators": 871,
            "max_depth": 3,
            "min_data_in_leaf": 45,
            "loss_function": "RMSE",
        },
    },
}

OUT_DIR = "./output/final_model"
os.makedirs(OUT_DIR, exist_ok=True)


def weighted_auc_like_models_py(y_count, score):
    """Match models.metrics(): weights = clip(count, 1), label = count>0."""
    y_count = np.asarray(y_count).astype(float)
    y_bin = (y_count > 0).astype(int)
    sw = np.clip(y_count, 1, None)
    return float(roc_auc_score(y_bin, score, sample_weight=sw))


def _build_cat(params):
    return CatBoostRegressor(
        iterations=params["n_estimators"],
        depth=params["max_depth"],
        min_data_in_leaf=params["min_data_in_leaf"],
        loss_function=params.get("loss_function", "RMSE"),
        allow_writing_files=False,
        verbose=False,
        random_seed=10,
    )


def _build_lgb(params):
    return LGBMRegressor(
        n_estimators=params["n_estimators"],
        max_depth=params["max_depth"],
        min_child_samples=params["min_data_in_leaf"],
        random_state=10,
        n_jobs=-1,
        verbosity=-1,
    )


def make_rm(y, spec):
    if spec["model_type"].lower() == "cat":
        base_model = _build_cat(spec["best_params"])
    elif spec["model_type"].lower() == "lgb":
        base_model = _build_lgb(spec["best_params"])
    else:
        raise ValueError(f"Unknown model_type: {spec['model_type']}")

    return models.Mod(
        ide_vars=x_vars,
        y=y,
        bin_y=False,
        mod=base_model,
    )


# ----------------------------
# Run per DV
# ----------------------------
metrics_rows = []

# start from a copy once, then reuse
holdout_base = holdout_data.copy().reset_index(drop=True)

for y, spec in DV_SPECS.items():
    if y not in train_data.columns:
        print(f"[SKIP] DV not found in train_data: {y}")
        continue
    if y not in holdout_base.columns:
        print(f"[SKIP] DV not found in holdout_data: {y}")
        continue

    dv_dir = os.path.join(OUT_DIR, y)
    os.makedirs(dv_dir, exist_ok=True)

    # --------------------
    # Fit on full training
    # --------------------
    rm_full = make_rm(y, spec)
    rm_full.fit(train_data)

    holdout = holdout_base.copy()
    holdout["score"] = rm_full.predict(holdout)

    wauc_holdout = weighted_auc_like_models_py(
        holdout[y].values, holdout["score"].values
    )

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

        rm_fold = make_rm(y, spec)
        rm_fold.fit(tr)
        oof.loc[te_idx, "score_oof"] = rm_fold.predict(te).values

    y_bin = (oof[y].values > 0).astype(int)
    sw_cal = np.clip(oof[y].values.astype(float), 1, None)

    cal = LogisticRegression(solver="lbfgs", max_iter=2000)
    cal.fit(oof[["score_oof"]].values, y_bin, sample_weight=sw_cal)

    holdout["prob"] = cal.predict_proba(holdout[["score"]].values)[:, 1]
    holdout["y_bin"] = (holdout[y] > 0).astype(int)
    holdout["sw"] = np.clip(holdout[y].astype(float), 1, None)

    # --------------------
    # Save predictions
    # --------------------
    # include these if present (handy for comparing DVs side-by-side)
    other_dvs = list(DV_SPECS.keys())
    base_cols = ["pin", "YEAR", y, "y_bin", "score", "prob", "sw"]
    keep_cols = [c for c in base_cols if c in holdout.columns] + [
        c for c in other_dvs if c in holdout.columns and c != y
    ]

    pred_path = os.path.join(dv_dir, "property_holdout_predictions.csv")
    holdout[keep_cols].to_csv(pred_path, index=False)

    # --------------------
    # Save metrics row
    # --------------------
    metrics_rows.append(
        {
            "model": f"{spec['model_type'].lower()}_regressor",
            "y": y,
            "best_params": str(spec["best_params"]),
            "holdout_weighted_auc": wauc_holdout,
            "holdout_top1000_weight": float(
                holdout.sort_values("score", ascending=False).head(1000)[y].sum()
            ),
            "holdout_n": int(len(holdout)),
            "holdout_positive_rate": float((holdout[y] > 0).mean()),
            "pred_path": pred_path,
        }
    )

    print(f"[DONE] {y} | Holdout WeightedAUC: {wauc_holdout:.6f} | Saved: {pred_path}")


# ----------------------------
# Write combined metrics
# ----------------------------
metrics_df = pd.DataFrame(metrics_rows)
metrics_path = os.path.join(OUT_DIR, "property_metrics.csv")
metrics_df.to_csv(metrics_path, index=False)

print("Saved combined metrics:", metrics_path)