import os
import pickle
import pandas as pd
import numpy as np

from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import roc_auc_score

from src import models

# params identified via hypertune
DV_SPECS = {
    "property_vicoffy": {
        "model_type": "lgb",
        "best_params": {"n_estimators": 209,
                        "max_depth": 9,
                        "min_data_in_leaf": 74},
    },
    "burglary_vicoffy": {
        "model_type": "cat",
        "best_params": {"n_estimators": 681,
                        "max_depth": 10,
                        "min_data_in_leaf": 83,
                        "loss_function": "Poisson"},
    },
    "mvtheft_vicoffy": {
        "model_type": "cat",
        "best_params": {"n_estimators": 579,
                        "max_depth": 10,
                        "min_data_in_leaf": 83,
                        "loss_function": "Poisson"},
    },
    "theft_vicoffy": {
        "model_type": "cat",
        "best_params": {"n_estimators": 871,
                        "max_depth": 3,
                        "min_data_in_leaf": 45,
                        "loss_function": "RMSE"},
    },
}

def weighted_auc(y_count, score):
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

def make_rm(y, spec, x_vars):
    mt = spec["model_type"].lower()
    if mt == "cat":
        base = _build_cat(spec["best_params"])
    elif mt == "lgb":
        base = _build_lgb(spec["best_params"])
    elif mt == "ols":
        base = LinearRegression()
    else:
        raise ValueError(f"Unknown model_type: {spec['model_type']}")
    return models.Mod(ide_vars=x_vars, y=y, bin_y=False, mod=base)

def train_all(train_data, holdout_data, x_vars, out_dir, k=5, dv_specs=None, include_ols=False):
    dv_specs = dv_specs or DV_SPECS
    os.makedirs(out_dir, exist_ok=True)

    holdout_base = holdout_data.copy().reset_index(drop=True)
    metrics_rows = []

    for y, spec in dv_specs.items():
        if y not in train_data.columns or y not in holdout_base.columns:
            print(f"[SKIP] DV missing: {y}")
            continue

        dv_dir = os.path.join(out_dir, y)
        os.makedirs(dv_dir, exist_ok=True)
        # fit on training dataset
        rm_full = make_rm(y, spec, x_vars)
        rm_full.fit(train_data)
        with open(os.path.join(dv_dir, "model.pkl"), "wb") as f:
            pickle.dump(rm_full, f)
        
        holdout = holdout_base.copy()
        holdout["score"] = rm_full.predict(holdout)
        wauc_holdout = weighted_auc(holdout[y].values,
                                    holdout["score"].values)
        
        # Platt calibration
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
        with open(os.path.join(dv_dir, "calibrator.pkl"), "wb") as f:
            pickle.dump(cal, f)
        
        holdout["prob"] = cal.predict_proba(holdout[["score"]].values)[:, 1]
        holdout["y_bin"] = (holdout[y] > 0).astype(int)
        holdout["sw"] = np.clip(holdout[y].astype(float), 1, None)

        base_cols = ["pin", "YEAR", y, "y_bin", "score", "prob", "sw"]
        other_dvs = [c for c in dv_specs if c != y and c in holdout.columns]
        keep_cols = [c for c in base_cols if c in holdout.columns] + other_dvs
        pred_path = os.path.join(dv_dir, f"{y}_holdout_predictions.csv")
        holdout[keep_cols].to_csv(pred_path, index=False)

        metrics_rows.append({
            "model": f"{spec['model_type'].lower()}_regressor",
            "y": y,
            "best_params": str(spec["best_params"]),
            "holdout_weighted_auc": wauc_holdout,
            "holdout_top1000_weight": float(
                holdout.sort_values("score", ascending=False).head(1000)[y].sum()),
            "holdout_n": int(len(holdout)),
            "holdout_positive_rate": float((holdout[y] > 0).mean()),
            "pred_path": pred_path,
        })
        print(f"[DONE] {y} | Holdout WeightedAUC: {wauc_holdout:.6f}")

        if include_ols:
            _fit_ols_baseline(y, train_data, holdout_base, k_folds, 
                            os.path.join(out_dir, "ols_outputs"), metrics_rows, x_vars)

    metrics_df = pd.DataFrame(metrics_rows)
    metrics_path = os.path.join(out_dir, "model_metrics.csv")
    metrics_df.to_csv(metrics_path, index=False)
    print("Saved combined metrics:", metrics_path)
    return metrics_df


def _fit_ols_baseline(y, train_data, holdout_base, k_folds, ols_dir, metrics_rows, x_vars):
    os.makedirs(ols_dir, exist_ok=True)
    ols_full = LinearRegression()
    ols_full.fit(train_data[x_vars], train_data[y])
    holdout = holdout_base.copy()
    holdout["score"] = ols_full.predict(holdout[x_vars])
    wauc = weighted_auc(holdout[y].values, holdout["score"].values)

    oof = train_data.copy().reset_index(drop=True)
    oof["score_oof"] = np.nan
    for fold in np.unique(k_folds):
        tr = train_data[k_folds != fold].reset_index(drop=True)
        te_idx = np.where(k_folds == fold)[0]
        te = train_data.iloc[te_idx].reset_index(drop=True)
        m = LinearRegression().fit(tr[x_vars], tr[y])
        oof.loc[te_idx, "score_oof"] = m.predict(te[x_vars])

    cal = LogisticRegression(solver="lbfgs", max_iter=2000)
    cal.fit(oof[["score_oof"]].values, (oof[y].values > 0).astype(int),
            sample_weight=np.clip(oof[y].values.astype(float), 1, None))
    holdout["prob"] = cal.predict_proba(holdout[["score"]].values)[:, 1]
    holdout["y_bin"] = (holdout[y] > 0).astype(int)
    holdout["sw"] = np.clip(holdout[y].astype(float), 1, None)

    out_path = os.path.join(ols_dir, f"{y}_ols_holdout_predictions.csv")
    holdout[["pin", y, "y_bin", "score", "prob", "sw"]].to_csv(
        out_path, index=False)
    metrics_rows.append({
        "model": "ols_regressor", "y": y, "best_params": "",
        "holdout_weighted_auc": wauc, "holdout_top1000_weight": np.nan,
        "holdout_n": int(len(holdout)),
        "holdout_positive_rate": float((holdout[y] > 0).mean()),
        "pred_path": out_path,
    })