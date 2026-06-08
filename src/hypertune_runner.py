import optuna
import json
import pandas as pd
from sklearn.linear_model import LinearRegression
from src import models
from src.hypertune import cv_eval, objective_cat, objective_lgb


def run_hypertune(y, train_data, x_vars, k_folds, n_trials=60, out_csv=None, out_json=None):
    """
    Tune models for one outcome and report results

    Parameters
    ----------

    y: str
        Outcome of interest
    train_data: DataFrame
        Training dataset
    x_vars: list of str
        Predictor columns
    k_folds: per-row fold assignemtns from models.kfold_split
    n_trials: int
        Trials per model family
    out_csv: str
        if not None, write results here
    out_json: str
        if not None, write the best specs for each y here
    
    Returns
    -------
    Full results table and best spec
    """
    res = {}

    # OLS baseline
    rm_ols = models.Mod(ide_vars=x_vars, y=y, bin_y=False, mod=LinearRegression())
    score, wauc, pei, top1k = cv_eval(rm_ols, data=train_data, ki=k_folds, y_name=y)
    res["ols"] = {"value": score, "params": {}, "wauc": wauc, "pei": pei, "top1k": top1k}

    for name, obj_factory in [("cat", objective_cat),
                            ("lgb", objective_lgb)]:
        study = optuna.create_study(direction="maximize")
        study.optimize(obj_factory(x_vars, y, train_data, k_folds), n_trials=n_trials)
        t = study.best_trial
        res[name] = {
            "value": t.value,
            "params": t.params,
            "wauc": t.user_attrs.get("WeightedAUC"),
            "pei": t.user_attrs.get("PEI"),
            "top1k": t.user_attrs.get("Top1000_Weight"),
        }

    print(f"\n\nTRIAL RESULTS for {y}\n")
    for m, t in res.items():
        print(f"{m}: score={t['value']} wauc={t['wauc']} "
            f"pei={t['pei']} top1k={t['top1k']}")
        print("  params:", t["params"])

    rows = [{"y": y, "model": m, "score": t["value"],
            "weighted_auc": t["wauc"], "pei": t["pei"],
            "top1000_weight": t["top1k"], "params": str(t["params"])}
            for m, t in res.items()]
    df = pd.DataFrame(rows)
    if out_csv:
        df.to_csv(out_csv, index=False)
        print(f"Saved tuning results to {out_csv}")
    tune = {m: t for m, t in res.items() if m != "ols"}
    best_name = max(tune, key=lambda m: tune[m]["value"])
    best_spec = {"model_type": best_name,
                "best_params": res[best_name]["params"]}
    if out_json:
        with open(out_json, "w") as f:
            json.dump(best_spec, f, indent=2)
        print(f"Saved best spec for {y} to {out_json}")         
    return df, best_spec