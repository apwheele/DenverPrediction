import optuna
import pandas as pd
from sklearn.linear_model import LinearRegression
from src import models
from src.hypertune import cv_eval, objective_cat, objective_lgb, objective_xgb


def run_hypertune(y, train_data, x_vars, k_folds, n_trials=60, out_csv=None):
    res = {}

    # OLS baseline
    rm_ols = models.Mod(ide_vars=x_vars, y=y, bin_y=False, mod=LinearRegression())
    score, wauc, pei, top1k = cv_eval(rm_ols, data=train_data, ki=k_folds, y_name=y)
    res["ols"] = {"value": score, "params": {}, "wauc": wauc, "pei": pei, "top1k": top1k}

    for name, obj_factory in [("cat", objective_cat),
                            ("lgb", objective_lgb),
                            ("xgb", objective_xgb)]:
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
    return df