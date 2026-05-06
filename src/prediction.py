import os
import pickle
import pandas as pd
import warnings
warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)

from src.prep import get_data, load_train_holdout


def run(entity_df, incident_df,
        dv_names=("property_vicoffy", "burglary_vicoffy",
                "mvtheft_vicoffy", "theft_vicoffy"),
        train=False,
        forward=False,
        model_dir="./output/final_model",
        hypertune=False,
        include_ols=False):
    if train:
        from src.train import train_all, DV_SPECS
        train_data, holdout_data, x_vars, lookup = load_train_holdout(
            entity_df, incident_df)
        if hypertune:
            from src.hypertune_runner import run_hypertune
            from src import models
            k_folds = models.kfold_split(train_data, 5, split="pin")
            for y in DV_SPECS:
                run_hypertune(y, train_data, x_vars, k_folds,
                                out_csv=f"./output/{y}_tuning_results.csv")
        train_all(train_data, holdout_data, x_vars, model_dir,
                include_ols=include_ols)
        if forward:
            score_data, lookup = get_data(entity_df, incident_df, predict_only=True)
        else:
            score_data = holdout_data
    else:
        if forward:
            score_data, lookup = get_data(entity_df, incident_df, predict_only=True)
        else:
            _, score_data, lookup = get_data(entity_df, incident_df)
    for dv in dv_names:
        model_path = os.path.join(model_dir, dv, "model.pkl")
        cal_path = os.path.join(model_dir, dv, "calibrator.pkl")
        with open(model_path, "rb") as f: rm = pickle.load(f)
        with open(cal_path, "rb") as f: cal = pickle.load(f)
        score_data[f"score_{dv}"] = rm.predict(score_data).values
        score_data[f"prob_{dv}"] = cal.predict_proba(
            score_data[[f"score_{dv}"]].values)[:, 1]

    return score_data.merge(lookup.rename(columns={"pin": "original_pin"}), left_on="pin", right_on="pin_new", how="left")