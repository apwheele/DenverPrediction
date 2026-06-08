import os
import pickle
import pandas as pd
import warnings
import json
from src import models
from src.train import train_all, DV_SPECS
from src.hypertune_runner import run_hypertune
from src.prep import get_data, load_train_holdout

warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)



def run(entity_df, incident_df,
        dv_names=("property_vicoffy", "burglary_vicoffy",
                "mvtheft_vicoffy", "theft_vicoffy"),
        train=False,
        forward=False,
        model_dir="./output/final_model",
        hypertune=False,
        include_ols=False,
        dv_specs=None,
        k_folds=5,
        split="pin",
        n_trials=60,
        tuning_dir="./output"):
    """
    Train and/or score the property-crime prediction models.
    Parameters
    ----------
    entity_df, incident_df : DataFrame
            Person-level and incident-level input
    dv_names : str
            Outcome columns to score; can subset to select DVs
    train : boolean
             If True, retrain on the input data; if False, load cached models
    forward : boolean
            If True, score future (predict-only) data;
            if False, score the holdout period as in model building
    model_dir : str
        Directory holding (or receiving) per-DV model.pkl / calibrator.pkl.
    hypertune : boolean
        If True (with train=True), run the Optuna search and write best specs as JSON.
    include_ols : boolean
        If True, also fit OLS baselines for comparison.
    dv_specs : dict
        Defaults is best specs from dv_specs.json, but can be manually defined
    k_folds : int
        Number of k-folds for tuning/calibration.
    split : str
        Grouping column so a group is never split across folds, this will (almost) always be pin
    n_trials : int
        Trials per model family when hypertune=True
    tuning_dir : str
        Directory where tuning CSVs and best-spec JSON files are written

    Returns
    -------
    DataFrame
        score_data with score_{dv} / prob_{dv} columns, joined to original pin
    """
    if dv_specs is None:
        dv_specs = DV_SPECS
    if train:
        train_data, holdout_data, x_vars, lookup = load_train_holdout(
            entity_df, incident_df)
        if hypertune:
            fold_ids = models.kfold_split(train_data, k_folds, split=split)
            best_specs = {}
            for y in dv_specs:
                _, best_specs[y] = run_hypertune(y, train_data, x_vars, fold_ids, n_trials=n_trials,
                                out_csv=os.path.join(tuning_dir, f"{y}_tuning_results.csv"),
                                out_json=os.path.join(tuning_dir, f"{y}_best_spec.json"))
            with open(os.path.join(tuning_dir, "dv_specs.json"), "w") as f:
                json.dump(best_specs, f, indent=2)
            dv_specs = best_specs
        train_all(train_data, holdout_data, x_vars, model_dir, k=k_folds,
                split=split, dv_specs=dv_specs, include_ols=include_ols)
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