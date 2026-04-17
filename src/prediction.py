import os
import pickle
import pandas as pd
import warnings
warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)

def _load_features(entity_files, incident_file, features_path):
    if features_path and os.path.exists(features_path):
        print(f"Found pre-processed features at {features_path}")
        return pd.read_csv(features_path)
    print("No pre-processed data, running raw data prep...")
    from src.prep import get_data
    _, score_data = get_data(entity_files=entity_files, incident_file=incident_file)
    if features_path:
        score_data.to_csv(features_path, index=False)
        print(f"Prepped features saved to {features_path}")
    return score_data


def score_and_flag(entity_files, incident_file,
                    dv_names=("property_vicoffy", "burglary_vicoffy",
                               "mvtheft_vicoffy", "theft_vicoffy"),
                    model_dir="./output/final_model",
                    output_dir="./output/new_predictions/",
                    lookup_path="./data/LookupTable.csv.zip",
                    features_path="./data/prepped_data.csv",
                    top_n=100, top_prop=None, rebuild=False):
    if top_n is None and top_prop is None:
        raise ValueError("Specify exactly one of top_n or top_prop.")

    os.makedirs(output_dir, exist_ok=True)
    if rebuild and features_path and os.path.exists(features_path):
        os.remove(features_path)
    score_data = _load_features(entity_files, incident_file, features_path)
    lookup = pd.read_csv(lookup_path)

    # score
    for dv in dv_names:
        model_path = os.path.join(model_dir, dv, "model.pkl")
        cal_path = os.path.join(model_dir, dv, "calibrator.pkl")
        if not os.path.exists(model_path):
            print(f"[SKIP] No model for {dv} — run train_model.py first")
            continue
        with open(model_path, "rb") as f:
            rm = pickle.load(f)
        with open(cal_path, "rb") as f:
            cal = pickle.load(f)
        score_data[f"score_{dv}"] = rm.predict(score_data).values
        score_data[f"prob_{dv}"] = cal.predict_proba(
            score_data[[f"score_{dv}"]].values)[:, 1]
        print(f"[DONE] scored {dv}")

    # flag
    for dv in dv_names:
        prob_col = f"prob_{dv}"
        if prob_col not in score_data.columns:
            continue
        if top_n is None:
            threshold = score_data[prob_col].quantile(1 - top_prop)
        else:
            threshold = score_data[prob_col].nlargest(top_n).min()
        score_data[f"flag_{dv}"] = score_data[prob_col] >= threshold

    # join pii
    result = score_data.merge(lookup, left_on="pin", right_on="pin_new", how="left")
    result = result.rename(columns={"pin_x": "pin_deidentified", "pin_y": "original_pin"})

    # export one CSV per outcome
    for dv in dv_names:
        flag_col = f"flag_{dv}"
        prob_col = f"prob_{dv}"
        if flag_col not in result.columns:
            continue
        flagged = result[result[flag_col]].sort_values(prob_col, ascending=False)
        if "original_pin" in flagged.columns:
            cols = flagged.columns.tolist()
            cols.insert(0, cols.pop(cols.index("original_pin")))
            flagged = flagged[cols]
        out_path = os.path.join(output_dir, f"{dv}_flagged.csv")
        flagged.to_csv(out_path, index=False)
        print(f"Saved {out_path} — {len(flagged)} persons flagged")