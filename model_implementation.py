import os
import pickle
import numpy as np
import pandas as pd


##############################################################################
# DEFINITIONS FOR FILES TO WORK WITH
##############################################################################

# this defines the locations for relevant items
ENTITY_FILES = [                                     # entity file(s)
    "./data/entity_data_1.csv",
    "./data/entity_data_2.csv"
]
INCIDENT_FILE = "./data/incident_data.csv"           # incident file(s)
FEATURES_PATH = "./data/prepped_data.csv"            # processed data if applicable, if not, location you want processed data to be saved
LOOKUP_PATH = "./data/LookupTable.csv.zip"           # de-identified PIN back to PII key
MODEL_DIR = "./output/final_model"                   # trained model
OUTPUT_PATH = "./output/new_predictions/flagged.csv" # place you want the output

# MODIFY THIS AS NEEDED
# defines the pct of top persons of interest to identify
TOP_N_PCT = 0.1

# DVs of interest
DV_NAMES = [
    "property_vicoffy",
    "burglary_vicoffy",
    "mvtheft_vicoffy",
    "theft_vicoffy"
]

##############################################################################
# 
##############################################################################

if os.path.exists(FEATURES_PATH):
    print(f"Found existing pre-processed data at {FEATURES_PATH}")
    score_data = pd.read_csv(FEATURES_PATH)
else:
    print("No pre-processed data, running raw data prep...")
    from src.prep import get_data
    _, score_data = get_data(entity_files=ENTITY_FILES, incident_file=INCIDENT_FILE)
    score_data.to_csv(FEATURES_PATH, index=False)
    print(f"Raw data prepped, saved to {FEATURES_PATH}")

lookup = pd.read_csv(LOOKUP_PATH) # load in PII key

os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)  # create output location if it doesnt already exist

##############################################################################
# SCORING FOR TOP N% IDENTIFICATION FOR EACH DV
##############################################################################

for dv in DV_NAMES:
    model_path = os.path.join(MODEL_DIR, dv, "model.pkl")
    cal_path = os.path.join(MODEL_DIR, dv, "calibrator.pkl")

    if not os.path.exists(model_path):
        print(f"  [SKIP] No saved model found for {dv} — run full_model.py first")
        continue

    with open(model_path, "rb") as f:
        rm = pickle.load(f)
    with open(cal_path, "rb") as f:
        cal = pickle.load(f)

    score_data[f"score_{dv}"] = rm.predict(score_data).values
    score_data[f"prob_{dv}"] = cal.predict_proba(score_data[[f"score_{dv}"]].values)[:, 1]

    print(f"  [DONE] {dv}")


##############################################################################
# FLAGGING TOP N%
##############################################################################

flag_cols = []

for dv in DV_NAMES:
    prob_col = f"prob_{dv}"
    if f"prob_{dv}" not in score_data.columns:
        continue
    flag_col = f"flag_{dv}"
    threshold = score_data[prob_col].quantile(1 - TOP_N_PCT)
    score_data[flag_col] = score_data[prob_col] >= threshold
    flag_cols.append(flag_col)

score_data["flagged_any"] = score_data[flag_cols].any(axis=1)


##############################################################################
# JOIN PIN WITH PII FOR PRACTICAL USE
##############################################################################

result = new_data.merge(
    lookup,
    left_on="pin",      # de-identified PIN in the prediction model data
    right_on="pin_new", # de-identified PIN in the lookup table
    how="left"
)

# rename the columns for ease of interpretation
result = result.rename(columns={
    "pin_x": "pin_deidentified",
    "pin_y": "original_pin"
})

# save out results
flagged = result[result["flagged_any"]].copy()
flagged.to_csv(OUTPUT_PATH, index=False)