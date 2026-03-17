import os
import pickle
import pandas as pd


##############################################################################
# DEFINITIONS FOR FILES TO WORK WITH
##############################################################################

# this defines the locations for relevant items
### for test runs, these are locations of actual data
# ENTITY_FILES = [
#       "./data/entity_data_2019to2022.csv",
#       "./data/entity_data_2022topres.csv"
#   ]
# INCIDENT_FILE = "./data/incident_data_2019topres.csv"

ENTITY_FILES = [                                     # entity file(s)
    "./data/entity_data_file_1.csv",
    "./data/entity_data_file_2.csv"
]
INCIDENT_FILE = "./data/incident_data_file.csv"           # incident file(s)

FEATURES_PATH = "./data/prepped_data.csv"            # processed data if applicable, if not, location you want processed data to be saved
LOOKUP_PATH = "./data/LookupTable.csv.zip"           # de-identified PIN back to PII key
MODEL_DIR = "./output/final_model"                   # trained model
OUTPUT_DIR = "./output/new_predictions/" # place you want the output


# defines the prop or count of top persons of interest to identify
TOP_PROP = None                     # Define proportion of interest, or set to None if you want a top N
TOP_N = 100                         # Define number of interest, or set to None if you want a top proportion (includes ties, so output may be more than defined)

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

os.makedirs(OUTPUT_DIR, exist_ok=True)  # create output location if it doesnt already exist

##############################################################################
# SCORING FOR TOP PROP IDENTIFICATION FOR EACH DV
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
# FLAGGING TOP PROP
##############################################################################

flag_cols = []

for dv in DV_NAMES:
    prob_col = f"prob_{dv}"
    if f"prob_{dv}" not in score_data.columns:
        continue
    flag_col = f"flag_{dv}"
    if TOP_N is None:
        threshold = score_data[prob_col].quantile(1 - TOP_PROP)
    else:
        threshold = score_data[prob_col].nlargest(TOP_N).min()
        
    score_data[flag_col] = score_data[prob_col] >= threshold
    flag_cols.append(flag_col)


##############################################################################
# JOIN PIN WITH PII FOR PRACTICAL USE
##############################################################################

result = score_data.merge(
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

##############################################################################
# EXPORT ONE CSV PER DV
##############################################################################

for dv in DV_NAMES:
    flag_col = f"flag_{dv}"
    prob_col = f"prob_{dv}"
    if flag_col not in result.columns:
        continue

    flagged = result[result[flag_col]].copy()

    # sort by probability
    flagged = flagged.sort_values(prob_col, ascending=False)

    # put PIN first in outputs
    cols = flagged.columns.tolist()
    cols.insert(0, cols.pop(cols.index("original_pin")))
    flagged = flagged[cols]

    out_path = os.path.join(OUTPUT_DIR, f"{dv}_flagged.csv")
    flagged.to_csv(out_path, index=False)
    print(f"Saved {out_path} - {len(flagged)} persons flagged")