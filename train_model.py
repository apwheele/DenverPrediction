from src.prep import load_train_holdout
from src.train import train_all, DV_SPECS
from src.hypertune_runner import run_hypertune
from src import models

##############################################################################
# MODIFY THESE VALUES AS NEEDED
##############################################################################

# set to True when training on new data
REBUILD = False

# set to True to re-run hyperparameter tuning (will take ~12 hours)
HYPERTUNE = False 

# entity data location/name
ENTITY_FILES = [
    "./data/entity_data_2019to2022.csv",
    "./data/entity_data_2022topres.csv",
]

# incident data location/name
INCIDENT_FILE = "./data/incident_data_2019topres.csv"


train_data, holdout_data, x_vars = load_train_holdout(
    entity_files=ENTITY_FILES,
    incident_file=INCIDENT_FILE,
    rebuild=REBUILD,
)

if HYPERTUNE:
    k_folds = models.kfold_split(train_data, 5, split="pin")
    for y in DV_SPECS:
        run_hypertune(y, train_data, x_vars, k_folds, out_csv=f"./output/{y}_tuning_results.csv")
else:
    train_all(train_data, holdout_data, x_vars, "./output/final_model")