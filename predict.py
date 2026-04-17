from src.prediction import score_and_flag

##############################################################################
# MODIFY THESE VALUES AS NEEDED
##############################################################################

ENTITY_FILES = [
    "./data/entity_data_2019to2022.csv",
    "./data/entity_data_2022topres.csv",
]

# incident data location/name
INCIDENT_FILE = "./data/incident_data_2019topres.csv"

# DVs of interest, include/exclude variables based on preferences
DV_NAMES = [
    "property_vicoffy",
    "burglary_vicoffy",
    "mvtheft_vicoffy",
    "theft_vicoffy",
]

# set True when predicting on new data
REBUILD = False

# Pick N individuals or proportion of individuals to flag per DV
# set ONE of these to a value, leave the other None
TOP_N    = 100      # top N persons
TOP_PROP = None     # top proportion (e.g. 0.05 for top 5%)

##############################################################################
score_and_flag(
    entity_files=ENTITY_FILES,
    incident_file=INCIDENT_FILE,
    dv_names=DV_NAMES,
    top_n=TOP_N,
    top_prop=TOP_PROP,
    rebuild=REBUILD,
)