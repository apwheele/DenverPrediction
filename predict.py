import pandas as pd
from src.prediction import run

##############################################################################
# MODIFY THESE VALUES AS NEEDED
##############################################################################

entity_df = pd.concat(
    [pd.read_csv("./data/entity_data_2019to2022.csv"),
    pd.read_csv("./data/entity_data_2022topres.csv")],
    ignore_index=True
)

# incident data 
incident_df = pd.read_csv("./data/incident_data_2019topres.csv")


##############################################################################

# Predict:
predictions = run(entity_df, incident_df, train=False)