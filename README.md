# A Randomized Experiment to Target High Risk Gun Offenders in Denver, Colorado

This is the repository for the project, *A Randomized Experiment to Target High Risk Gun Offenders in Denver, Colorado*. Joint project with Andrew Wheeler & Captain Jacob Herrara, Denver Police Department

This work is supported by the [Arnold Foundation](https://www.arnoldventures.org/newsroom/grants-announcement-arnold-ventures-criminal-justice-research-grants-in-first-half-of-2025-demonstrate-its-ongoing-commitment-to-evidence-based-solutions)

For any questions, contact [Andrew Wheeler](andrew.wheeler@crimede-coder.com)

Andrew Wheeler
[Crime De-Coder LLC](https://crimede-coder.com/)

## Replicating the environment

I use a standard Python venv and quarto. So to replicate the python environment see the `requirements.txt` file in the root. In addition to this, I use [quarto](https://quarto.org/), which that will also need to be installed to render the `.qmd` files.

## Data Released

The data is sensitive, so only the aggregated data is uploaded. The original unique identifiers in Denver's data are replaced with a random numeric value.

## Usage

The file needed to operate the model in this repository is: 

    predict.py

## Data

All raw input data used to make the model lives in `./data/`. The pipeline expects three kinds of files:

1. **Entity files** — person-level records. Must include at minimum: `pin`, `occ_date`, `role`, `ucr`, `incident_no`.
2. **Incident file** — incident-level records. Must include: `incident_no`, `x_coordinate`, `y_coordinate`, `ibr_code`, `offense_desc`, `ucr`.
3. **Lookup table** — `./data/LookupTable.csv.zip`, mapping the de-identified `pin` used in modeling back to the original PII identifier. This is generated **automatically** by the data prep on first run; you don't need to create it yourself.

The outcome (DV) columns (`property_vicoffy`, `theft_vicoffy`, `burglary_vicoffy`, `mvtheft_vicoffy`) are created from UCR codes and role categories in `src/prep.py`, so their names stay stable across datasets. If your incoming data has roles or UCR codes that aren't mapped, prep will print a warning, and then proceed.

### Using the model with new data

1. Point the `predict.py` file at entity and incident level data. Pandas dfs are accepted as well as csvs
2. Modify the function's arguments according to preferences, below are the default arguments:
                
                run(entity_df, incident_df,
                        dv_names=("property_vicoffy", "burglary_vicoffy",
                                "mvtheft_vicoffy", "theft_vicoffy"),
                        train=False,
                        forward=False,
                        model_dir="./output/final_model",
                        hypertune=False,
                        include_ols=False)

    - DV names are created from the data prep embedded in the function, the argument can be subsetted to model only select outcomes
    - When train=False, the model is based on its initial construction. Users can set this arg to True to retrain on new input data
    - The forward=False argument toggles whether the model should predict on holdout data or the future. Default is False so users can investigate model validity against actual values of a given variable
    - When train=False, this argument defines where the specs for the model being implemented live. The default location are the models trained as a part of the model's initial construction. If train=True and the model_dir is left with its default, the newly trained models will overwrite the older model cache and become the new default
    - The hypertune=False argument avoids the ~12 hour model parameter invesitgation. These parameters are already defined from initial hypertuning.
    - include_ols=False omits simple OLS models for comparison to more complex MLM models.

                        
3. Run:
    
    python predict.py
    
Newly trained model specs land in `./output/final_model/<dv>/`.
Relevant outputs will be stored in a `predictions` object with the `prob_{dv}_vicoffy` column defining each PIN's probability of a given outcome.

## Reports

**Before running, edit the activation line in `render_report.bat`** to point at your own environment. The committed script activates a local `venv`:

```bat
call "..\venv\Scripts\activate.bat"
```

Replace with whatever activates your environment, e.g.:

- Conda: `call C:\path\to\conda\Scripts\activate.bat` then `call conda activate <env-name>`
- A venv elsewhere: `call C:\path\to\your\venv\Scripts\activate.bat`

Quarto reports live in `reports/`. To render `fit_discussion.qmd` to Word and PDF on a Windows machine with a venv, quarto, and Microsoft Word installed:

    reports\render_report.bat

The script renders the `.qmd` to `.docx`, then converts the `.docx` to `.pdf`. The report will generate model fit and prediction accuracy statistics against and OLS baseline.

### Requirements

    - [quarto](https://quarto.org/) installed and on your PATH
    - Microsoft Word installed (needed by `docx2pdf`)
    - `docx2pdf` installed in your venv (`pip install docx2pdf`)
    - Trained model outputs in `./output/final_model/` (run `train_model.py` first if the folder is empty)