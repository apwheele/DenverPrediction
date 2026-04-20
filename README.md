# A Randomized Experiment to Target High Risk Gun Offenders in Denver, Colorado

This is the repository for the project, *A Randomized Experiment to Target High Risk Gun Offenders in Denver, Colorado*. Joint project with Andrew Wheeler & Captain Jacob Herrara, Denver Police Department

This work is supported by the [Arnold Foundation](https://www.arnoldventures.org/newsroom/grants-announcement-arnold-ventures-criminal-justice-research-grants-in-first-half-of-2025-demonstrate-its-ongoing-commitment-to-evidence-based-solutions)

For any questions, contact [Andrew Wheeler](andrew.wheeler@crimede-coder.com)

Andrew Wheeler
[Crime De-Coder LLC](https://crimede-coder.com/)

## Replicating the environment

I use conda and quarto. So to replicate the python environment see the `requirements.txt` file in the root. In addition to this, I use [quarto](https://quarto.org/), which that will also need to be installed to render the `.qmd` files.

## Data Released

The data is sensitive, so only the aggregated data is uploaded. The original unique identifiers in Denver's data are replaced with a random numeric value.

## Usage

The only two files needed to operate the model in this repository are: 

    train_model.py
    predict.py

## Data

All raw input data lives in `./data/`. The pipeline expects three kinds of files:

1. **Entity files** — person-level records (one or more CSVs). Must include at minimum: `pin`, `occ_date`, `role`, `ucr`, `incident_no`.
2. **Incident file** — incident-level records (one CSV). Must include: `incident_no`, `x_coordinate`, `y_coordinate`, `ibr_code`, `offense_desc`, `ucr`.
3. **Lookup table** — `./data/LookupTable.csv.zip`, mapping the de-identified `pin` used in modeling back to the original PII identifier. This is generated **automatically** by the data prep on first run; you don't need to create it yourself.

The outcome (DV) columns (`property_vicoffy`, `theft_vicoffy`, `burglary_vicoffy`, `mvtheft_vicoffy`) are created from UCR codes and role categories in `src/prep.py`, so their names stay stable across datasets. If your incoming data has roles or UCR codes that aren't mapped, prep will print a warning, and then proceed.

### Training on new data

1. Put your entity and incident level data CSVs in `./data/`.
2. Open `train_model.py` and edit `ENTITY_FILES` and `INCIDENT_FILE` to match your filenames.
3. Set `REBUILD = True` so the prepped data in `./train_data/` get rebuilt from your new raw files.
4. Run:
    
    python train_model.py
    
Trained models land in `./output/final_model/<dv>/`.

To re-run hyperparameter tuning before fitting the final models, set `HYPERTUNE = True` in `train_model.py`. Tuning results are written to `./output/`; copy the best params into `DV_SPECS` in `src/train.py`, then re-run with `HYPERTUNE = False`. Note that the current models have already been hypertuned, and the hypertuning process will take a long time (~12 hours).

### Predicting on new data

1. Put your CSVs in `./data/` (or leave them as-is if you used the data for training)
2. Open `predict.py` and edit `ENTITY_FILES` and `INCIDENT_FILE` to match your filenames.
3. Set `REBUILD = True` so features get rebuilt from your new raw files.
4. Optionally adjust `DV_NAMES`, `TOP_N`, or `TOP_PROP` according to preferences (set exactly one of the last two; leave the other `None`).
5. Run:

    python predict.py

Flagged CSVs (one per DV) land in `./output/new_predictions/`.

## Reports

**Before running, edit the activation line in `render_report.bat`** to point at your own environment. The committed script activates a local `venv`:

```bat
call "..\venv\Scripts\activate.bat"
```

Replace with whatever activates your environment, e.g.:

- Conda: `call C:\path\to\conda\Scripts\activate.bat` then `call conda activate <env-name>`
- A venv elsewhere: `call C:\path\to\your\venv\Scripts\activate.bat`

Quarto reports live in `reports/`. To render `fit_discussion.qmd` to Word and PDF on a Windows machine with conda, quarto, and Microsoft Word installed:

    reports\render_report.bat

The script renders the `.qmd` to `.docx`, then converts the `.docx` to `.pdf`. The report will generate model fit and prediction accuracy statistics against and OLS baseline.

### Requirements

    - [quarto](https://quarto.org/) installed and on your PATH
    - Microsoft Word installed (needed by `docx2pdf`)
    - `docx2pdf` installed in your conda env (`pip install docx2pdf`)
    - Trained model outputs in `./output/final_model/` (run `train_model.py` first if the folder is empty)