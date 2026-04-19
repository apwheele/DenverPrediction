::: Automates building the fit_discussion report.
::: Prereqs (run once, in your own Python env):
:::   pip install -r requirements.txt
:::   Install Quarto: https://quarto.org/docs/get-started/
::: Then from an activated Python env:
:::   reports\render_report.bat
@echo on

:: Move to the reports folder so relative paths in the .qmd resolve correctly
cd /d "%~dp0"

:: Activate venv
call "..\venv\Scripts\activate.bat"

:: Render qmd -> docx, then docx -> pdf
call quarto render fit_discussion.qmd
call docx2pdf fit_discussion.docx