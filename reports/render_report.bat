::: Automates building the fit_discussion report
::: Run from a local machine with Word installed
@echo on

:: Move to the reports folder so relative paths in the .qmd resolve correctly
cd /d "%~dp0"

:: Activate conda
set conda_act=D:\Python\Scripts\activate.bat
call %conda_act%
call conda activate crimscrape

:: Render qmd -> docx, then docx -> pdf
call quarto render fit_discussion.qmd
call docx2pdf fit_discussion.docx