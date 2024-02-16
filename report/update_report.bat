echo off

@REM Save caller's directory 
set currentDirectory=%CD%
set root=%~dp0..

echo (Re-)Generating all report figures and tables...

@REM @REM Move to root folder hpc-labs/
@REM cd %root%

@REM @REM Make sure python utils is added to search path
@REM set PYTHONPATH=%PYTHONPATH%;%root%\python_utils\python_utils

echo Move to activate virtual env script location at "%root%/.venv/Scripts"
cd %root% && cd .venv\Scripts && call activate && cd %root%\report

echo activate virtual env, move to run analyses, move back to caller directory
run_analyses.bat && cd %currentDirectory%