echo off

@REM Save caller's directory 
set currentDirectory=%CD%

echo (Re-)Generating all report figures and tables...

@REM Move to root folder hpc-labs/
cd %~dp0 && cd ..

@REM Move to activate virtual env script location
cd .venv/Scripts

@REM activate virtual env, move to run analyses, move back to caller directory
@activate && cd ..\..\report && run_analyses.bat && cd %currentDirectory%