echo off
@REM Move to project root
cd %~dp0\..

@REM Make sure python utils is added to search path
@REM export PYTHONPATH=$PYTHONPATH:${workspaceFolder}/python_utils

@REM Runnning analyses
echo Intro assignments...
cd intro && py analysis.py --show-output= False && cd ..
echo Assignment 1...
cd assignment_1 && py analysis.py --show-output= False && cd ..
echo Assignment 2...
cd assignment_2 && py analysis.py --show-output= False && cd ..
echo Assignment 3...
cd assignment_2 && py analysis.py --show-output= False && cd ..
echo Done!

@REM @REM Move back to directory to which batch file was called from
@REM cd %CD%