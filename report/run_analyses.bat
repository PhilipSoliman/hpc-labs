echo off
@REM Move to project root
cd %~dp0\..
@REM Runnning analyses
echo Intro assignments...
py "intro\analysis.py" --show-output= False
echo Assignment 1...
py "assignment_1\analysis.py" --show-output= False
echo Assignment 2...
py "assignment_2\analysis.py" --show-output= False
echo Assignment 3...
py "assignment_3\analysis.py" --show-output= False
echo Done!

@REM @REM Move back to directory to which batch file was called from
@REM cd %CD%