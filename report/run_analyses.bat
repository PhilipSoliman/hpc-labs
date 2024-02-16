echo off
set root=%~dp0\..
set pyexe=%root%\.venv\Scripts\python.exe
@REM Make sure python utils is added to search path
@REM set PYTHONPATH=%PYTHONPATH%;%root%\python_utils

echo Runnning analyses
cd %root%
echo Intro assignments...
cd intro && "%pyexe%" analysis.py --show-output= False & cd ..
echo Assignment 1...
cd assignment_1 && "%pyexe%" analysis.py --show-output= False & cd ..
echo Assignment 2...
cd assignment_2 && "%pyexe%" analysis.py --show-output= False & cd ..
echo Assignment 3...
cd assignment_2 && "%pyexe%" analysis.py --show-output= False & cd ..
echo Done!

@REM @REM Move back to directory to which batch file was called from
@REM cd %CD%