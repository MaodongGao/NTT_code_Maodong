rem @echo off

SET PYTHON="C:\Program Files (x86)\Microsoft Visual Studio\Shared\Python36_64"

SET PATH=%PATH%;%PYTHON%;
call %USERPROFILE%\AppData\Local\Continuum\anaconda3\Scripts\activate.bat %USERPROFILE%\AppData\Local\Continuum\anaconda3

cd /d %~dp0

python SLMDLL2.py %*

pause
