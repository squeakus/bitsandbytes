@echo off
xcopy /Y C:\HLSWinPro\USER.mdb C:\backup

For /f "tokens=1-3 delims=/ " %%a in ('date /t') do 

(set mydate=%%c-%%b-%%a)
For /f "tokens=1-3 delims=/:/ " %%a in ('time /t') 

do (set mytime=%%a-%%b)
echo %mydate%_%mytime%
set zipname=%mydate%_%mytime%.zip

"c:\Program Files\7-Zip\7z.exe" a -tzip c:\Users\Peter\Dropbox\backup\%zipname% c:\backup\*.*
