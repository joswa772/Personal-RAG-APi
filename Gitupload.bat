@echo off
cd /d "%~dp0"

echo 📁 In project folder: %cd%
echo.

REM Ask for commit message
set /p commitMsg=📝 Enter commit message: 

echo 🔄 Adding all changes...
git add .

echo ✅ Committing with message: "%commitMsg%"
git commit -m "%commitMsg%"

echo 🚀 Pushing to GitHub (main branch)...
git push origin main

echo ✅ All done!
pause
