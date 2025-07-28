@echo off
setlocal EnableDelayedExpansion

:: Ask for commit message
set /p commitmsg=📝 Enter commit message: 

echo 🔄 Pulling latest changes...
git pull --rebase origin main

echo 📂 Adding all changes...
git add .

echo 🧠 Committing with message: "!commitmsg!"
git commit -m "!commitmsg!"

echo 🚀 Pushing to GitHub...
git push origin main

echo ✅ All done!
pause
