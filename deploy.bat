@echo off
setlocal

:: === CONFIG ===
set ZIP_NAME=investor-gpt-app-%date:~10,4%%date:~4,2%%date:~7,2%.zip
set VERSION_TAG=v%date:~10,4%.%date:~4,2%.%date:~7,2%
set SOURCE_APP=D:\workspace\hfpsource\HFP\OPENAI\pia\app.py
set TARGET_APP=D:\investor-gpt-app\app.py

:: === COPY app.py from dev source ===
copy /Y "%SOURCE_APP%" "%TARGET_APP%"

:: === CLEAN OLD ZIP ===
if exist %ZIP_NAME% del %ZIP_NAME%

:: === ZIP (exclude .git and __pycache__) ===
powershell -Command "Compress-Archive -Path * -DestinationPath %ZIP_NAME% -CompressionLevel Optimal -Force -Exclude '.git', '__pycache__', '*.pyc', '*.pyo'"

:: === GIT ADD / COMMIT / TAG / PUSH ===
git add .
git commit -m "Deploy %DATE%: updates and cleanup"
git tag %VERSION_TAG%
git push
git push origin %VERSION_TAG%

echo.
echo ✅ Deployment complete: %ZIP_NAME%
pause
