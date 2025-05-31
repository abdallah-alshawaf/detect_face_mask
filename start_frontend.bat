@echo off
echo Starting Face Mask Detection Frontend...
echo.

cd frontend

echo Installing dependencies...
call npm install

echo.
echo Starting React development server...
echo Frontend will be available at: http://localhost:3000
echo.

call npm start

pause 