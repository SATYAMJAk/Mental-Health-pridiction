# Student Mental Health Prediction - Project Runner
Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host "  Student Mental Health Prediction Platform (ML + MERN)   " -ForegroundColor Green
Write-Host "==========================================================" -ForegroundColor Cyan

# 1. Start Python FastAPI ML Microservice on port 8001
Write-Host "`n[1/3] Starting Python ML FastAPI Service (Port 8001)..." -ForegroundColor Yellow
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd '$PSScriptRoot\ml_service'; python -m uvicorn main:app --host 127.0.0.1 --port 8001 --reload"

# 2. Start Express Node.js Backend on port 5055
Write-Host "[2/3] Starting Express Node.js Backend (Port 5055)..." -ForegroundColor Yellow
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd '$PSScriptRoot\backend'; node server.js"

# 3. Start Vite React Frontend on port 5173
Write-Host "[3/3] Starting React Frontend (Port 5173)..." -ForegroundColor Yellow
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd '$PSScriptRoot\frontend'; npm run dev"

Write-Host "`nAll 3 services successfully launched!" -ForegroundColor Green
Write-Host "Frontend Portal: http://localhost:5173" -ForegroundColor White
Write-Host "Express API:     http://localhost:5055" -ForegroundColor White
Write-Host "ML FastApi Docs: http://localhost:8001/docs" -ForegroundColor White
