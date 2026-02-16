# Quick Start Script for Windows PowerShell
# Starts both backend and frontend in separate terminals

Write-Host "================================" -ForegroundColor Green
Write-Host "Career Path Recommender Startup" -ForegroundColor Green
Write-Host "================================" -ForegroundColor Green
Write-Host ""

# Get project root directory
$projectRoot = Split-Path $PSScriptRoot -Parent

# Terminal 1: Backend
Write-Host "Starting Backend (Terminal 1)..." -ForegroundColor Cyan
Write-Host "Command: cd backend && .\venv\Scripts\activate && python -m uvicorn src.app.api:app --port 8000" -ForegroundColor Gray

Start-Process powershell {
    cd "$($projectRoot)\backend"
    if (!(Test-Path "venv")) {
        Write-Host "Virtual environment not found. Creating..." -ForegroundColor Yellow
        python -m venv venv
        Write-Host "Installing dependencies..." -ForegroundColor Yellow
        .\venv\Scripts\activate
        pip install -r requirements.txt
        Write-Host "Dependencies installed." -ForegroundColor Green
    }
    .\venv\Scripts\activate
    Write-Host "Backend starting on http://localhost:8000" -ForegroundColor Green
    Write-Host "API docs: http://localhost:8000/docs" -ForegroundColor Green
    python -m uvicorn src.app.api:app --host 0.0.0.0 --port 8000 --reload
}

Start-Sleep -Seconds 2

# Terminal 2: Frontend
Write-Host "Starting Frontend (Terminal 2)..." -ForegroundColor Cyan
Write-Host "Command: cd frontend && npm run dev" -ForegroundColor Gray

Start-Process powershell {
    cd "$($projectRoot)\frontend"
    if (!(Test-Path "node_modules")) {
        Write-Host "Dependencies not installed. Installing..." -ForegroundColor Yellow
        npm install
        Write-Host "Dependencies installed." -ForegroundColor Green
    }
    Write-Host "Frontend starting on http://localhost:8080" -ForegroundColor Green
    npm run dev
}

Write-Host ""
Write-Host "Both services starting..." -ForegroundColor Green
Write-Host ""
Write-Host "Frontend: http://localhost:8080" -ForegroundColor Blue
Write-Host "Backend: http://localhost:8000" -ForegroundColor Blue
Write-Host "API Docs: http://localhost:8000/docs" -ForegroundColor Blue
Write-Host ""
Write-Host "Press Ctrl+C in either terminal to stop that service" -ForegroundColor Yellow
