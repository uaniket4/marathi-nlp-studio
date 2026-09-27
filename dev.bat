@echo off
REM Marathi NLP Studio - start backend + frontend with one command (Windows cmd).
REM Usage: double-click dev.bat, or run  dev.bat  from a terminal.
REM Opens the backend in its own window; runs the frontend in this window.
REM Close the backend window (or Ctrl+C in it) to stop the backend.

setlocal
set "ROOT=%~dp0"

REM Install frontend deps on first run.
if not exist "%ROOT%frontend\node_modules" (
  echo Installing frontend dependencies ^(first run^)...
  call npm --prefix "%ROOT%frontend" install
)

echo Starting backend on http://localhost:8000 ...
start "Marathi NLP Studio - backend" cmd /k "cd /d "%ROOT%backend" && set PYTHONIOENCODING=utf-8 && py -3.12 -m uvicorn app.main:app --port 8000"

echo Starting frontend on http://localhost:5173 ...
cd /d "%ROOT%frontend"
call npm run dev -- --port 5173

endlocal
