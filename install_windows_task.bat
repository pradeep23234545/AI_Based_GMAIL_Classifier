@echo off
REM ============================================================
REM install_windows_task.bat
REM ------------------------------------------------------------
REM Registers a Windows Scheduled Task that runs the Gmail AI
REM Agent automatically every N minutes - no window, no exe,
REM nothing you need to keep open. Your PC just needs to be on
REM and you logged in for it to fire (this uses the simple
REM "run only when logged on" mode, so no password needs to be
REM stored).
REM
REM Run this once. To change how often it runs, edit INTERVAL
REM below and run this file again (it replaces the old task).
REM ============================================================

setlocal
set TASKNAME=GmailAIAgent
set SCRIPT_DIR=%~dp0
set INTERVAL=15

echo ============================================================
echo  Gmail AI Agent - Windows Scheduled Task installer
echo ============================================================
echo This will register a task named "%TASKNAME%" that silently
echo runs the agent every %INTERVAL% minutes while you are logged in.
echo.
echo IMPORTANT: the agent starts in DRY-RUN mode (it only logs what
echo it would do). Once you've checked the log and you're happy,
echo enable real changes by running:
echo     python "%SCRIPT_DIR%src\agent.py" --once --enable
echo.
set /p CONFIRM=Continue installing the scheduled task? (Y/N):
if /i not "%CONFIRM%"=="Y" goto :eof

where pythonw >nul 2>nul
if %errorlevel%==0 (
    set PYTHON_EXE=pythonw
) else (
    set PYTHON_EXE=python
)

schtasks /create /tn "%TASKNAME%" ^
    /tr "\"%PYTHON_EXE%\" \"%SCRIPT_DIR%src\agent.py\" --once" ^
    /sc minute /mo %INTERVAL% /f

if %errorlevel%==0 (
    echo.
    echo SUCCESS: "%TASKNAME%" is now scheduled to run every %INTERVAL% minutes.
    echo Log file: check the "Open Folder" location shown by the app, under logs\agent.log
    echo.
    echo Useful commands:
    echo   Run it right now:   schtasks /run /tn "%TASKNAME%"
    echo   Check its status:   schtasks /query /tn "%TASKNAME%"
    echo   Remove it:          schtasks /delete /tn "%TASKNAME%" /f
) else (
    echo Something went wrong creating the scheduled task - see the error above.
)
pause