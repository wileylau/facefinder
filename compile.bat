@echo off
set MODEL_DIR=%USERPROFILE%\.insightface\models\buffalo_s
if not exist "%MODEL_DIR%" (
    echo Model not found at %MODEL_DIR%
    echo Run the app once first to download it, or place buffalo_s files there manually.
    pause
    exit /b 1
)
pyinstaller --noconfirm --onefile --windowed --name "FaceFinder" --add-data "%MODEL_DIR%;buffalo_s" find_gui.py
echo.
echo Done. Executable is in dist\FaceFinder.exe
pause
