@echo off
REM Run script for Gargantua CUDA Kerr Black Hole Raytracer

cd /d "%~dp0"

if not exist "gargantua_cuda.exe" (
    echo gargantua_cuda.exe not found.
    echo Build first with build_cuda.bat
    exit /b 1
)

echo === Running Gargantua CUDA Kerr Raytracer ===
.\gargantua_cuda.exe
if %errorlevel% neq 0 (
    echo Run failed!
    exit /b 1
)

echo.
echo Done.
echo Output: gargantua.ppm
