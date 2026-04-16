@echo off
REM Build script for Gargantua CUDA Kerr Black Hole Raytracer
REM Requires: CUDA Toolkit 12.x, Visual Studio 2022

call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvarsall.bat" x64 >nul 2>&1

echo === Building Gargantua CUDA Kerr Raytracer ===
nvcc -O3 -arch=sm_89 -o gargantua_cuda.exe gargantua_cuda.cu
if %errorlevel% neq 0 (
    echo Build failed!
    exit /b 1
)
echo Build successful: gargantua_cuda.exe

echo.
echo Run with: gargantua_cuda.exe
echo Output:   gargantua.ppm  (convert with: ffmpeg -i gargantua.ppm gargantua.png)
