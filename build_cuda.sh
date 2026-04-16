#!/bin/bash
# Build script for Gargantua CUDA Kerr Black Hole Raytracer
# Requires: CUDA Toolkit 12.x

echo "=== Building Gargantua CUDA Kerr Raytracer ==="
nvcc -O3 -o gargantua_cuda gargantua_cuda.cu

if [ $? -ne 0 ]; then
    echo "Build failed!"
    exit 1
fi
echo "Build successful: gargantua_cuda"

echo ""
echo "Run with: ./gargantua_cuda"
echo "Output:   gargantua.ppm  (convert with: ffmpeg -i gargantua.ppm gargantua.png)"

# CPU-only fallback (no CUDA required):
# g++ -O3 -fopenmp -x c++ -DCPU_ONLY gargantua_cuda.cu -o gargantua_cpu -lm
