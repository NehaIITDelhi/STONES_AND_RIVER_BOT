#!/bin/bash
set -e  # stop if any command fails

echo "--- Cleaning previous build artifacts ---"
rm -rf build  # <--- CRITICAL ADDITION FOR CRASH PREVENTION

echo "--- Preparing build directory ---"
mkdir -p build
cd build

echo "--- Running CMake ---"
# We let CMake detect the best compiler (Clang on Mac, GCC on Linux) 
# unless you specifically need GCC.
cmake .. -Dpybind11_DIR=$(python3 -m pybind11 --cmakedir)

echo "--- Building both agents ---"
make

cd ..
echo "--- Compile finished! ---"