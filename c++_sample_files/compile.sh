#!/bin/bash
set -e  # stop if any command fails

# Step 1: Create build folder if not exists
mkdir -p build
cd build

# Step 2: Run CMake with pybind11 support
cmake .. -Dpybind11_DIR=$(python3 -m pybind11 --cmakedir) \
    -DCMAKE_C_COMPILER=gcc \
    -DCMAKE_CXX_COMPILER=g++

# Step 3: Build project
make

# Step 4: Go back to root
cd ..

# Step 5: Rename the compiled .so file to a consistent name
SOFILE=$(ls build/student_agent_module*.so | head -n 1)
if [ -f "$SOFILE" ]; then
    cp "$SOFILE" build/student_agent_module.so
    echo "Compiled and renamed: $SOFILE → build/student_agent_module.so"
else
    echo "Error: Could not find compiled .so file."
    exit 1
fi