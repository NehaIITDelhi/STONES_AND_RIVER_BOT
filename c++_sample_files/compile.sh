#!/bin/bash
set -e  # stop if any command fails

echo "--- Preparing build directory ---"
mkdir -p build
cd build

echo "--- Running CMake ---"
cmake .. -Dpybind11_DIR=$(python3 -m pybind11 --cmakedir) \
    -DCMAKE_C_COMPILER=gcc \
    -DCMAKE_CXX_COMPILER=g++

echo "--- Building both agents ---"
make

cd ..
echo "--- Compile finished! ---"
echo "Created: build/student_agent_module.so (IMPROVED agent)"
echo "Created: build/baseline_agent.so (BASELINE agent)"