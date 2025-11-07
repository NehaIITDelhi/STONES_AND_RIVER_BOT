cmake_minimum_required(VERSION 3.12)
project(student_agents)

set(CMAKE_CXX_STANDARD 17)

find_package(pybind11 REQUIRED)

# Compiles your NEW code (improved) into "student_agent_module"
# (This matches student_agent_cpp.py)
pybind11_add_module(student_agent_module student_agent_improved.cpp)

# Compiles your OLD code (baseline) into "baseline_agent"
# (This matches baseline_agent_cpp.py)
pybind11_add_module(baseline_agent student_agent.cpp)