#Rum : cmake -S . -B build -DME630_NUM_ASSIGNMENTS=6

cmake_minimum_required(VERSION 3.18)

project(
    ME630Assignments
    VERSION 1.0
    DESCRIPTION "ME630 Python/Jupyter Assignments"
    LANGUAGES NONE
)

# ==============================================================================
# Configuration
# ==============================================================================

# Number of assignment directories to create.
#
# Override from the command line with:
#   cmake -S . -B build -DME630_NUM_ASSIGNMENTS=5
#
set(
    ME630_NUM_ASSIGNMENTS
    3
    CACHE STRING
    "Number of ME630 assignment directories to create"
)

# Virtual environment name
set(ME630_VENV_NAME "ME630-venv")

# Place the virtual environment in the repository root.
set(ME630_VENV_DIR "${CMAKE_SOURCE_DIR}/${ME630_VENV_NAME}")


# ==============================================================================
# Find system Python
# ==============================================================================

message(STATUS "")
message(STATUS "============================================================")
message(STATUS " ME630 Assignments Setup")
message(STATUS "============================================================")
message(STATUS "")

message(STATUS "Searching for Python 3...")

find_package(
    Python3
    COMPONENTS Interpreter
    REQUIRED
)

message(STATUS "System Python found:")
message(STATUS "  Executable : ${Python3_EXECUTABLE}")
message(STATUS "  Version    : ${Python3_VERSION}")


# ==============================================================================
# Create Python Virtual Environment
# ==============================================================================

message(STATUS "")
message(STATUS "Checking Python virtual environment...")

if(NOT EXISTS "${ME630_VENV_DIR}")

    message(STATUS "Creating virtual environment:")
    message(STATUS "  ${ME630_VENV_DIR}")

    execute_process(
        COMMAND
            "${Python3_EXECUTABLE}"
            -m
            venv
            "${ME630_VENV_DIR}"

        RESULT_VARIABLE VENV_CREATE_RESULT
    )

    if(NOT VENV_CREATE_RESULT EQUAL 0)
        message(FATAL_ERROR
            "Failed to create Python virtual environment at:\n"
            "${ME630_VENV_DIR}"
        )
    endif()

    message(STATUS "Virtual environment created successfully.")

else()

    message(STATUS "Existing virtual environment found:")
    message(STATUS "  ${ME630_VENV_DIR}")

endif()


# ==============================================================================
# Determine virtual environment Python executable
# ==============================================================================

if(WIN32)

    set(
        ME630_PYTHON
        "${ME630_VENV_DIR}/Scripts/python.exe"
    )

    set(
        ME630_PIP
        "${ME630_VENV_DIR}/Scripts/pip.exe"
    )

else()

    set(
        ME630_PYTHON
        "${ME630_VENV_DIR}/bin/python"
    )

    set(
        ME630_PIP
        "${ME630_VENV_DIR}/bin/pip"
    )

endif()


# Verify that the virtual environment Python actually exists.
if(NOT EXISTS "${ME630_PYTHON}")
    message(FATAL_ERROR
        "Python executable was not found inside the virtual environment:\n"
        "${ME630_PYTHON}"
    )
endif()

message(STATUS "Virtual environment Python:")
message(STATUS "  ${ME630_PYTHON}")


# ==============================================================================
# Upgrade pip
# ==============================================================================

message(STATUS "")
message(STATUS "Checking pip...")

execute_process(
    COMMAND
        "${ME630_PYTHON}"
        -m
        pip
        install
        --upgrade
        pip

    RESULT_VARIABLE PIP_UPGRADE_RESULT
)

if(NOT PIP_UPGRADE_RESULT EQUAL 0)
    message(WARNING "Could not upgrade pip.")
else()
    message(STATUS "pip is ready.")
endif()


# ==============================================================================
# Python Package Installation Function
# ==============================================================================

#
# Usage:
#
#   me630_require_python_package(
#       PACKAGE_NAME
#       IMPORT_NAME
#   )
#
# Example:
#
#   me630_require_python_package(numpy numpy)
#
# PACKAGE_NAME:
#   Name given to pip.
#
# IMPORT_NAME:
#   Name used in Python:
#
#       import numpy
#
#
function(me630_require_python_package PACKAGE_NAME IMPORT_NAME)

    message(STATUS "")
    message(STATUS "Checking Python package: ${PACKAGE_NAME}")

    execute_process(
        COMMAND
            "${ME630_PYTHON}"
            -c
            "import ${IMPORT_NAME}"

        RESULT_VARIABLE PACKAGE_IMPORT_RESULT

        OUTPUT_QUIET
        ERROR_QUIET
    )

    if(PACKAGE_IMPORT_RESULT EQUAL 0)

        message(STATUS "  ${PACKAGE_NAME}: FOUND")

    else()

        message(STATUS "  ${PACKAGE_NAME}: NOT FOUND")
        message(STATUS "  Installing ${PACKAGE_NAME}...")

        execute_process(
            COMMAND
                "${ME630_PYTHON}"
                -m
                pip
                install
                "${PACKAGE_NAME}"

            RESULT_VARIABLE PACKAGE_INSTALL_RESULT
        )

        if(NOT PACKAGE_INSTALL_RESULT EQUAL 0)

            message(FATAL_ERROR
                "Failed to install required Python package:\n"
                "${PACKAGE_NAME}"
            )

        endif()

        # Verify installation
        execute_process(
            COMMAND
                "${ME630_PYTHON}"
                -c
                "import ${IMPORT_NAME}"

            RESULT_VARIABLE PACKAGE_VERIFY_RESULT

            OUTPUT_QUIET
            ERROR_QUIET
        )

        if(NOT PACKAGE_VERIFY_RESULT EQUAL 0)

            message(FATAL_ERROR
                "${PACKAGE_NAME} was installed, but Python still "
                "cannot import '${IMPORT_NAME}'."
            )

        endif()

        message(STATUS "  ${PACKAGE_NAME}: INSTALLED")

    endif()

endfunction()


# ==============================================================================
# Required Python Packages
# ==============================================================================

message(STATUS "")
message(STATUS "============================================================")
message(STATUS " Checking Python Dependencies")
message(STATUS "============================================================")

# Numerical computing
me630_require_python_package(
    numpy
    numpy
)

# Scientific computing
me630_require_python_package(
    scipy
    scipy
)

# Plotting
me630_require_python_package(
    matplotlib
    matplotlib
)

# Visualization Toolkit
me630_require_python_package(
    vtk
    vtk
)

# Jupyter environment
me630_require_python_package(
    jupyter
    jupyter
)

# Required for selecting the virtual environment as a Jupyter kernel
me630_require_python_package(
    ipykernel
    ipykernel
)


# ==============================================================================
# Register Jupyter Kernel
# ==============================================================================

message(STATUS "")
message(STATUS "Registering ME630 Jupyter kernel...")

execute_process(
    COMMAND
        "${ME630_PYTHON}"
        -m
        ipykernel
        install
        --user
        --name
        "ME630-venv"
        --display-name
        "Python (ME630-venv)"

    RESULT_VARIABLE KERNEL_INSTALL_RESULT
)

if(KERNEL_INSTALL_RESULT EQUAL 0)

    message(STATUS
        "Jupyter kernel registered as: Python (ME630-venv)"
    )

else()

    message(WARNING
        "Could not register the Jupyter kernel automatically.\n"
        "The virtual environment itself is still usable."
    )

endif()


# ==============================================================================
# Assignment Directory Creation
# ==============================================================================

message(STATUS "")
message(STATUS "============================================================")
message(STATUS " Creating Assignment Directory Structure")
message(STATUS "============================================================")


function(me630_create_assignment ASSIGNMENT_NUMBER)

    set(
        ASSIGNMENT_DIR
        "${CMAKE_SOURCE_DIR}/Assignment ${ASSIGNMENT_NUMBER}"
    )

    message(STATUS "Assignment ${ASSIGNMENT_NUMBER}")

    file(
        MAKE_DIRECTORY
        "${ASSIGNMENT_DIR}/Src"
    )

    file(
        MAKE_DIRECTORY
        "${ASSIGNMENT_DIR}/Images"
    )

    file(
        MAKE_DIRECTORY
        "${ASSIGNMENT_DIR}/Dumps"
    )

    file(
        MAKE_DIRECTORY
        "${ASSIGNMENT_DIR}/Results"
    )

    file(
        TOUCH
        "${ASSIGNMENT_DIR}/Src/.gitkeep"
        "${ASSIGNMENT_DIR}/Images/.gitkeep"
        "${ASSIGNMENT_DIR}/Dumps/.gitkeep"
        "${ASSIGNMENT_DIR}/Results/.gitkeep"
    )
endfunction()


foreach(
    ASSIGNMENT_NUMBER
    RANGE
    1
    ${ME630_NUM_ASSIGNMENTS}
)

    me630_create_assignment(
        ${ASSIGNMENT_NUMBER}
    )

endforeach()


# ==============================================================================
# Setup Summary
# ==============================================================================

message(STATUS "")
message(STATUS "============================================================")
message(STATUS " ME630 Setup Complete")
message(STATUS "============================================================")
message(STATUS "")
message(STATUS "Repository:")
message(STATUS "  ${CMAKE_SOURCE_DIR}")
message(STATUS "")
message(STATUS "Virtual environment:")
message(STATUS "  ${ME630_VENV_DIR}")
message(STATUS "")
message(STATUS "Python:")
message(STATUS "  ${ME630_PYTHON}")
message(STATUS "")
message(STATUS "Assignments created:")
message(STATUS "  1 through ${ME630_NUM_ASSIGNMENTS}")
message(STATUS "")

if(WIN32)

    message(STATUS "Activate the environment with:")
    message(STATUS "  ${ME630_VENV_NAME}\\Scripts\\activate")

else()

    message(STATUS "Activate the environment with:")
    message(STATUS "  source ${ME630_VENV_NAME}/bin/activate")

endif()

message(STATUS "")
message(STATUS "Start Jupyter with:")
message(STATUS "  jupyter lab")
message(STATUS "")
message(STATUS "Select the kernel:")
message(STATUS "  Python (ME630-venv)")
message(STATUS "")