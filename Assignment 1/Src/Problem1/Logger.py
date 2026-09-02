# ============================================================================
# Logger.py
#
# Finite Difference Method Performance Logger
#
# OUTPUT:
#
#   MeshRefinementResults.txt
#
#   TemperatureSolution_h_0.02000.csv
#   TemperatureSolution_h_0.01000.csv
#   TemperatureSolution_h_0.00500.csv
#
# Existing files are overwritten every time the simulation is run.
# ============================================================================


import os
import gc
import sys
import time

from pathlib import Path
from collections import Counter
from contextlib import redirect_stdout

import numpy as np
import psutil

from scipy.linalg import lu_factor, lu_solve


# ============================================================================
# Default Output Directory
# ============================================================================

DEFAULT_DUMP_DIRECTORY = "../../Dumps/Problem1"


# ============================================================================
# Process Information
# ============================================================================

processInformation = psutil.Process(
    os.getpid()
)


# ============================================================================
# Console + File Output
# ============================================================================

class ConsoleAndFileOutput:

    def __init__(
        self,
        *outputStreams
    ):

        self.outputStreams = (
            outputStreams
        )


    def write(
        self,
        text
    ):

        for outputStream in self.outputStreams:

            outputStream.write(
                text
            )


    def flush(
        self
    ):

        for outputStream in self.outputStreams:

            outputStream.flush()


# ============================================================================
# General Utility Functions
# ============================================================================

def ConvertBytesToMB(
    numberOfBytes
):

    return (
        numberOfBytes
        / (1024 ** 2)
    )


def GetCurrentProcessMemoryMB():

    return ConvertBytesToMB(

        processInformation
        .memory_info()
        .rss
    )


def CreateDumpDirectory(
    dumpDirectory
):

    dumpPath = Path(
        dumpDirectory
    )

    dumpPath.mkdir(
        parents=True,
        exist_ok=True
    )

    return dumpPath


# ============================================================================
# Output Formatting
# ============================================================================

def PrintMainHeader(
    headerTitle
):

    print()

    print(
        "=" * 100
    )

    print(
        f"{headerTitle:^100}"
    )

    print(
        "=" * 100
    )


def PrintSectionHeader(
    sectionTitle
):

    print()

    print(
        "-" * 100
    )

    print(
        sectionTitle
    )

    print(
        "-" * 100
    )


def PrintInformationRow(
    informationName,
    informationValue
):

    print(
        f"{informationName:<58} : "
        f"{informationValue}"
    )


# ============================================================================
# Performance Measurement
# ============================================================================

def StartPerformanceMeasurement():

    return {

        "cpuStartTime":
            time.process_time(),

        "wallStartTime":
            time.perf_counter(),

        "memoryStartMB":
            GetCurrentProcessMemoryMB()
    }


def StopPerformanceMeasurement(
    startInformation
):

    cpuTime = (

        time.process_time()
        - startInformation[
            "cpuStartTime"
        ]
    )

    wallTime = (

        time.perf_counter()
        - startInformation[
            "wallStartTime"
        ]
    )

    memoryEndMB = (
        GetCurrentProcessMemoryMB()
    )

    memoryDifferenceMB = (

        memoryEndMB
        - startInformation[
            "memoryStartMB"
        ]
    )

    return {

        "cpuTime":
            cpuTime,

        "wallTime":
            wallTime,

        "memoryStartMB":
            startInformation[
                "memoryStartMB"
            ],

        "memoryEndMB":
            memoryEndMB,

        "memoryDifferenceMB":
            memoryDifferenceMB,

        "executed":
            True
    }


def CreateSkippedPerformanceMeasurement():

    currentMemoryMB = (
        GetCurrentProcessMemoryMB()
    )

    return {

        "cpuTime":
            0.0,

        "wallTime":
            0.0,

        "memoryStartMB":
            currentMemoryMB,

        "memoryEndMB":
            currentMemoryMB,

        "memoryDifferenceMB":
            0.0,

        "executed":
            False
    }


# ============================================================================
# Save Temperature Solution
# ============================================================================

def SaveTemperatureSolutionToCSV(
    temperatureSolutionVector,
    h,
    dumpDirectory
):

    dumpPath = (
        CreateDumpDirectory(
            dumpDirectory
        )
    )

    # ------------------------------------------------------------------------
    # NO TIMESTAMP
    #
    # Existing file for the same h is overwritten.
    # ------------------------------------------------------------------------

    fileName = (
        f"TemperatureSolution_h_{h:.5f}.csv"
    )

    filePath = (
        dumpPath
        / fileName
    )

    temperatureValues = (
        temperatureSolutionVector
        .flatten()
    )

    pointIndices = np.arange(
        temperatureValues.size
    )

    temperatureOutputData = (
        np.column_stack(
            (
                pointIndices,
                temperatureValues
            )
        )
    )

    np.savetxt(
        filePath,
        temperatureOutputData,
        delimiter=",",
        header="Index,Temperature",
        comments="",
        fmt=[
            "%d",
            "%.12f"
        ]
    )

    return filePath


# ============================================================================
# Mesh Information Report
# ============================================================================

def PrintMeshInformation(
    a,
    b,
    h,
    nX,
    nY,
    N,
    linaearizedMeshGrid
):

    PrintSectionHeader(
        "MESH INFORMATION"
    )

    numberOfXIntervals = (
        nX - 1
    )

    numberOfYIntervals = (
        nY - 1
    )

    numberOfComputationalColumns = (
        nX - 1
    )

    numberOfComputationalRows = (
        nY - 2
    )

    actualXStepSize = (

        a
        / numberOfXIntervals
    )

    actualYStepSize = (

        b
        / numberOfYIntervals
    )

    # ------------------------------------------------------------------------
    # Geometry
    # ------------------------------------------------------------------------

    PrintInformationRow(
        "Domain length in x-direction",
        f"{a:.6f} m"
    )

    PrintInformationRow(
        "Domain length in y-direction",
        f"{b:.6f} m"
    )

    PrintInformationRow(
        "Computational domain area",
        f"{a * b:.6f} m^2"
    )

    print()

    # ------------------------------------------------------------------------
    # Grid Spacing
    # ------------------------------------------------------------------------

    PrintInformationRow(
        "Requested mesh spacing, h",
        f"{h:.6f} m"
    )

    PrintInformationRow(
        "Actual x-direction spacing, dx",
        f"{actualXStepSize:.6f} m"
    )

    PrintInformationRow(
        "Actual y-direction spacing, dy",
        f"{actualYStepSize:.6f} m"
    )

    print()

    # ------------------------------------------------------------------------
    # Grid Dimensions
    # ------------------------------------------------------------------------

    PrintInformationRow(
        "Grid points in x-direction, nX",
        f"{nX:,}"
    )

    PrintInformationRow(
        "Grid points in y-direction, nY",
        f"{nY:,}"
    )

    PrintInformationRow(
        "x-direction intervals",
        f"{numberOfXIntervals:,}"
    )

    PrintInformationRow(
        "y-direction intervals",
        f"{numberOfYIntervals:,}"
    )

    PrintInformationRow(
        "Computational columns",
        f"{numberOfComputationalColumns:,}"
    )

    PrintInformationRow(
        "Computational rows",
        f"{numberOfComputationalRows:,}"
    )

    PrintInformationRow(
        "Total unknowns, N",
        f"{N:,}"
    )

    # ========================================================================
    # Node Classification
    # ========================================================================

    PrintSectionHeader(
        "COMPUTATIONAL NODE CLASSIFICATION"
    )

    meshPointTypeCounts = Counter(

        meshPointInformation[1]

        for meshPointInformation
        in linaearizedMeshGrid.values()
    )

    nodeTypeDisplayNames = {

        "Interior":
            "Interior nodes",

        "NorthEnd":
            "North boundary nodes",

        "SouthEnd":
            "South boundary nodes",

        "EastEnd":
            "East boundary nodes",

        "WestEnd":
            "West symmetry boundary nodes",

        "SymmetricSouthWestCorner":
            "Symmetric south-west corner",

        "SymmetricNorthWestCorner":
            "Symmetric north-west corner",

        "DirichletSouthEastCorner":
            "Dirichlet south-east corner",

        "DirichletNorthEastCorner":
            "Dirichlet north-east corner"
    }

    for (
        nodeType,
        displayName
    ) in nodeTypeDisplayNames.items():

        PrintInformationRow(
            displayName,
            (
                f"{meshPointTypeCounts.get(nodeType, 0):,}"
            )
        )

    totalClassifiedNodes = sum(
        meshPointTypeCounts.values()
    )

    print()

    PrintInformationRow(
        "Total classified nodes",
        f"{totalClassifiedNodes:,}"
    )

    PrintInformationRow(
        "Expected number of nodes, N",
        f"{N:,}"
    )

    PrintInformationRow(
        "Node classification consistency",
        (
            "PASS"
            if totalClassifiedNodes == N
            else "FAIL"
        )
    )


# ============================================================================
# Linear System Information
# ============================================================================

def PrintLinearSystemInformation(
    systemMatrix,
    temperatureVector
):

    PrintSectionHeader(
        "LINEAR SYSTEM INFORMATION"
    )

    numberOfMatrixElements = (
        systemMatrix.size
    )

    numberOfNonZeroMatrixElements = (
        np.count_nonzero(
            systemMatrix
        )
    )

    matrixSparsityPercentage = (

        100.0
        *
        (
            1.0
            -
            (
                numberOfNonZeroMatrixElements
                / numberOfMatrixElements
            )
        )
    )

    systemMatrixMemoryMB = (
        ConvertBytesToMB(
            systemMatrix.nbytes
        )
    )

    temperatureVectorMemoryMB = (
        ConvertBytesToMB(
            temperatureVector.nbytes
        )
    )

    # ------------------------------------------------------------------------
    # Dimensions
    # ------------------------------------------------------------------------

    PrintInformationRow(
        "System matrix dimensions",
        (
            f"{systemMatrix.shape[0]:,}"
            f" x "
            f"{systemMatrix.shape[1]:,}"
        )
    )

    PrintInformationRow(
        "Temperature vector dimensions",
        (
            f"{temperatureVector.shape[0]:,}"
            f" x "
            f"{temperatureVector.shape[1]:,}"
        )
    )

    PrintInformationRow(
        "Number of equations",
        f"{systemMatrix.shape[0]:,}"
    )

    PrintInformationRow(
        "Matrix data type",
        str(
            systemMatrix.dtype
        )
    )

    print()

    # ------------------------------------------------------------------------
    # Sparsity
    # ------------------------------------------------------------------------

    PrintInformationRow(
        "Total matrix coefficients",
        f"{numberOfMatrixElements:,}"
    )

    PrintInformationRow(
        "Non-zero matrix coefficients",
        f"{numberOfNonZeroMatrixElements:,}"
    )

    PrintInformationRow(
        "Matrix sparsity",
        f"{matrixSparsityPercentage:.4f} %"
    )

    print()

    # ------------------------------------------------------------------------
    # Storage
    # ------------------------------------------------------------------------

    PrintInformationRow(
        "Exact system matrix storage",
        f"{systemMatrixMemoryMB:.6f} MB"
    )

    PrintInformationRow(
        "Exact temperature vector storage",
        f"{temperatureVectorMemoryMB:.6f} MB"
    )

    PrintInformationRow(
        "Exact matrix + RHS storage",
        (
            f"{systemMatrixMemoryMB + temperatureVectorMemoryMB:.6f} MB"
        )
    )


# ============================================================================
# Performance Table Helper
# ============================================================================

def PrintPerformanceResult(
    stepName,
    performanceInformation
):

    if not performanceInformation[
        "executed"
    ]:

        print(
            f"{stepName:<38}"
            f"{'SKIPPED':>16}"
            f"{'SKIPPED':>16}"
            f"{'---':>18}"
        )

        return

    print(
        f"{stepName:<38}"
        f"{performanceInformation['cpuTime']:>16.6f}"
        f"{performanceInformation['wallTime']:>16.6f}"
        f"{performanceInformation['memoryDifferenceMB']:>18.3f}"
    )


# ============================================================================
# Remove Old Large Arrays Before Next Mesh
# ============================================================================

def ReleasePreviousSimulationArrays(
    notebookGlobals
):

    arrayNames = [

        "systemMatrix",

        "temperatureVector",

        "factorizedSystemMatrix",

        "pivotIndices",

        "temperatureSolutionVector"
    ]

    for arrayName in arrayNames:

        if arrayName in notebookGlobals:

            notebookGlobals[
                arrayName
            ] = None

    gc.collect()


# ============================================================================
# Run One Finite Difference Simulation
# ============================================================================

def RunFiniteDifferenceSimulation(
    a,
    b,
    h,
    MeshSetup,
    GenerateSystemOfEquations,
    notebookGlobals,
    MeshValidation=None,
    RunMeshValidation=False,
    DumpDirectory=DEFAULT_DUMP_DIRECTORY,
    SaveTemperatureCSV=True
):

    # ========================================================================
    # Clear Arrays From Previous Mesh
    # ========================================================================

    ReleasePreviousSimulationArrays(
        notebookGlobals
    )

    initialProcessMemoryMB = (
        GetCurrentProcessMemoryMB()
    )

    overallPerformanceStart = (
        StartPerformanceMeasurement()
    )

    # ========================================================================
    # STEP 1 — Mesh Setup
    # ========================================================================

    meshSetupPerformanceStart = (
        StartPerformanceMeasurement()
    )

    MeshSetup(
        a,
        b,
        h
    )

    meshSetupPerformance = (
        StopPerformanceMeasurement(
            meshSetupPerformanceStart
        )
    )

    # ------------------------------------------------------------------------
    # Retrieve updated mesh variables
    # ------------------------------------------------------------------------

    nX = notebookGlobals[
        "nX"
    ]

    nY = notebookGlobals[
        "nY"
    ]

    N = notebookGlobals[
        "N"
    ]

    linaearizedMeshGrid = (
        notebookGlobals[
            "linaearizedMeshGrid"
        ]
    )

    # ========================================================================
    # STEP 2 — Mesh Validation
    # ========================================================================

    if (
        RunMeshValidation
        and MeshValidation is not None
    ):

        meshValidationPerformanceStart = (
            StartPerformanceMeasurement()
        )

        MeshValidation(
            a,
            b,
            h
        )

        meshValidationPerformance = (
            StopPerformanceMeasurement(
                meshValidationPerformanceStart
            )
        )

    else:

        meshValidationPerformance = (
            CreateSkippedPerformanceMeasurement()
        )

    # ========================================================================
    # STEP 3 — Allocate Matrix and RHS
    # ========================================================================

    matrixAllocationPerformanceStart = (
        StartPerformanceMeasurement()
    )

    notebookGlobals[
        "temperatureVector"
    ] = np.zeros(
        (N, 1),
        dtype=float
    )

    notebookGlobals[
        "systemMatrix"
    ] = np.zeros(
        (N, N),
        dtype=float
    )

    matrixAllocationPerformance = (
        StopPerformanceMeasurement(
            matrixAllocationPerformanceStart
        )
    )

    # ========================================================================
    # STEP 4 — Generate System of Equations
    # ========================================================================

    matrixSetupPerformanceStart = (
        StartPerformanceMeasurement()
    )

    GenerateSystemOfEquations()

    matrixSetupPerformance = (
        StopPerformanceMeasurement(
            matrixSetupPerformanceStart
        )
    )

    systemMatrix = (
        notebookGlobals[
            "systemMatrix"
        ]
    )

    temperatureVector = (
        notebookGlobals[
            "temperatureVector"
        ]
    )

    # ========================================================================
    # STEP 5 — LU Factorization
    # ========================================================================

    luFactorizationPerformanceStart = (
        StartPerformanceMeasurement()
    )

    factorizedSystemMatrix, pivotIndices = (
        lu_factor(
            systemMatrix
        )
    )

    luFactorizationPerformance = (
        StopPerformanceMeasurement(
            luFactorizationPerformanceStart
        )
    )

    # ========================================================================
    # STEP 6 — Solve System
    # ========================================================================

    systemSolutionPerformanceStart = (
        StartPerformanceMeasurement()
    )

    temperatureSolutionVector = (
        lu_solve(
            (
                factorizedSystemMatrix,
                pivotIndices
            ),
            temperatureVector
        )
    )

    systemSolutionPerformance = (
        StopPerformanceMeasurement(
            systemSolutionPerformanceStart
        )
    )

    # ========================================================================
    # Store Results Back in Notebook
    # ========================================================================

    notebookGlobals[
        "factorizedSystemMatrix"
    ] = factorizedSystemMatrix

    notebookGlobals[
        "pivotIndices"
    ] = pivotIndices

    notebookGlobals[
        "temperatureSolutionVector"
    ] = temperatureSolutionVector

    # ========================================================================
    # Numerical Verification
    # ========================================================================

    residualVector = (

        systemMatrix
        @ temperatureSolutionVector
        - temperatureVector
    )

    residualNorm = (
        np.linalg.norm(
            residualVector
        )
    )

    # ========================================================================
    # Finish Numerical Timing
    #
    # CSV writing is not included in computational timing.
    # ========================================================================

    overallPerformance = (
        StopPerformanceMeasurement(
            overallPerformanceStart
        )
    )

    finalProcessMemoryMB = (
        GetCurrentProcessMemoryMB()
    )

    # ========================================================================
    # Exact Numerical Storage
    # ========================================================================

    systemMatrixMemoryMB = (
        ConvertBytesToMB(
            systemMatrix.nbytes
        )
    )

    temperatureVectorMemoryMB = (
        ConvertBytesToMB(
            temperatureVector.nbytes
        )
    )

    factorizedSystemMatrixMemoryMB = (
        ConvertBytesToMB(
            factorizedSystemMatrix.nbytes
        )
    )

    pivotIndicesMemoryMB = (
        ConvertBytesToMB(
            pivotIndices.nbytes
        )
    )

    temperatureSolutionVectorMemoryMB = (
        ConvertBytesToMB(
            temperatureSolutionVector.nbytes
        )
    )

    totalTrackedNumericalStorageMB = (

        systemMatrixMemoryMB

        + temperatureVectorMemoryMB

        + factorizedSystemMatrixMemoryMB

        + pivotIndicesMemoryMB

        + temperatureSolutionVectorMemoryMB
    )

    # ========================================================================
    # Save Temperature CSV
    # ========================================================================

    temperatureCSVFilePath = None

    if SaveTemperatureCSV:

        temperatureCSVFilePath = (
            SaveTemperatureSolutionToCSV(

                temperatureSolutionVector=
                    temperatureSolutionVector,

                h=
                    h,

                dumpDirectory=
                    DumpDirectory
            )
        )

    # ========================================================================
    # Direct Solver Timing
    # ========================================================================

    directSolverCPUTime = (

        luFactorizationPerformance[
            "cpuTime"
        ]

        + systemSolutionPerformance[
            "cpuTime"
        ]
    )

    directSolverWallTime = (

        luFactorizationPerformance[
            "wallTime"
        ]

        + systemSolutionPerformance[
            "wallTime"
        ]
    )

    # ========================================================================
    # Formatted Report
    # ========================================================================

    PrintMainHeader(
        f"FINITE DIFFERENCE PERFORMANCE REPORT — h = {h:.6f} m"
    )

    PrintMeshInformation(
        a,
        b,
        h,
        nX,
        nY,
        N,
        linaearizedMeshGrid
    )

    PrintLinearSystemInformation(
        systemMatrix,
        temperatureVector
    )

    # ========================================================================
    # Performance Table
    # ========================================================================

    PrintSectionHeader(
        "PERFORMANCE OF MAJOR COMPUTATIONAL STEPS"
    )

    print(
        f"{'Computational Step':<38}"
        f"{'CPU Time [s]':>16}"
        f"{'Wall Time [s]':>16}"
        f"{'Delta Memory [MB]':>18}"
    )

    print(
        "-" * 88
    )

    PrintPerformanceResult(
        "1. Mesh Setup",
        meshSetupPerformance
    )

    PrintPerformanceResult(
        "2. Mesh Validation",
        meshValidationPerformance
    )

    PrintPerformanceResult(
        "3. Matrix / RHS Allocation",
        matrixAllocationPerformance
    )

    PrintPerformanceResult(
        "4. System Matrix Setup",
        matrixSetupPerformance
    )

    PrintPerformanceResult(
        "5. LU Factorization",
        luFactorizationPerformance
    )

    PrintPerformanceResult(
        "6. System Solution",
        systemSolutionPerformance
    )

    print(
        "-" * 88
    )

    PrintPerformanceResult(
        "TOTAL",
        overallPerformance
    )

    # ========================================================================
    # Direct Solver Summary
    # ========================================================================

    PrintSectionHeader(
        "DIRECT SOLVER SUMMARY"
    )

    PrintInformationRow(
        "Solution method",
        "LU decomposition with partial pivoting"
    )

    print()

    PrintInformationRow(
        "LU factorization CPU time",
        (
            f"{luFactorizationPerformance['cpuTime']:.6f} s"
        )
    )

    PrintInformationRow(
        "System solution CPU time",
        (
            f"{systemSolutionPerformance['cpuTime']:.6f} s"
        )
    )

    PrintInformationRow(
        "Total direct solver CPU time",
        f"{directSolverCPUTime:.6f} s"
    )

    print()

    PrintInformationRow(
        "LU factorization wall time",
        (
            f"{luFactorizationPerformance['wallTime']:.6f} s"
        )
    )

    PrintInformationRow(
        "System solution wall time",
        (
            f"{systemSolutionPerformance['wallTime']:.6f} s"
        )
    )

    PrintInformationRow(
        "Total direct solver wall time",
        f"{directSolverWallTime:.6f} s"
    )

    # ========================================================================
    # Memory Usage
    # ========================================================================

    PrintSectionHeader(
        "MEMORY USAGE"
    )

    PrintInformationRow(
        "Process memory before simulation",
        f"{initialProcessMemoryMB:.3f} MB"
    )

    PrintInformationRow(
        "Process memory after simulation",
        f"{finalProcessMemoryMB:.3f} MB"
    )

    PrintInformationRow(
        "Net process memory change",
        (
            f"{finalProcessMemoryMB - initialProcessMemoryMB:.3f} MB"
        )
    )

    print()

    PrintInformationRow(
        "System matrix storage",
        f"{systemMatrixMemoryMB:.6f} MB"
    )

    PrintInformationRow(
        "Temperature vector storage",
        f"{temperatureVectorMemoryMB:.6f} MB"
    )

    PrintInformationRow(
        "Factorized system matrix storage",
        f"{factorizedSystemMatrixMemoryMB:.6f} MB"
    )

    PrintInformationRow(
        "Pivot indices storage",
        f"{pivotIndicesMemoryMB:.6f} MB"
    )

    PrintInformationRow(
        "Temperature solution vector storage",
        f"{temperatureSolutionVectorMemoryMB:.6f} MB"
    )

    PrintInformationRow(
        "Total tracked numerical storage",
        f"{totalTrackedNumericalStorageMB:.6f} MB"
    )

    # ========================================================================
    # Numerical Verification
    # ========================================================================

    PrintSectionHeader(
        "NUMERICAL VERIFICATION"
    )

    PrintInformationRow(
        "Temperature solution vector shape",
        str(
            temperatureSolutionVector.shape
        )
    )

    PrintInformationRow(
        "Temperature index range",
        f"0 to {N - 1}"
    )

    PrintInformationRow(
        "Minimum temperature",
        (
            f"{np.min(temperatureSolutionVector):.8f}"
        )
    )

    PrintInformationRow(
        "Maximum temperature",
        (
            f"{np.max(temperatureSolutionVector):.8f}"
        )
    )

    PrintInformationRow(
        "Residual norm ||A*T - b||",
        f"{residualNorm:.6e}"
    )

    # ========================================================================
    # CSV Information
    # ========================================================================

    PrintSectionHeader(
        "TEMPERATURE DATA OUTPUT"
    )

    if temperatureCSVFilePath is not None:

        PrintInformationRow(
            "Temperature CSV",
            str(
                temperatureCSVFilePath
            )
        )

        PrintInformationRow(
            "CSV index range",
            f"0 to {N - 1}"
        )

        PrintInformationRow(
            "CSV number of temperature values",
            f"{N:,}"
        )

    PrintMainHeader(
        "SIMULATION COMPLETE"
    )

    # ========================================================================
    # Return Results
    # ========================================================================

    return {

        # --------------------------------------------------------------------
        # Mesh
        # --------------------------------------------------------------------

        "h":
            h,

        "nX":
            nX,

        "nY":
            nY,

        "N":
            N,

        "systemMatrixShape":
            systemMatrix.shape,

        # --------------------------------------------------------------------
        # CPU Time
        # --------------------------------------------------------------------

        "meshSetupCPUTime":
            meshSetupPerformance[
                "cpuTime"
            ],

        "meshValidationCPUTime":
            meshValidationPerformance[
                "cpuTime"
            ],

        "matrixAllocationCPUTime":
            matrixAllocationPerformance[
                "cpuTime"
            ],

        "matrixSetupCPUTime":
            matrixSetupPerformance[
                "cpuTime"
            ],

        "luFactorizationCPUTime":
            luFactorizationPerformance[
                "cpuTime"
            ],

        "systemSolutionCPUTime":
            systemSolutionPerformance[
                "cpuTime"
            ],

        "directSolverCPUTime":
            directSolverCPUTime,

        "overallCPUTime":
            overallPerformance[
                "cpuTime"
            ],

        # --------------------------------------------------------------------
        # Wall Time
        # --------------------------------------------------------------------

        "meshSetupWallTime":
            meshSetupPerformance[
                "wallTime"
            ],

        "meshValidationWallTime":
            meshValidationPerformance[
                "wallTime"
            ],

        "matrixAllocationWallTime":
            matrixAllocationPerformance[
                "wallTime"
            ],

        "matrixSetupWallTime":
            matrixSetupPerformance[
                "wallTime"
            ],

        "luFactorizationWallTime":
            luFactorizationPerformance[
                "wallTime"
            ],

        "systemSolutionWallTime":
            systemSolutionPerformance[
                "wallTime"
            ],

        "directSolverWallTime":
            directSolverWallTime,

        "overallWallTime":
            overallPerformance[
                "wallTime"
            ],

        # --------------------------------------------------------------------
        # Memory
        # --------------------------------------------------------------------

        "systemMatrixMemoryMB":
            systemMatrixMemoryMB,

        "temperatureVectorMemoryMB":
            temperatureVectorMemoryMB,

        "factorizedSystemMatrixMemoryMB":
            factorizedSystemMatrixMemoryMB,

        "totalTrackedNumericalStorageMB":
            totalTrackedNumericalStorageMB,

        "initialProcessMemoryMB":
            initialProcessMemoryMB,

        "finalProcessMemoryMB":
            finalProcessMemoryMB,

        # --------------------------------------------------------------------
        # Verification
        # --------------------------------------------------------------------

        "residualNorm":
            residualNorm,

        # --------------------------------------------------------------------
        # Solution
        # --------------------------------------------------------------------

        "temperatureSolutionVector":
            temperatureSolutionVector,

        # --------------------------------------------------------------------
        # CSV
        # --------------------------------------------------------------------

        "temperatureCSVFilePath":
            (
                str(
                    temperatureCSVFilePath
                )

                if temperatureCSVFilePath
                is not None

                else None
            )
    }


# ============================================================================
# Mesh Refinement Study
# ============================================================================

def RunMeshRefinementStudy(
    a,
    b,
    meshStepSizes,
    MeshSetup,
    GenerateSystemOfEquations,
    notebookGlobals,
    MeshValidation=None,
    RunMeshValidation=False,
    DumpDirectory=DEFAULT_DUMP_DIRECTORY,
    SaveTemperatureCSV=True
):

    # ========================================================================
    # Create Dump Directory
    # ========================================================================

    dumpPath = (
        CreateDumpDirectory(
            DumpDirectory
        )
    )

    # ========================================================================
    # Single Report File
    #
    # NO TIMESTAMP.
    # This file is overwritten every run.
    # ========================================================================

    reportFilePath = (

        dumpPath
        / "MeshRefinementResults.txt"
    )

    meshRefinementResults = []

    # ========================================================================
    # Internal Refinement Execution
    # ========================================================================

    def ExecuteMeshRefinementStudy():

        PrintMainHeader(
            "STARTING MESH REFINEMENT STUDY"
        )

        PrintInformationRow(
            "Number of mesh refinements",
            len(
                meshStepSizes
            )
        )

        PrintInformationRow(
            "Mesh step sizes",
            (
                ", ".join(
                    f"{meshStepSize:.5f} m"
                    for meshStepSize
                    in meshStepSizes
                )
            )
        )

        # ====================================================================
        # Run Every Mesh
        # ====================================================================

        for h in meshStepSizes:

            simulationResults = (
                RunFiniteDifferenceSimulation(
                    a=a,
                    b=b,
                    h=h,
                    MeshSetup=MeshSetup,
                    GenerateSystemOfEquations=
                        GenerateSystemOfEquations,
                    notebookGlobals=
                        notebookGlobals,
                    MeshValidation=
                        MeshValidation,
                    RunMeshValidation=
                        RunMeshValidation,
                    DumpDirectory=
                        DumpDirectory,
                    SaveTemperatureCSV=
                        SaveTemperatureCSV
                )
            )

            meshRefinementResults.append(
                simulationResults
            )

        # ====================================================================
        # Final Comparison Table
        # ====================================================================

        PrintMainHeader(
            "MESH REFINEMENT PERFORMANCE COMPARISON"
        )

        print(
            f"{'h [m]':>10}"
            f"{'N':>12}"
            f"{'Matrix MB':>14}"
            f"{'Setup CPU':>14}"
            f"{'LU CPU':>14}"
            f"{'Solve CPU':>14}"
            f"{'Direct CPU':>14}"
            f"{'Total CPU':>14}"
        )

        print(
            "-" * 106
        )

        for result in meshRefinementResults:

            print(
                f"{result['h']:>10.5f}"

                f"{result['N']:>12,d}"

                f"{result['systemMatrixMemoryMB']:>14.3f}"

                f"{result['matrixSetupCPUTime']:>14.6f}"

                f"{result['luFactorizationCPUTime']:>14.6f}"

                f"{result['systemSolutionCPUTime']:>14.6f}"

                f"{result['directSolverCPUTime']:>14.6f}"

                f"{result['overallCPUTime']:>14.6f}"
            )

        print(
            "-" * 106
        )

        # ====================================================================
        # Files Created
        # ====================================================================

        PrintSectionHeader(
            "OUTPUT FILES"
        )

        PrintInformationRow(
            "Performance report",
            str(
                reportFilePath
            )
        )

        for result in meshRefinementResults:

            if (
                result[
                    "temperatureCSVFilePath"
                ]
                is not None
            ):

                PrintInformationRow(
                    (
                        f"Temperature CSV "
                        f"for h = "
                        f"{result['h']:.5f} m"
                    ),
                    result[
                        "temperatureCSVFilePath"
                    ]
                )

        PrintMainHeader(
            "MESH REFINEMENT STUDY COMPLETE"
        )

    # ========================================================================
    # Send Output To BOTH:
    #
    #   1. Jupyter console
    #   2. MeshRefinementResults.txt
    # ========================================================================

    with open(
        reportFilePath,
        "w",
        encoding="utf-8"
    ) as reportFile:

        combinedOutput = (
            ConsoleAndFileOutput(
                sys.stdout,
                reportFile
            )
        )

        with redirect_stdout(
            combinedOutput
        ):

            ExecuteMeshRefinementStudy()

    return meshRefinementResults