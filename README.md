# Heptagon-Tiling-Solver

This project explores edge-to-edge tilings of the hyperbolic plane $\mathbb{H}^2$ in the Poincaré disk model by congruent heptagons.

## 📌 About This Project

The programs in this repository support computational exploration of this problem in two stages:

1. **Matrix generation:** Enumerate candidate edge-angle constraint matrices under the selected rank, balance, and equivalence settings.
2. **Trial tiling:** Use known matrix pairs to search for finite tiling patches with a specified number of heptagons.

**Algorithmic Constraints:**

* The program automatically filters out angle combinations that yield no positive solutions.
* Strict geometric rule: The degree of each vertex (i.e., the number of polygons meeting at a single vertex) must be $\e 3$.

## 📂 Core Files

| Stage | MATLAB script distributed as `.txt` | C++ MEX backend | Purpose |
| --- | --- | --- | --- |
| Matrix generation | `Edge-angle matrix generation.txt` | `all_basis_generator.cpp` | Generate candidate edge and angle matrices. |
| Trial tiling | `Trial tiling.txt` | `heptagon_search_final.cpp` | Search for trial tiling patches satisfying a supplied matrix pair. |

Each MATLAB script must be used with its corresponding C++ file. The first stage returns `E_results` and `A_results`; the second stage uses these arrays and records successful candidate indices in `jilu`.

## ⚙️ Requirements & Setup

The solver uses MATLAB for configuration and matrix processing, with multithreaded C++ MEX backends for computation. The trial-search backend uses OpenMP.

Recommended Windows setup:

1. **[MATLAB](https://www.mathworks.com/products/matlab.html):** R2024b or later.
2. **[Visual Studio 2022](https://visualstudio.microsoft.com/vs/older-downloads/):** Install the **Desktop development with C++** workload. Check the [MathWorks supported compiler list](https://www.mathworks.com/support/requirements/supported-compilers.html) for your MATLAB release.

Keep each script and its corresponding C++ file in the same MATLAB working folder. Change the `.txt` files to `.m` scripts, using valid MATLAB filenames such as `edge_angle_matrix_generation.m` and `trial_tiling.m`. Configure a supported C++ compiler with `mex -setup C++`; the scripts contain the MEX compilation commands.

## 🚀 Usage

### 1. Generate Edge-Angle Matrices

Use `Edge-angle matrix generation.txt` with `all_basis_generator.cpp` to enumerate candidate edge-angle matrices. Adjust the following variables before running:

| Parameter | Meaning |
| --- | --- |
| `n` | Number of polygon sides. Use `7` for this project and the trial backend. |
| `jieshu` | Vertex degree used to generate angle rows. Use `3` for the maximal heptagon case. |
| `Target_RE` | Target edge-matrix rank (`0`–`6`); `0` selects angle-only generation. |
| `Target_RA` | Target angle-matrix rank (`0`–`7`); `0` selects edge-only generation. |
| `pingheng` | Balance policy: `0` disables the check, `1` applies the weak check, and `2` adds the strong balance check. |
| `pu` | Angle-only equivalence: `0` for arbitrary column permutations, `1` for cyclic shifts and reflection. |
| `Run_Mode` | Hardware profile: `1` for a server, `2` for a laptop. |
| `known_pool` | Known matrix pairs used to filter cases covered by previously analyzed tilings in joint generation. |

**Add edge-angle matrix pairs that have already been generated, analyzed, and confirmed to admit tilings to `known_pool`.** Its entries use `bian` for the edge matrix and `jiao` for the angle matrix. Update the initialization block, since the script rebuilds the pool from `seeds` at startup.

The outputs are `E_results` (edge matrices) and `A_results` (angle matrices). Candidate `k` is the pair `E_results(:,:,k)` and `A_results(:,:,k)`.

**For subsequent trial tiling, use joint generation with both `Target_RE > 0` and `Target_RA > 0`.** The supplied generator defaults to angle-only mode (`Target_RE = 0`).

### 2. Run Trial Tilings

Use `Trial tiling.txt` with `heptagon_search_final.cpp` to trial known matrix pairs supplied through `E_results` and `A_results`. Each edge matrix has seven columns; each angle matrix has eight columns, including its constant column. The script derives the expanded constraints `Matrix_E` and `Matrix_A` from each pair.

Adjust the following variables before running:

| Parameter | Meaning |
| --- | --- |
| `Target_N` | Number of heptagons in the trial patch. Use an integer from `2` to `300` with the current backend. |
| `FindAll` | `0` stops after the requested number of solutions; `1` requests all solutions in the configured search. |
| `FindCount` | Requested number of solutions when `FindAll = 0`. |
| `SkipEdge` | `0` enforces edge constraints; `1` skips them. |
| `SkipAngle` | `0` enforces angle constraints; `1` skips them. |
| `Run_Mode` | `1` enables the AVX2 build profile; `2` uses the general laptop profile. |

**Successful candidate indices are recorded in `jilu`.** An index `k` refers to `E_results(:,:,k)` and `A_results(:,:,k)`. A candidate succeeds when at least `FindCount` solutions are found in count-limited mode, or at least one solution is found in all-solutions mode. For the current candidate, `res.count` gives the number of solutions and `res.solutions` contains their tile-adjacency data.

Trial results describe finite patches with `Target_N` tiles; full-plane and closed-surface tilings require further mathematical analysis.
