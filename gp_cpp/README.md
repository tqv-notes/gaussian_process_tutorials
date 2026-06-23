# GP tutorial
A single-file implementation of standard Gaussian Process regression
(Rasmussen & Williams, *GPML* 2006, Algorithm 2.1).

- **File:** `gp.cpp`
- **Dependencies:** Armadillo + LAPACK/BLAS
- **Output:** `gp_pred_before.csv`, `gp_pred_after.csv`, `gp_training.csv`

# SPGP tutorial
A single-file, self-contained implementation of the **Sparse Pseudo-input
Gaussian Process** (Snelson & Ghahramani, NIPS 2006).

- **File:** `spgp.cpp`
- **Dependencies:** Armadillo + LAPACK/BLAS
- **Output:** three CSVs (`spgp_pred_before.csv`, `spgp_pred_after.csv`, `spgp_pseudo_inputs_ini.csv`, `spgp_pseudo_inputs.csv`)

## Build
These programs can be built via CMake as:

cmake -B build -DCMAKE_TOOLCHAIN_FILE=C:/Users/quangvu197/vcpkg/scripts/buildsystems/vcpkg.cmake
cmake --build build

build\Debug\gp_demo.exe
build\Debug\spgp_demo.exe

## Visualization
Plot results with the shared `plot_results.py`. 
```
python plot_results.py --gp        # plot GP
python plot_results.py --spgp      # plot SPGP
python plot_results.py --compare   # plot GP vs SPGP
```