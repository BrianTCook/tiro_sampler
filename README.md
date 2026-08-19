# tiro_sampler

Sampling and parameter-space exploration built on
[TIRO](https://github.com/lucyarditi/TIRO) (Arditi & Varri 2025), the
triaxial-rotating tidally-perturbed King model. TIRO itself is vendored
unmodified as a git submodule; everything here sits on top of it.

## Layout

| File | What it does |
| --- | --- |
| `generate_ics_for_spinning_clusters.py` | Samples phase-space initial conditions for rotating, tidally perturbed King models and writes them in Henon units with measured rotation kinematics. |
| `critical_spin.py` | Critical spin `zeta_crit` of a TIRO model from the Jacobi integral: the fastest rotation whose zero-velocity surface is still closed at L1. |
| `explore_critical_spin.py` | Sweeps `(r_gal, psi, rho_0)` to map how much rotation support a critically spinning model can carry. |
| `host_potential.py` | Host galactic potentials (gala-backed), supplying `Omega^2(R)` and the tidal anisotropy `nu(R) = 3 - dlnM/dlnR` that TIRO needs. |
| `jacobi_escape.py` | Restricted-three-body escape machinery for an extended spherical host: `Phi_eff`, the collinear Lagrange points, the generalised Jacobi radius `r_J^3 = R^3 m / (nu M(<R))`, and `E_Jacobi`. Run it (`python -m jacobi_escape`) to print the host's Lagrange-point structure. |
| `collate_ics_table.py` | Collates generated IC runs into a LaTeX summary table. |
| `plot_rotation_diagnostic.py` | Regenerates `rotation_diagnostic.png` from an existing `particles.csv`. |
| `get_cluster_ics.sh` | Driver for the IC runs used in the paper. |
| `tiro_path.py` | Puts the TIRO submodule on `sys.path` (TIRO is loose modules, not an installable package). |

## Getting started

```sh
git clone --recurse-submodules <this repo>
cd tiro_sampler
pip install -r requirements.txt
```

If you already cloned without `--recurse-submodules`:

```sh
git submodule update --init --recursive
```

Scripts that need TIRO do `import tiro_path` before `from TIRO import Model`;
no `PYTHONPATH` fiddling is required as long as the submodule is populated.

## Updating TIRO

```sh
git -C TIRO fetch origin && git -C TIRO checkout origin/main
git add TIRO && git commit -m "Bump TIRO submodule"
```
