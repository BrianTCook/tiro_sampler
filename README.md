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
| `sollima_rotation_amplitude.py` | Measures the Sollima et al. (2019) rotation amplitude `A` from the particles of each IC realisation (joint fit of axis position angle, inclination and `A` to the three projected velocity components) and reports `A/sigma_0` and `A/sigma_1D`, the form in which MW GC rotation is tabulated, plus the rotational kinetic-energy fraction `xi` from their Appendix B. |
| `get_cluster_ics.sh` | Driver for the IC runs used in the paper. |
| `get_ngc5466_ics.sh` | Driver for the NGC 5466 stream runs: one non-spinning control and one maximally spinning model per orbit, with every number taken from the planner in `spinning_gcs_paper/ngc5466_krios/`. |
| `MW2022_DISK_IN_TIRO.md` | What it would take to build TIRO models in the full MilkyWayPotential2022, disk included, and why that is not done for the NGC 5466 runs. |
| `tiro_path.py` | Puts the TIRO submodule on `sys.path` (TIRO is loose modules, not an installable package). |

## Host potentials and what TIRO can take from them

TIRO is a King model in equilibrium under the quadratic perturbation

    a_pert = (9 eps / 2) [ -zeta (x^2 + y^2) + z^2 - nu x^2 ],

which is the co-rotating tidal tensor of a **circular orbit in a spherical
host** written as a quadratic form. Everything the host contributes is two
scalars at the orbital radius `R`: the circular frequency `Omega^2 = Phi'(R)/R`,
which sets `eps`, and the anisotropy `nu = 1 - Phi''(R)/Omega^2`. The code
assumes that diagonal form throughout (the harmonic expansion in `TIRO.py`,
the L1 search along `+x`, the octant integrals in the sampler), so
`host_potential.py` offers only spherical hosts: a Kepler point mass and
`mw2022_sph`, the NFW halo plus Hernquist bulge and nucleus of gala's
`MilkyWayPotential2022` with its MN3 disk dropped. Inside the disk that is a
real omission; the models built here are the spherical approximation to the
Milky Way, not the Milky Way. `MW2022_DISK_IN_TIRO.md` sets out the two
levels of generalisation that would lift the restriction (an axisymmetric
host on a midplane orbit needs one extra parameter; an inclined or eccentric
orbit needs the full six-component tensor and full-sphere integrals) and
argues that neither is warranted for the runs below.

### The NGC 5466 runs are not equilibria of the potential they orbit in

The KRIOS runs driven by `get_ngc5466_ics.sh` integrate in the **full**
MilkyWayPotential2022 on NGC 5466's own eccentric, inclined orbit
(`r_peri` 5-7 kpc, `r_apo` 46-55 kpc). Their initial conditions are TIRO
equilibria of the spherical host at a fiducial radius `r_gal` chosen so that
the spherical host's circular frequency equals the orbit-averaged
`<Omega_c^2>^(1/2)` over the radial period containing the t = -5 Gyr start.
`r_gal` (19.09 and 15.32 kpc for the catalogue and stream-tuned orbits) is
therefore a proxy for a frequency, not a place the cluster ever sits. The
approximation breaks in three places, in decreasing order of size:

1. **Eccentricity.** At the start the cluster is near apocentre, where the
   local `Omega_c` is only 0.31-0.36 of the orbit-averaged value used to build
   the model, and the azimuthal term of the tidal tensor is -0.87 to -0.90
   (in units of `Omega_c^2`) rather than TIRO's 0. The radial term grows by a
   factor of order `(r_apo/r_peri)^3` by pericentre, where the Pfeffer tidal
   radius (44-48 pc) falls inside the model's truncation radius (53-55 pc).
   No choice of host removes this; the model is out of equilibrium through
   every pericentre.
2. **`nu` mismatch.** The generator accepts only a host and `r_gal`, so it
   cannot match `Omega` and `nu` independently. Matching `Omega` leaves `nu`
   at 1.83 and 1.77 against the orbit-effective `<nu Omega_c^2>/<Omega_c^2>`
   of 2.06 and 2.01, 11-12% low. That shifts `r_J` by about 4% and
   `zeta_crit` by more. This is the only part of the mismatch that is a
   limitation of the sampler rather than of the physics.
3. **Disk and inclination.** At the start (37-39 kpc) the disk changes the
   radial tidal term by 8-10% and the vertical term by under 1%, with
   off-diagonal couplings of 1-2%; it carries a median 17% of the tensor norm
   along the orbit and up to 80% at pericentre. The disk also torques the
   orbit, turning the angular-momentum axis by 17-28 degrees over 5 Gyr, so no
   fixed spin frame is an equilibrium frame.

Evaluating the full potential at `r_gal` itself would not help: in the
midplane at 15-19 kpc the disk gives a vertical coefficient of 3-4 instead of
1, which describes a location the cluster never occupies. The practice is
therefore: spherical host at the `Omega`-matched radius for the initial
conditions, full MilkyWayPotential2022 in KRIOS, and the numbers above stated
as the approximation in the paper.

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

## Use of AI tools

Anthropic's Claude models (Claude Code) were used to write and validate parts
of this codebase, including portions of the code, the docstrings and the
design notes. Every line was reviewed by the author, who takes full
responsibility for the contents of this repository and for any errors in it.
