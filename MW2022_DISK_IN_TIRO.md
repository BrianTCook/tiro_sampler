# Building TIRO initial conditions in the full MilkyWayPotential2022

2026-11-04. What it would take for `generate_ics_for_spinning_clusters.py` to accept the full MW2022
potential, disk included, instead of its spherical part, and what that would and would not buy for the
NGC 5466 runs.

## 1. What TIRO takes from the host today, and why the host must be spherical

TIRO (Arditi & Varri 2025) is a King model in equilibrium under the quadratic perturbing potential

    a_pert = (9 eps / 2) [ -zeta (x^2 + y^2) + z^2 - nu x^2 ],

with the model's +x pointing from the host to the cluster, +z along the orbital angular momentum, and
the model spinning about z at omega^2 = (1 + zeta) Omega^2. The host enters through two scalars
evaluated at the orbital radius R: Omega^2 = Phi'(R)/R, which sets eps = Omega^2 / (4 pi G rho_0), and
nu = 1 - Phi''(R)/Omega^2. The bracket is the co-rotating effective tidal tensor of a **circular orbit in
a spherical host**,

    T_eff = -Hess(Phi) + Omega^2 diag(1, 1, 0) = Omega^2 diag(nu, 0, -1),

written as a quadratic form, plus the asynchronous spin term. Every piece of the pipeline leans on that
diagonal form:

- `TIRO.py`: `tidal_harmonics` carries only the T00, T20 and T22 coefficients that `-nu x^2 + z^2`
  produces, the `f00` source in `poisson` is `-9 (1 - nu - 2 zeta)`, and the Frobenius start values
  `y0[2:4]` use the same combination; the external solution and the L1 search follow from those.
- `host_potential.py`: `SphericalMW2022Host` drops the MN3 disk, the one flattened component, because
  `nu` is defined through `Phi(|r|)`; `host_tidal_diagnostics` checks `lambda_1 / (nu Omega^2) = 1` and
  warns when the host is not spherical.
- `critical_spin.l1_radius` scans the +x axis for the saddle.
- `jacobi_escape.py` takes a bare spherical pair `Phi(r)`, `M(<r)` for `E_J` and `E_tilde`.
- The density envelope and the mass integrals (`_prepare_shells`, `_compute_dimless_mass_rhm_sigma`,
  `NBody.py`) integrate one octant and multiply by 8, which assumes the model is even in x, y and z.

So the KRIOS runs today orbit in the full MW2022 (`HOST_POTENTIAL_TO_USE=1`) with initial conditions
that are equilibria of a different, spherical host at a fiducial radius.

## 2. What the disk changes

### 2a. Circular orbit in the midplane

For an axisymmetric host and a circular orbit at z = 0 the co-rotating tensor is still diagonal in
(R, phi, z): the phi term is still exactly 0 (Phi_yy = Phi'/R = Omega^2 by axisymmetry), the radial term is
still nu Omega^2, but the vertical term is -Phi_zz = -nu_z^2, the square of the vertical epicyclic
frequency, not -Omega^2. The perturbation becomes

    a_pert = (9 eps / 2) [ -zeta (x^2 + y^2) + c_z z^2 - nu x^2 ],      c_z = nu_z^2 / Omega^2,

with c_z -> 1 far outside the disk and c_z > 1 inside it (the disk squeezes vertically). This is a
one-parameter generalization of TIRO; the model keeps its x, y, z reflection symmetry.

### 2b. NGC 5466's orbit: inclined, eccentric, precessing

For an inclined orbit the Hessian in the cluster's (r_hat, phi_hat, L_hat) frame is no longer diagonal and
its vertical axis is not L_hat; and because the disk torques the orbit, L_hat itself turns (17 deg over the
catalogue run, 28 deg over the tuned one), so there is no steadily rotating frame in which the model is an
equilibrium. Measured at the two t = -5 Gyr start points (r = 37-39 kpc), `T_eff / Omega_c^2` in the
(r, phi, L) frame with the instantaneous orbital Omega in the centrifugal term:

| | catalogue start, full MW2022 | catalogue start, spherical part | tuned start, full MW2022 | tuned start, spherical part |
|---|---|---|---|---|
| T_rr | +1.329 | +1.224 | +1.282 | +1.165 |
| T_phiphi | -0.865 | -0.845 | -0.899 | -0.885 |
| T_LL | -0.996 | -1.000 | -0.995 | -1.000 |
| largest off-diagonal | 0.015 | 0 | 0.017 | 0 |
| nu = 1 - H_rr/Omega_c^2 | 2.197 | 2.069 | 2.185 | 2.051 |

Two things follow. The disk's effect at the start points is modest: 8-10% in the radial term, under 1% in
the vertical term, off-diagonal couplings of 1-2%. The eccentricity is the larger departure from TIRO's
form: the phi term is -0.87 to -0.90 instead of 0 because the cluster is near apocentre, where the orbital
Omega is only 0.31-0.36 of the local circular Omega_c; and the radial term grows by a factor of order
(r_apo/r_peri)^3 ~ 500 by pericentre. Averaging the full-MW2022 tensor over the radial period containing
-5 Gyr in the co-moving frame gives (r, phi, L) diagonals of about (+3.0, +1.0, -0.95) with r-phi
couplings of 0.15 for both orbits, which is not a Hill tensor of any host. The disk matters most where the
cluster is stripped: over the catalogue orbit its share of ||T||_F has median 0.17 and reaches 0.8 at the
8 kpc pericentre (`ngc5466_krios/outputs/ngc5466_orbit_summary.txt`).

So for the NGC 5466 runs the disk is a second-order correction to an approximation whose first-order error
is the eccentric, precessing orbit. For a run on a circular midplane orbit inside the disk, by contrast,
2a is the whole story and c_z is an order-unity effect.

## 3. The generalization, in two levels

### Level 1: axisymmetric host, circular midplane orbit (one new parameter)

TIRO is vendored unmodified as a submodule, so this needs a fork (a branch of `lucyarditi/TIRO` under our
control, with the submodule pointer moved to it). Changes in `TIRO.py`, all driven by replacing the z^2
coefficient 1 by c_z, default 1:

- `Model.param` gains a fifth entry `c_z`; `Model([psi, eps, zeta, nu])` must keep working and give
  today's models bit for bit (that is the regression test).
- `tidal_harmonics`: with `-nu x^2 + c_z z^2 = r^2 [ (c_z - nu)/3 + (2 c_z + nu)/3 P2(cos theta)
  - (nu/2) sin^2 theta cos 2 phi ]`, the coefficients become `T00 = -3 sqrt(pi) (nu - c_z) r^2`,
  `T20 = 3 sqrt(pi/5) (2 c_z + nu) r^2`, `T22` unchanged.
- `poisson`: the `f00` source `-9 (1 - nu - 2 zeta)` becomes `-9 (c_z - nu - 2 zeta)`; the same
  combination in `y0[2]` and `y0[3]`. `constants`, `external_solution`, `external_gradient`,
  `tidal_radius` and `boundary` are written in terms of `T` and need no change.
- `Critical.py`, `NBody.py`, `Physical_Units.py`: pass the parameter through.

In the sampler:

- `host_potential.py`: a `MW2022Host` built from the full gala potential, with `vertical_coefficient(R)
  = Phi_zz(R, 0) / Omega^2(R)` from the Hessian it already evaluates in `tidal_eigenvalue`. The sphericity
  warning in `host_tidal_diagnostics` becomes a check that `T_eff` is diagonal in (R, phi, z), which holds
  for any axisymmetric host on a midplane circular orbit. `r_J^3 = G M / (nu Omega^2)` is unchanged: the
  largest eigenvalue is still nu Omega^2.
- `generate_ics_for_spinning_clusters.py`: pass `c_z` into `Model`; `--host mw2022` as a new choice;
  record `c_z` in `parameters.csv`; `calc_external_pe_numba` gets the c_z z^2 term.
- `critical_spin.py`: unchanged, the saddle is still on +x.
- `jacobi_escape.py`: assumes `Phi(r)`. Either extend it to the axisymmetric `Phi_eff(R, z)` or write the
  `E_J`/`E_tilde` columns as NaN for the new host and say so in the docstring. KRIOS does not read them.

Effort: about a day, most of it the TIRO fork and the regression check.

### Level 2: a general tidal tensor (inclined or eccentric orbits)

Replace `(nu, c_z)` by the full co-rotating tensor in the model frame, six numbers. Its trace sets the
monopole term and its traceless part is five real l = 2 harmonics instead of TIRO's two. The structure of
TIRO's solution survives: the internal solution `psi0 + eps [ f00 + (A0 Y20 + A2 Y22) gamma2 ]` becomes
`psi0 + eps [ f00 + sum_m A_m Y2m gamma2 ]` with the same radial function `gamma2`, since the l = 2
homogeneous equation does not know m, and the external solution adds the matching `a_m / r^3 Y2m` terms.
What has to change beyond bookkeeping:

- `l1_radius` must search along the eigenvector of the largest tensor eigenvalue, not +x, and `boundary`
  along any direction; both are one-dimensional root solves already.
- The octant integrals in the sampler and in `NBody.py` must become full-sphere integrals, and the
  rejection envelope in `_prepare_shells` must sample theta over [0, pi] and phi over [0, 2 pi): the model
  is no longer even in x, y, z. This is the main surgery.
- Where the tensor comes from is a modeling choice, not a code question. The planner
  (`ngc5466_krios/plan_ngc5466_ics.py`) already evaluates the full-MW2022 Hessian along the orbit. The
  instantaneous tensor at t = -5 Gyr is the only one that is a Hill tensor of anything; it comes with the
  local Omega_c of 5.1-5.4 km/s/kpc rather than the orbit-averaged 9.4-11.5 used now, so eps falls by a
  factor 4-5 and zeta_crit changes with it. The orbit-averaged tensor is not Hill-like (section 2b) and
  should not be fed to TIRO as if it were.

KRIOS needs nothing from either level. It already integrates in the full MW2022, and the Euler-angle spin
frame (`src/reference_frame/spin_orientation.h`) already orients the model with +x on the host direction
and +z on L_hat, which is the frame a level-2 tensor would be expressed in.

## 4. How to tell it worked

1. `c_z = 1`, or a diagonal (nu, 0, -1) tensor, reproduces the existing `tiro_ics_*` directories bit for
   bit from the same seed.
2. Pick a run that isolates the disk: a psi = 3 model on a circular midplane orbit at R = 8 kpc, sampled
   once with the spherical host and once with the full host at level 1, both integrated by KRIOS in the
   full MW2022. The level-1 model should show the smaller initial transient: virial ratio and r_hm in the
   first few dynamical times, the l = 2 power of the SCF coefficients, and the count of `escape` rows in
   the first 50 Myr. The spherical-host model starts out of equilibrium in the vertical direction by the
   factor c_z and must relax into it.
3. For NGC 5466 at level 2, the same comparison on the real orbit, with the expectation from section 2b
   that the difference is small at the start and that neither model is an equilibrium through pericentre.

## 5. Recommendation

Do level 1 if and when a midplane run inside the disk is planned; there the disk is an order-unity term
and the change is small. For the NGC 5466 sets the disk is not the limiting approximation, the orbit is,
and level 2 would spend most of its effort on symmetry bookkeeping to correct a 1-10% term while the
eccentricity leaves the model out of equilibrium by the time of the first pericentre regardless. The
current practice, the spherical host at the radius matching the orbit-averaged circular frequency with the
full MW2022 in KRIOS, should be stated as such in the paper, with the numbers in section 2b.
