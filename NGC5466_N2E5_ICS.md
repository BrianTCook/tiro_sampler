# What tiro_sampler must produce for the N = 2e5 NGC 5466 runs

2026-11-16. For KRIOS branch `ngc5466_models_old_orbits_n_2e5`, config set `src/ngc5466_n2e5_config_files/`:
a non-spinning NGC 5466 progenitor and one spinning at the critical rate **against** its orbit,
omega = -omega_crit <L_hat>, with <L_hat> the run-averaged direction of its orbital angular momentum, on the catalogue orbit (the one that ends at NGC 5466's present position).

## The two models

Both are psi = 3 TIRO models with the same physical scale, set by hand rather than by the mass-loss law:

| | value | source |
|---|---|---|
| N | 200 000 equal-mass particles | chosen |
| M_GC | 5.89e4 Msun | 1.05 x the present-day 5.61e4 Msun (Baumgardt) |
| r_hm | 14.44 pc | 1.05 x the present-day 13.75 pc |
| host, r_gal | `mw2022_sph`, 19.0897 kpc | radius whose circular frequency equals the catalogue orbit's averaged <Omega_c^2>^1/2 = 9.43 km/s/kpc; gives nu = 1.833 |
| rho_0 | 7.518 Msun/pc^3 | makes the zeta = -1 model's r_hm equal 14.44 pc at this M_GC (epsilon = 2.191e-4) |

```sh
# krios_env python, from tiro_sampler/, one at a time (or: bash get_ngc5466_ics.sh)
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
python generate_ics_for_spinning_clusters.py 200000 --psi 3 --host mw2022_sph --r_gal 19.0897 --M_GC 5.89e+04 --rho_0 7.518 --n_real 1 --zeta -1
python generate_ics_for_spinning_clusters.py 200000 --psi 3 --host mw2022_sph --r_gal 19.0897 --M_GC 5.89e+04 --rho_0 7.518 --n_real 1 --zeta_max
```

1. **Non-spinning** (`--zeta -1`, omega = 0): the control.
2. **Critical spin** (`--zeta_max`): zeta = 0.99 zeta_crit = 3.921 (zeta_crit = 3.961), i.e.
   omega = sqrt(1 + zeta) Omega = 2.22 Omega at the matched radius, spin period 294 Myr.

### The earlier N = 1.5e5 models are deprecated

These are the only NGC 5466 models in use; the earlier N = 1.5e5 set (`src/ngc5466_config_files/`) is deprecated.
They are not that set's models at a larger N. M_0 = 1.05 M_f and r_hm0 = 1.05 r_hm,f change M_GC
(5.947e4 -> 5.89e4), rho_0 (8.066 -> 7.518) and with them zeta_crit (4.39 -> 3.96). Directories generated with the
first set's M_GC and rho_0 at N = 2e5 carry `mgc5.95e+04` and `zeta4.4` in their names and will not match these inis.
Do not repoint the inis at such directories either: KRIOS would rescale them to 5.89e4 Msun and 14.44 pc, a lower
central density than they were built for, and the spinning file's zeta = 4.35 would then sit above this set's
zeta_crit = 3.96, so the model would start supercritical and shed stars immediately. Regenerate with the commands
above instead.

## The retrograde run needs no new sampler output

The sampler writes a model spinning about its own +z, with +x the tidal axis. "Retrograde" is a statement about how
that frame sits in the host, and KRIOS sets it: the ini's ZYZ Euler angles put the file's +z on -<L_hat>
and its +x on the host-to-cluster direction at the start (projected perpendicular to that axis). That is a proper rotation (a half turn about the tidal axis relative to
the prograde frame), and TIRO's model is even in y and z with the spin entering only as omega^2 = (1 + zeta) Omega^2,
so the rotated file is the same TIRO equilibrium with the sense of rotation reversed.

Consequences for the sampler:

- Generate the **ordinary** `--zeta_max` model. Do not negate velocities or flip any axis in the file; KRIOS does the
  flip, and doing it twice gives a prograde run.
- zeta_crit is sign-blind: TIRO's closed-Jacobi-surface criterion sees omega^2 only. Real retrograde clusters are
  harder to strip than prograde ones, so "critical" here means the TIRO limit, the same |omega| a prograde model would
  carry, not a measured retrograde stability limit.
- The spin axis is -<L_hat>, minus the unit vector of the time mean of L_hat(t) over the 5 Gyr run:
  <L_hat> = (+0.609172, -0.731662, +0.305908). The orbit's L_hat turns 16.8 deg over the run, so no fixed axis is
  antiparallel throughout; this one stays within 8.9 deg at all times (7.8 deg at the start, 8.9 deg today), against
  16.8 deg at the end for the start-of-run axis. The choice lives in the planner's Euler angles, not in the IC files.

## What to check after generating

Expected directories (the names must match `ICS_FILENAME` in the inis):

- `tiro_ics_N200000_psi3.0_zeta-1.0_hostmw2022_sph_nu1.833_mgc5.89e+04_rho08e+00_rgal19.0897/seed42/`
- `tiro_ics_N200000_psi3.0_zeta3.9_hostmw2022_sph_nu1.833_mgc5.89e+04_rho08e+00_rgal19.0897/seed42/`

In each `seed42/`:

- `particles.csv`: 200 000 rows, 16 columns, mass 5e-6 each (Henon units).
- `parameters.csv`: `zeta` = -1 and 3.9212; `zeta_crit` = 3.9608 in both; `nu` = 1.8327.
- `physical_units.csv`: `r_hm_pc` = 14.44 (control) and 14.54 (spinning); `m_gc_phys_msun` = 5.89e4;
  `epsilon` = 2.191e-4. KRIOS rescales both to R_HALF_MASS_PC = 14.437 pc.
- Spinning model: the generator's printed mean L_z is positive (spin about +z) and <v_rot/sigma> is about 0.19;
  control: both consistent with zero.

Then copy both directories to the KRIOS run's
`data/ngc5466_tiro_ics_subdir/`, replacing any `mgc5.95e+04` N = 200000 directories there (they belong to no
config set). At startup the retrograde
run must print `omega_hat (model +z) = (-0.609172, +0.731662, -0.305908)`, the negative of the control's.

Numbers come from `spinning_gcs_paper/ngc5466_krios/plan_ngc5466_n2e5_ics.py`
(`outputs/n2e5/ngc5466_n2e5_plan.txt`).
