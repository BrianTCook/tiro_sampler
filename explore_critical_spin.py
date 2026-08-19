#!/usr/bin/env python3
"""How much rotation support a critically spinning TIRO model can carry.

The critical spin is the fastest a cluster can rotate with its Jacobi surface
still closed -- no star born already able to escape through L1 (``critical_spin``):

    omega_crit = sqrt(4 pi G rho_0 epsilon (1 + zeta_crit)).

What that buys observationally is the mass-weighted rotation-to-dispersion ratio

    A = < r omega_crit / sigma(psi(r)) >,

and the question this script answers is how to maximise A at a *fixed* orbital
radius, where the only free parameters are the concentration Psi and the central
density rho_0.

Two collapses make the answer legible
-------------------------------------
First, epsilon = Omega^2/(4 pi G rho_0) by construction, so the prefactor
cancels and

    omega_crit = Omega(r_gal) sqrt(1 + zeta_crit),

with Omega fixed by the host and r_gal alone.  Psi and rho_0 reach omega_crit
only through zeta_crit.

Second, in King units (lengths in r_0 = 3 sigma_K / sqrt(4 pi G rho_0),
velocities in sigma_K) the ratio r/sigma carries a factor 3/sqrt(4 pi G rho_0),
which cancels the sqrt(G rho_0) in omega_crit outright:

    A = 3 sqrt(epsilon (1 + zeta_crit)) <r/sigma>,

with <r/sigma> the mass-weighted structure integral over the critical model
itself (:func:`rotation_support`), taken at the epsilon and zeta_crit of that
grid point -- not in the spherical King limit.  Both factors therefore depend on
rho_0 only through epsilon.

Both also saturate, so A rises with rho_0 and flattens rather than turning over.
The reason is worth being precise about, because the obvious version of it is
wrong.  On the critical curve zeta_crit grows as 1/epsilon, so the *product*
chi = epsilon(1 + zeta_crit) -- which is what sets the centrifugal distortion --
tends to a Psi-only constant instead of vanishing: at Psi = 3, r_gal = 5 kpc it
is 1.181e-3 at rho_0 = 100 and 1.3045e-3 from rho_0 = 1e4 on.  A critical model
therefore stays rotationally flattened by a fixed amount however dense it is; it
is the tidal terms, which carry a bare epsilon (and nu), that die away.

So <r/sigma> saturates too, but *not* onto the spherical King value: for Psi = 3
it settles at 2.809 against the non-rotating King 2.654, a permanent +5.9%,
because the flattened boundary keeps mass out at large radius where sigma is
small and r/sigma is largest.  The offset runs +5.7% to +6.6% across
Psi = 1 to 7, which is why using the King limit understated A almost uniformly
and left the ranking intact.

What decides the optimum is entirely Psi: <r/sigma> climbs steeply with
concentration while the ceiling on chi falls faster.  With both factors
saturating, the ceiling on A is a function of Psi alone -- independent of r_gal
to four digits from 2 to 20 kpc, and of nu with it.

``A`` uses spherical r, as posed.  For solid-body rotation the azimuthal speed is
omega R with R = r sin(theta) the cylindrical radius, so the observable
<v_rot/sigma> is the same integral with an extra sin(theta); it is tabulated
alongside.  In the spherical limit the two differ by exactly <sin theta> = pi/4,
and on the real distorted model by very nearly that, so the ranking is the same
either way.

Against the sampler: for the Psi = 3, r_gal = 5 kpc, rho_0 = 100 critical model
this predicts <v_rot/sigma> = 0.229,
where the IC generator measures 0.187 +/- 0.007 from 20k particles.  The gap is
in the diagnostic, not here -- its sigma is a 30-nearest-neighbour estimate,
which in the sparse, steep-gradient outskirts smooths over the sigma gradient and
reads high: binned against the model, the particle sigma profile sits 1.4x above
sigma(psi) in the ninth radial decile and 1.8x in the tenth, while agreeing to
3% inside the half-mass radius.  Those outer bins are exactly the ones that
dominate <r/sigma>.

This supersedes plot_tiro_parameter_space.py, which plotted the same competition
through r_hm omega_crit / <sigma> -- a ratio of averages rather than the average
of the ratio -- on the critical curve at a single rho_0.
"""

import argparse
import os
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache

import numpy as np
import pandas as pd
from astropy import units as u
from scipy.integrate import fixed_quad, quad
from scipy.optimize import brentq

import tiro_path  # noqa: F401  (puts the TIRO submodule on sys.path)
from TIRO import Model

import matplotlib
import matplotlib.pyplot as plt

try:
    matplotlib.style.use("/home/btcook/Desktop/github_repositories/"
                         "krios_python/matplotlib_style_file")
except OSError:
    pass

from critical_spin import (SearchLimitExceeded, TidallyOverfilled,
                           critical_zeta, l1_radius, tidal_strength)
from host_potential import make_host

_DEFAULT_HOST = "mw2022_sph"
_DEFAULT_PSI = [3.0, 4.0, 5.0, 6.0, 7.0]
_DEFAULT_RHO_0 = [1.0e2, 1.0e3, 1.0e4]
_DEFAULT_R_GAL = (2.0, 20.0, 25)
# Gauss-Legendre orders for the (theta, phi) integrals over one octant. The
# angular structure is carried by Y20 and Y22 alone, so this converges fast:
# <r/sigma> is stable to five decimals by (6, 4), and (8, 6) leaves margin for
# the extra sin(theta) weight of the v_rot average.
_N_THETA, _N_PHI = 8, 6


def rotation_support(psi_central, epsilon, zeta, nu):
    """``(<r/sigma>, <r sin(theta)/sigma>)``, mass-weighted over the real model.

    Both averages are taken over the actual TIRO model at this (Psi, epsilon,
    zeta, nu) -- density, dispersion and the shape of the boundary all carry the
    first-order tidal and centrifugal distortion, rather than the epsilon -> 0
    spherical King limit this used to assume.  For Psi = 3 at r_gal = 5 kpc,
    rho_0 = 100, that raises <r/sigma> from 2.654 to 2.778 (+4.7%): the tidal
    bulge adds mass at large radius, where sigma is small and r/sigma is
    therefore largest, and the spherical limit misses exactly that.

    ``sigma`` is TIRO's ``velocity_dispersion(psi)``, the one-dimensional
    dispersion (in the isothermal limit psi -> infinity it tends to 1 in units
    of a^(-1/2), where the Maxwellian's per-component dispersion is 1/a).  That
    matches the convention of the sampler's k-NN estimate, so the two are
    directly comparable.

    Both averages come from one octant by the 8-fold symmetry of the
    perturbation.  The radial integral runs to the *local* boundary
    r_b(theta, phi) -- the psi = 0 root -- rather than to a fixed radius with the
    integrand clipped at zero.  That is what makes this affordable: clipping
    leaves a discontinuity inside the domain that adaptive quadrature cannot
    resolve, and the same integral costs 90 s as a plain ``tplquad`` against
    ~3 s this way, for the same answer to seven significant figures.
    """
    model = Model([psi_central, epsilon, zeta, nu])
    model.integrate()
    # constants() is a pure function of the integrated model and is otherwise
    # recomputed on every global_solution call, which is the inner loop here.
    model.constants = lru_cache(maxsize=1)(model.constants)
    r_l1 = l1_radius(model)

    def psi_at(r, theta, phi):
        return float(model.global_solution(r, theta, phi))

    @lru_cache(maxsize=None)
    def radial_moments(theta, phi):
        """``(mass, mass * <r/sigma>)`` along one ray, out to psi = 0.

        Memoised because the two angular integrals reuse the same Gauss nodes;
        without it every ray is integrated twice.
        """
        def psi_r(r):
            return psi_at(r, theta, phi)

        # psi > 0 at the centre and psi(r_L1) <= 0 for any model whose Jacobi
        # surface is closed, so the boundary is bracketed by construction.
        r_b = (brentq(psi_r, 1e-6 * model.r_trunc, r_l1, xtol=1e-12)
               if psi_r(r_l1) < 0.0 else r_l1)

        def mass_weight(r):
            psi = psi_r(r)
            return float(model.density(psi)) * r * r if psi > 0.0 else 0.0

        def r_over_sigma_weight(r):
            psi = psi_r(r)
            if psi <= 0.0:
                return 0.0
            return (float(model.density(psi)) * r**3 /
                    float(model.velocity_dispersion(psi)))

        return (quad(mass_weight, 0.0, r_b, limit=100)[0],
                quad(r_over_sigma_weight, 0.0, r_b, limit=100)[0])

    def octant_integral(component, sin_weighted):
        """Integrate one radial moment over the octant, in theta and phi.

        theta is the outer variable rather than cos(theta) so that the extra
        sin(theta) of the v_rot average enters as sin^2(theta) -- smooth -- and
        not as a square root with an endpoint singularity, which Gauss-Legendre
        resolves poorly.
        """
        def over_theta(theta_values):
            out = []
            for theta in theta_values:
                theta = float(theta)

                def over_phi(phi_values):
                    return np.array([
                        radial_moments(theta, float(phi))[component]
                        for phi in phi_values
                    ])

                value = fixed_quad(over_phi, 0.0, 0.5 * np.pi, n=_N_PHI)[0]
                weight = np.sin(theta)
                out.append(value * weight**2 if sin_weighted else value * weight)
            return np.array(out)

        return 8.0 * fixed_quad(over_theta, 0.0, 0.5 * np.pi, n=_N_THETA)[0]

    mass = octant_integral(0, sin_weighted=False)
    return (octant_integral(1, sin_weighted=False) / mass,
            octant_integral(1, sin_weighted=True) / mass)


def _solve_point(args):
    """One grid point: ``(psi, epsilon, nu) -> (zeta_crit, <r/sig>, <rsin/sig>, status)``.

    The structure integrals are done here, on the critical model itself, so they
    ride the same process pool as the zeta_crit solve -- they are the expensive
    half of a grid point and there is no reason to serialise them in the parent.

    Module level and argument-packed so it can be pickled to a worker.  The two
    ways a point can fail are kept apart, because only one of them is physics:
    ``overfilled`` means no spin closes the Jacobi surface at that (rho_0,
    r_gal), while ``search_limit`` means the bracket ran out below a zeta_crit
    that does exist.
    """
    psi, epsilon, nu = args
    try:
        zeta_crit = critical_zeta(psi, epsilon, nu)
    except TidallyOverfilled:
        return np.nan, np.nan, np.nan, "overfilled"
    except SearchLimitExceeded:
        return np.nan, np.nan, np.nan, "search_limit"
    except (ValueError, RuntimeError):
        return np.nan, np.nan, np.nan, "failed"

    try:
        r_over_sigma, r_sin_over_sigma = rotation_support(psi, epsilon,
                                                          zeta_crit, nu)
    except (ValueError, RuntimeError):
        return zeta_crit, np.nan, np.nan, "structure_failed"
    return zeta_crit, r_over_sigma, r_sin_over_sigma, "ok"


def scan(psi_values,
         rho_0_values,
         r_gal_values,
         host_name=_DEFAULT_HOST,
         n_jobs=None):
    """Tabulate zeta_crit, omega_crit and A over the (r_gal, Psi, rho_0) grid.

    Returns a tidy DataFrame, one row per grid point.  Omega and nu depend only
    on the host and r_gal, and <r/sigma> only on Psi, so each is evaluated once
    per axis value and broadcast.
    """
    host = make_host(host_name)
    r_gal_values = np.asarray(r_gal_values, dtype=float)

    omega_orb = np.sqrt(host.omega2(r_gal_values).to_value(u.s**-2))
    nu_values = np.asarray(host.nu(r_gal_values), dtype=float)

    rows = []
    for rho_0 in rho_0_values:
        eps_values = np.asarray(tidal_strength(host, r_gal_values, rho_0),
                                dtype=float)
        for psi in psi_values:
            for r_gal, eps, nu, om in zip(r_gal_values, eps_values, nu_values,
                                          omega_orb):
                rows.append({
                    "r_gal_kpc": r_gal,
                    "psi": psi,
                    "rho_0_msun_pc3": rho_0,
                    "nu": nu,
                    "epsilon": eps,
                    "omega_orbital_per_s": om,
                })

    tasks = [(row["psi"], row["epsilon"], row["nu"]) for row in rows]
    with ProcessPoolExecutor(max_workers=n_jobs) as pool:
        results = list(pool.map(_solve_point, tasks, chunksize=1))

    df = pd.DataFrame(rows)
    df["zeta_crit"] = [r[0] for r in results]
    df["r_over_sigma"] = [r[1] for r in results]
    df["r_sin_theta_over_sigma"] = [r[2] for r in results]
    df["status"] = [r[3] for r in results]

    # omega_crit = Omega sqrt(1 + zeta_crit); rho_0 was already absorbed into
    # zeta_crit through epsilon (see the module docstring).
    df["omega_crit_over_omega_orbital"] = np.sqrt(1.0 + df["zeta_crit"])
    omega_crit = df["omega_orbital_per_s"] * df["omega_crit_over_omega_orbital"]
    df["omega_crit_per_myr"] = (omega_crit.to_numpy() / u.s).to_value(u.Myr**-1)
    df["period_crit_myr"] = 2.0 * np.pi / df["omega_crit_per_myr"]
    # omega_crit in units of sqrt(G rho_0) -- TIRO's own dimensionless spin, and
    # what TiroInitialConditionSampler prints as "omega".
    df["omega_crit_over_sqrt_g_rho0"] = np.sqrt(
        4.0 * np.pi * df["epsilon"] * (1.0 + df["zeta_crit"]))
    # omega_crit in the model's own velocity/length units: 3 sqrt(chi) with
    # chi = eps (1+zeta), matching TiroInitialConditionSampler.omega_vel_units.
    omega_model = 3.0 * np.sqrt(df["epsilon"] * (1.0 + df["zeta_crit"]))
    # A = <r omega_crit / sigma>, the quantity to maximise, and the observable
    # <v_rot/sigma> that uses the cylindrical radius r sin(theta) instead.
    df["rot_support"] = omega_model * df["r_over_sigma"]
    df["v_rot_over_sigma"] = omega_model * df["r_sin_theta_over_sigma"]
    return df


def best_per_radius(df):
    """The (Psi, rho_0) maximising ``rot_support`` at each r_gal in ``df``."""
    solved = df.dropna(subset=["rot_support"])
    if solved.empty:
        return solved
    best = solved.loc[solved.groupby("r_gal_kpc")["rot_support"].idxmax()]
    return best.sort_values("r_gal_kpc")


def plot_scan(df, output_stem, host_name=_DEFAULT_HOST):
    """omega_crit, omega_crit/Omega and A against r_gal, one column per rho_0."""
    rho_values = np.unique(df["rho_0_msun_pc3"])
    psi_values = np.unique(df["psi"])
    colors = plt.cm.viridis(np.linspace(0.0, 0.9, len(psi_values)))

    fig, axes = plt.subplots(3,
                             len(rho_values),
                             figsize=(3.6 * len(rho_values), 8.8),
                             sharex=True,
                             sharey="row",
                             squeeze=False)

    for j, rho_0 in enumerate(rho_values):
        for color, psi in zip(colors, psi_values):
            sel = df[(df["rho_0_msun_pc3"] == rho_0)
                     & (df["psi"] == psi)].sort_values("r_gal_kpc")
            label = rf"$\Psi = {psi:g}$" if j == 0 else None
            axes[0, j].plot(sel["r_gal_kpc"], sel["omega_crit_per_myr"],
                            color=color, lw=1.4, label=label)
            axes[1, j].plot(sel["r_gal_kpc"],
                            sel["omega_crit_over_omega_orbital"],
                            color=color, lw=1.4)
            axes[2, j].plot(sel["r_gal_kpc"], sel["rot_support"],
                            color=color, lw=1.4)

        axes[0, j].set_title(
            rf"$\rho_0 = 10^{{{np.log10(rho_0):.0f}}}"
            r"\,M_\odot\,\mathrm{pc}^{-3}$", fontsize=10)
        axes[2, j].set_xlabel(r"$r_{\rm gal}\ [\rm kpc]$")
        axes[1, j].axhline(1.0, color="k", lw=0.5, ls=":", zorder=0)
        for ax in axes[:, j]:
            ax.set_xscale("log")

    axes[0, 0].set_yscale("log")
    axes[0, 0].set_ylabel(r"$\omega_{\rm crit}\ [\rm Myr^{-1}]$")
    axes[1, 0].set_ylabel(
        r"$\omega_{\rm crit} / \Omega = \sqrt{1+\zeta_{\rm crit}}$")
    axes[2, 0].set_ylabel(r"$\langle r\,\omega_{\rm crit}/\sigma \rangle$")
    axes[0, 0].legend(fontsize=8, frameon=False)
    fig.suptitle(f"TIRO critical spin, host = {host_name}", fontsize=11)

    path = f"{output_stem}.png"
    fig.savefig(path, bbox_inches="tight", dpi=300)
    print(f"  Saved: {path}")
    return fig


def plot_optimum(df, output_stem, host_name=_DEFAULT_HOST):
    """A over the (Psi, rho_0) plane at one r_gal, with the maximum marked."""
    r_gal = float(np.unique(df["r_gal_kpc"])[0])
    psi_values = np.unique(df["psi"])
    rho_values = np.unique(df["rho_0_msun_pc3"])

    grid = (df.pivot_table(index="psi",
                           columns="rho_0_msun_pc3",
                           values="rot_support",
                           dropna=False).reindex(index=psi_values,
                                                 columns=rho_values).to_numpy())

    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    mesh = ax.pcolormesh(np.log10(rho_values),
                         psi_values,
                         np.ma.masked_invalid(grid),
                         shading="nearest",
                         cmap="magma")
    fig.colorbar(mesh, ax=ax,
                 label=r"$\langle r\,\omega_{\rm crit}/\sigma \rangle$")

    solved = df.dropna(subset=["rot_support"])
    if not solved.empty:
        best = solved.loc[solved["rot_support"].idxmax()]
        ax.plot(np.log10(best["rho_0_msun_pc3"]), best["psi"], marker="*",
                ms=16, color="w", mec="k", mew=0.6, zorder=5)
        ax.set_title(
            rf"$r_{{\rm gal}} = {r_gal:g}$ kpc: max "
            rf"$\langle r\omega/\sigma\rangle = {best['rot_support']:.3f}$ at "
            rf"$\Psi = {best['psi']:g}$, "
            rf"$\rho_0 = 10^{{{np.log10(best['rho_0_msun_pc3']):.1f}}}$",
            fontsize=9)

    ax.set_xlabel(r"$\log_{10}\ \rho_0\ [M_\odot\,\mathrm{pc}^{-3}]$")
    ax.set_ylabel(r"$\Psi$")

    path = f"{output_stem}_optimum_r{r_gal:g}kpc.png"
    fig.savefig(path, bbox_inches="tight", dpi=300)
    print(f"  Saved: {path}")
    return fig


def _report(df, label):
    counts = df["status"].value_counts()
    print(f"  {label}: " + ", ".join(f"{n} {s}" for s, n in counts.items()))
    if "search_limit" in counts:
        print("    (search_limit points are bound models whose zeta_crit is "
              "above the search cap -- raise critical_spin._ZETA_MAX)")


def main():
    parser = argparse.ArgumentParser(
        description="Maximise <r omega_crit / sigma> over Psi and rho_0")
    parser.add_argument("--psi", type=float, nargs="+", default=_DEFAULT_PSI)
    parser.add_argument("--rho_0", type=float, nargs="+",
                        default=_DEFAULT_RHO_0,
                        help="Central densities in Msun/pc^3.")
    parser.add_argument("--r_gal_min", type=float, default=_DEFAULT_R_GAL[0])
    parser.add_argument("--r_gal_max", type=float, default=_DEFAULT_R_GAL[1])
    parser.add_argument("--n_r", type=int, default=_DEFAULT_R_GAL[2])
    parser.add_argument("--optimize_at_r_gal", type=float, default=None,
                        help="Also map A over a dense (Psi, rho_0) grid at this "
                        "orbital radius [kpc] and report the maximum.")
    parser.add_argument("--map_psi", type=float, nargs=3,
                        metavar=("MIN", "MAX", "N"), default=[1.0, 7.0, 13],
                        help="Psi range for --optimize_at_r_gal.")
    parser.add_argument("--map_rho_0", type=float, nargs=3,
                        metavar=("LOG10_MIN", "LOG10_MAX", "N"),
                        default=[1.0, 5.0, 9],
                        help="log10(rho_0) range for --optimize_at_r_gal.")
    parser.add_argument("--host", choices=["kepler", "mw2022_sph"],
                        default=_DEFAULT_HOST)
    parser.add_argument("--n_jobs", type=int, default=None,
                        help="Worker processes (default: os.cpu_count()).")
    parser.add_argument("--output", default="tiro_critical_spin")
    parser.add_argument("--reuse", action="store_true",
                        help="Load the CSV if it already exists instead of "
                        "recomputing the grid.")
    args = parser.parse_args()

    r_gal_values = np.geomspace(args.r_gal_min, args.r_gal_max, args.n_r)
    csv_path = f"{args.output}_host{args.host}.csv"

    if args.reuse and os.path.exists(csv_path):
        print(f"Loading {csv_path}")
        df = pd.read_csv(csv_path)
    else:
        n_points = len(args.psi) * len(args.rho_0) * args.n_r
        print(f"Scanning {n_points} grid points "
              f"({len(args.psi)} Psi x {len(args.rho_0)} rho_0 x "
              f"{args.n_r} radii), host = {args.host} ...")
        df = scan(args.psi, args.rho_0, r_gal_values,
                  host_name=args.host, n_jobs=args.n_jobs)
        df.to_csv(csv_path, index=False)
        print(f"  Saved: {csv_path}")

    _report(df, "radial scan")
    plot_scan(df, args.output, host_name=args.host)

    if args.optimize_at_r_gal is not None:
        psi_grid = np.linspace(args.map_psi[0], args.map_psi[1],
                               int(args.map_psi[2]))
        rho_grid = np.logspace(args.map_rho_0[0], args.map_rho_0[1],
                               int(args.map_rho_0[2]))
        print(f"\nMapping {len(psi_grid) * len(rho_grid)} (Psi, rho_0) points "
              f"at r_gal = {args.optimize_at_r_gal:g} kpc ...")
        df_map = scan(psi_grid, rho_grid, [args.optimize_at_r_gal],
                      host_name=args.host, n_jobs=args.n_jobs)
        map_csv = f"{args.output}_optimum_r{args.optimize_at_r_gal:g}kpc.csv"
        df_map.to_csv(map_csv, index=False)
        print(f"  Saved: {map_csv}")
        _report(df_map, "optimum map")

        best = best_per_radius(df_map)
        if not best.empty:
            row = best.iloc[0]
            print(f"  Maximum <r omega/sigma> = {row['rot_support']:.4f} "
                  f"(<v_rot/sigma> = {row['v_rot_over_sigma']:.4f}) at "
                  f"Psi = {row['psi']:g}, rho_0 = "
                  f"{row['rho_0_msun_pc3']:.3g} Msun/pc^3, "
                  f"zeta_crit = {row['zeta_crit']:.4g}")
        plot_optimum(df_map, args.output, host_name=args.host)

    print("Done.")


if __name__ == "__main__":
    main()
