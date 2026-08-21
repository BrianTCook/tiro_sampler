#!/usr/bin/env python3
"""Collate TIRO IC runs (tiro_ics_*/seed*/) into a LaTeX summary table."""

import csv
import glob
import os
import sys

import astropy.units as u
from astropy.constants import G as G_apy

import numpy as np
from scipy.spatial import cKDTree

from host_potential import make_host

BASE_DIR = "/home/btcook/Desktop/spinning_gcs_paper/aug19_tiro_ics_subdir/" #os.path.dirname(os.path.abspath(__file__))

_M_GC = 2.0e5 * u.Msun


def _tidal_radius_pc(host_name, r_gal_kpc):
    """Generalised Jacobi radius r_J = (G M_GC / lambda_1)^(1/3).

    lambda_1 is the largest eigenvalue of the effective (co-rotating) tidal
    tensor at r_gal, computed directly from the host's gala potential (see
    host_potential.tidal_eigenvalue). For a spherical host this equals
    nu Omega^2, recovering r_J = r_gal (M_GC / (3 M_gal))^(1/3) for a Kepler
    point mass.
    """
    host = make_host(host_name)
    lam1 = host.tidal_eigenvalue(r_gal_kpc)
    return ((G_apy * _M_GC / lam1)**(1.0 / 3.0)).to_value(u.pc)

def _read_csv_row(path):
    with open(path) as f:
        reader = csv.DictReader(f)
        return next(reader)


def _compute_rot_over_disp(particles_path, k_nn=30):
    """Mean of r_perp*omega/sigma_local over all particles.

    particles.csv columns 4..9 are (x, y, z, vx, vy, vz) in Henon units.
    """
    arr = np.loadtxt(particles_path, delimiter=",", usecols=(4, 5, 6, 7, 8, 9))
    pos = arr[:, :3]
    vel = arr[:, 3:]

    X, Y = pos[:, 0], pos[:, 1]
    v_X, v_Y = vel[:, 0], vel[:, 1]
    R_proj = np.maximum(np.sqrt(X * X + Y * Y), 1e-30)
    omega = (X * v_Y - Y * v_X) / R_proj**2

    tree = cKDTree(pos)
    _, idx = tree.query(pos, k=k_nn + 1)
    idx = idx[:, 1:]

    sigma_local = np.empty(len(pos))
    for i in range(len(pos)):
        nbr = idx[i]
        dv = vel[nbr] - np.mean(vel[nbr], axis=0)
        sigma_local[i] = np.sqrt(np.mean(dv * dv))
    sigma_local = np.maximum(sigma_local, 1e-30)

    return float(np.mean(R_proj * omega / sigma_local))


def _half_mass_relaxation_gyr(n_particles, r_hm_pc, m_gc_msun, gamma=0.6):
    """Spitzer half-mass relaxation time,

        t_rh = 0.138 N / ln(gamma N) * sqrt(r_hm^3 / (G M_GC)),

    returned in Gyr.
    """
    t_dyn = np.sqrt((r_hm_pc * u.pc)**3 /
                    (G_apy * m_gc_msun * u.Msun))
    t_rh = 0.138 * n_particles / np.log(gamma * n_particles) * t_dyn
    return t_rh.to_value(u.Gyr)


def collect_runs(base_dir):
    pattern = os.path.join(base_dir, "tiro_ics_*", "seed*")
    rows = []
    for seed_dir in sorted(glob.glob(pattern)):
        params_path = os.path.join(seed_dir, "parameters.csv")
        phys_path = os.path.join(seed_dir, "physical_units.csv")
        particles_path = os.path.join(seed_dir, "particles.csv")

        if not all(os.path.exists(p)
                   for p in (params_path, phys_path, particles_path)):
            print(f"  skipping {seed_dir} (missing files)", file=sys.stderr)
            continue

        params = _read_csv_row(params_path)
        phys = _read_csv_row(phys_path)

        print(f"  computing rot/disp for {seed_dir} ...", file=sys.stderr)
        rot_mean = _compute_rot_over_disp(particles_path)

        # Older runs predate the host column; default to the Kepler point mass.
        host_name = params.get("host", "kepler")
        nu = float(params.get("nu", 3.0))
        r_tidal_pc = _tidal_radius_pc(host_name, float(params["r_gal_kpc"]))

        r_hm_pc = float(phys["r_hm_pc"])
        tid_fill_ratio = np.log10(r_hm_pc / r_tidal_pc)

        n_particles = int(float(params["n_particles"]))
        m_gc_msun = float(phys["m_gc_phys_msun"])
        t_rh_gyr = _half_mass_relaxation_gyr(n_particles, r_hm_pc, m_gc_msun)

        rows.append({
            "epsilon": float(phys["epsilon"]) / 1e-5,
            "r_tidal_pc": r_tidal_pc,
            "sigma_0_kms": float(phys["sigma_0_kms"]),
            "r_hm_pc": r_hm_pc,
            "tid_fill_ratio": tid_fill_ratio,
            "rot_mean": rot_mean,
            "t_rh_gyr": t_rh_gyr,
            "r_gal_kpc": float(params["r_gal_kpc"]),
            "zeta": float(params["zeta"]),
            "psi": float(params["psi_central"]),
            "host": host_name,
            "nu": nu,
            "seed": int(params["seed"]),
        })

    return rows


def _fixed(val, decimals=2):
    """Fixed-point value; collapses a rounded-to-zero negative to plain zero."""
    rounded = round(val, decimals)
    if rounded == 0.0:
        rounded = 0.0
    return f"{rounded:.{decimals}f}"


def format_latex_table(rows):
    rows.sort(key=lambda r: (r["r_gal_kpc"], r["zeta"]))

    lines = [
        r"\begin{table}",
        r"  \centering",
        r"  \caption{Summary of TIRO initial-condition realisations.}",
        r"  \label{tab:tiro_ics}",
        r"  \begin{tabular}{ccccc}",
        r"    \hline\hline",
        (r"    $r_{\mathrm{hm}}$ [pc] & "
         r"$\log_{10}(r_{\mathrm{hm}}/r_{t})$ & "
         r"$\langle v_{\phi}/\sigma_{\rm local}\rangle$ & "
         r"$t_{\rm rh}$ [Gyr] & "
         r"$r'_{\rm init}$ [kpc] \\"),
        r"    \hline",
    ]

    for r in rows:
        lines.append(
            f"    {r['r_hm_pc']:.2f} & "
            f"{r['tid_fill_ratio']:.2f} & "
            f"{_fixed(r['rot_mean'])} & "
            f"{r['t_rh_gyr']:.2f} & "
            f"{r['r_gal_kpc']:.2f} \\\\"
        )

    lines += [
        r"    \hline",
        r"  \end{tabular}",
        r"\end{table}",
    ]
    return "\n".join(lines)


def main():
    rows = collect_runs(BASE_DIR)
    if not rows:
        print("No tiro_ics_* directories found.", file=sys.stderr)
        sys.exit(1)

    print(format_latex_table(rows))


if __name__ == "__main__":
    main()
