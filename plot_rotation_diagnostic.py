#!/usr/bin/env python3
"""Recreate rotation_diagnostic.png from a particles.csv without re-running ICs.

Usage:
    python plot_rotation_diagnostic.py PATH [PATH ...]

PATH may be either a particles.csv file or a directory containing one. The
output rotation_diagnostic.png is written next to each particles.csv.
"""

import argparse
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

try:
    matplotlib.style.use("/home/btcook/Desktop/github_repositories/"
                         "krios_python/matplotlib_style_file")
except OSError:
    pass

_PARTICLES_COLUMNS = [
    "id",
    "k",
    "m",
    "Reff",
    "x",
    "y",
    "z",
    "vx",
    "vy",
    "vz",
    "id2",
    "k2",
    "m2",
    "Reff2",
    "a",
    "e",
]


def load_particles(path):
    df = pd.read_csv(path,
                     header=None,
                     names=_PARTICLES_COLUMNS,
                     sep=r"\s*,\s*",
                     engine="python")
    pos = df[["x", "y", "z"]].to_numpy()
    vel = df[["vx", "vy", "vz"]].to_numpy()
    return pos, vel


def measure_rotation(pos, vel, k_nn=30):
    """Same kinematic measurement as TiroInitialConditionSampler._measure_rotation."""
    X, Y, _ = pos.T
    v_X, v_Y, _ = vel.T

    R_proj = np.maximum(np.sqrt(X**2 + Y**2), 1e-30)
    omega = (X * v_Y - Y * v_X) / R_proj**2

    tree = cKDTree(pos)
    _, idx = tree.query(pos, k=k_nn + 1)
    idx = idx[:, 1:]

    sigma_local = np.empty(len(pos))
    for i in range(len(pos)):
        nbr = idx[i]
        dv = vel[nbr] - np.mean(vel[nbr], axis=0)
        sigma_local[i] = np.sqrt(np.mean(dv**2))
    sigma_local = np.maximum(sigma_local, 1e-30)

    rot_over_disp = R_proj * omega / sigma_local

    # Normal-Inverse-Chi-Squared conjugate update with weakly-informed N(0,1) prior.
    n = rot_over_disp.size
    x_bar = float(np.mean(rot_over_disp))
    s_sq = float(np.var(rot_over_disp, ddof=1))
    mu_0, sigma_0_sq, kappa_0, nu_0 = 0.0, 1.0, 1.0, 1.0

    kappa_n = kappa_0 + n
    nu_n = nu_0 + n
    mu_n = (kappa_0 * mu_0 + n * x_bar) / kappa_n
    sigma_n_sq = (nu_0 * sigma_0_sq + (n - 1) * s_sq +
                  (kappa_0 * n / kappa_n) * (x_bar - mu_0)**2) / nu_n

    mean_unc = float(np.sqrt(sigma_n_sq / kappa_n * nu_n / (nu_n - 2.0)))
    stdev = float(np.sqrt(sigma_n_sq))
    stdev_unc = float(np.sqrt(sigma_n_sq / (2.0 * nu_n)))

    return rot_over_disp, float(mu_n), mean_unc, stdev, stdev_unc


def plot_rotation_diagnostic(rot_over_disp, mean, mean_unc, stdev, stdev_unc,
                             out_path):
    fig, ax = plt.subplots(1, 1, figsize=(5, 4), constrained_layout=True)

    ax.hist(rot_over_disp, bins=200, histtype="step", color="k", lw=1.0)
    ax.axvline(mean, color="red", lw=2.0)

    ax.set_xlabel(r"$r_{\perp}\,\omega\,/\,\sigma_{\rm local}$", fontsize=18)
    ax.set_ylabel(r"$N_{\mathrm{particles}}$", fontsize=18)
    ax.annotate(rf"$\mu = {mean:.4f} \pm {mean_unc:.4f}$",
                xy=(0.6, 0.9),
                xycoords="axes fraction",
                fontsize=12)
    ax.annotate(rf"$\sigma = {stdev:.4f} \pm {stdev_unc:.4f}$",
                xy=(0.6, 0.85),
                xycoords="axes fraction",
                fontsize=12)

    fig.savefig(out_path, dpi=500, bbox_inches="tight")
    plt.close(fig)


def resolve_particles_csv(path):
    p = Path(path)
    if p.is_dir():
        return p / "particles.csv"
    return p


def main():
    parser = argparse.ArgumentParser(
        description="Recreate rotation_diagnostic.png from particles.csv.")
    parser.add_argument(
        "paths",
        nargs="+",
        help="particles.csv files or directories containing one")
    parser.add_argument("--k_nn",
                        type=int,
                        default=30,
                        help="Number of nearest neighbours for sigma_local.")
    args = parser.parse_args()

    for raw in args.paths:
        particles_path = resolve_particles_csv(raw)
        if not particles_path.is_file():
            print(f"[skip] {particles_path}: not found")
            continue

        print(f"[load] {particles_path}")
        pos, vel = load_particles(particles_path)

        rot, mean, mean_unc, stdev, stdev_unc = measure_rotation(
            pos, vel, k_nn=args.k_nn)
        print(f"       mu    = {mean:+.4f} +/- {mean_unc:.4f}")
        print(f"       sigma =  {stdev:.4f} +/- {stdev_unc:.4f}")

        out_path = particles_path.parent / "rotation_diagnostic.png"
        plot_rotation_diagnostic(rot, mean, mean_unc, stdev, stdev_unc,
                                 out_path)
        print(f"[save] {out_path}")


if __name__ == "__main__":
    main()
