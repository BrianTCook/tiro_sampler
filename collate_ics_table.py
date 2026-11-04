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

# The run sets that go into the table, in the order they should appear.
_PAPER_DIR = "/Users/btcook/system76/Desktop/spinning_gcs_paper/"
BASE_DIRS = (
    _PAPER_DIR + "sep1_tiro_ics_subdir/",
    _PAPER_DIR + "sep16_tiro_ics_subdir/",
)
# Only for runs old enough to predate the m_gc columns; anything current
# carries its own mass, since --M_GC is a swept parameter.
_FALLBACK_M_GC_MSUN = 5.0e4


def _tidal_radius_pc(host_name, r_gal_kpc, m_gc_msun):
    """Generalised Jacobi radius r_J = (G M_GC / lambda_1)^(1/3).

    lambda_1 is the largest eigenvalue of the effective (co-rotating) tidal
    tensor at r_gal, computed directly from the host's gala potential (see
    host_potential.tidal_eigenvalue). For a spherical host this equals
    nu Omega^2, recovering r_J = r_gal (M_GC / (3 M_gal))^(1/3) for a Kepler
    point mass.

    m_gc_msun is the run's own cluster mass rather than a module constant: r_J
    goes as M_GC^(1/3), so a hard-wired mass would misreport r_t -- and with it
    the tidal filling ratio log10(r_hm/r_t) -- by 26% for every 2x in mass, and
    would do it silently, since the two runs differ in no other column.
    """
    host = make_host(host_name)
    lam1 = host.tidal_eigenvalue(r_gal_kpc)
    return ((G_apy * m_gc_msun * u.Msun / lam1)**(1.0 / 3.0)).to_value(u.pc)


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
    t_dyn = np.sqrt((r_hm_pc * u.pc)**3 / (G_apy * m_gc_msun * u.Msun))
    t_rh = 0.138 * n_particles / np.log(gamma * n_particles) * t_dyn
    return t_rh.to_value(u.Gyr)


def collect_runs(base_dirs):
    """One row per seed directory, tagged with the run set it came from."""
    if isinstance(base_dirs, str):
        base_dirs = (base_dirs, )
    rows = []
    for dataset_order, base_dir in enumerate(base_dirs):
        dataset = os.path.basename(os.path.normpath(base_dir))
        pattern = os.path.join(base_dir, "tiro_ics_*", "seed*")
        _collect_one(pattern, dataset, dataset_order, rows)
    return rows


def _collect_one(pattern, dataset, dataset_order, rows):
    for seed_dir in sorted(glob.glob(pattern)):
        params_path = os.path.join(seed_dir, "parameters.csv")
        phys_path = os.path.join(seed_dir, "physical_units.csv")
        particles_path = os.path.join(seed_dir, "particles.csv")

        if not all(
                os.path.exists(p)
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

        # The realised mass from physical_units.csv, which is the requested
        # --M_GC to round-off (r_0 is solved for so that it is), falling back to
        # parameters.csv and then to the old hard-wired value for runs that
        # predate both columns.
        m_gc_msun = float(
            phys.get("m_gc_phys_msun")
            or params.get("m_gc_msun") or _FALLBACK_M_GC_MSUN)

        r_tidal_pc = _tidal_radius_pc(host_name, float(params["r_gal_kpc"]),
                                      m_gc_msun)

        r_hm_pc = float(phys["r_hm_pc"])
        tid_fill_ratio = np.log10(r_hm_pc / r_tidal_pc)

        # rho_0 is an input, so parameters.csv is the primary source; it is
        # echoed into physical_units.csv unchanged.
        rho_0_msun_pc3 = float(
            params.get("rho_0_msun_pc3") or phys["rho_0_msun_pc3"])

        n_particles = int(float(params["n_particles"]))
        t_rh_gyr = _half_mass_relaxation_gyr(n_particles, r_hm_pc, m_gc_msun)

        rows.append({
            "epsilon": float(phys["epsilon"]) / 1e-5,
            "r_tidal_pc": r_tidal_pc,
            "sigma_0_kms": float(phys["sigma_0_kms"]),
            "r_hm_pc": r_hm_pc,
            "tid_fill_ratio": tid_fill_ratio,
            "rot_mean": rot_mean,
            "t_rh_gyr": t_rh_gyr,
            "m_gc_msun": m_gc_msun,
            "rho_0_msun_pc3": rho_0_msun_pc3,
            "r_gal_kpc": float(params["r_gal_kpc"]),
            "zeta": float(params["zeta"]),
            "psi": float(params["psi_central"]),
            "host": host_name,
            "nu": nu,
            "seed": int(params["seed"]),
            "dataset": dataset,
            "dataset_order": dataset_order,
            "run_dir": os.path.dirname(seed_dir),
        })


# Everything here is a property of the model and its physical scales, which
# every realisation of a configuration shares; only rot_mean is measured from
# the particles and so varies with the seed.
_SEED_INDEPENDENT = ("r_hm_pc", "tid_fill_ratio", "t_rh_gyr", "r_tidal_pc",
                     "sigma_0_kms", "epsilon")


def average_over_realisations(rows):
    """Collapse the seeds of each configuration into a single row.

    The seed-independent columns are averaged too, but checked first: if the
    realisations of one configuration disagree about r_hm or t_rh then the runs
    are not what the directory name says they are, and a mean would bury that.
    """
    groups = {}
    for r in rows:
        groups.setdefault(r["run_dir"], []).append(r)

    averaged = []
    for run_dir, members in groups.items():
        out = dict(members[0])
        for key in _SEED_INDEPENDENT:
            vals = np.array([m[key] for m in members], dtype=float)
            scale = max(abs(float(vals.mean())), 1.0)
            if np.ptp(vals) > 1e-6 * scale:
                print(f"  WARNING: {key} varies across the seeds of "
                      f"{os.path.basename(run_dir)} "
                      f"(spread {np.ptp(vals):.3g})", file=sys.stderr)
            out[key] = float(vals.mean())
        rot = np.array([m["rot_mean"] for m in members], dtype=float)
        out["rot_mean"] = float(rot.mean())
        out["rot_stdev"] = float(rot.std(ddof=1)) if rot.size > 1 else 0.0
        out["n_seeds"] = rot.size
        out.pop("seed", None)
        averaged.append(out)
    return averaged


def _fixed(val, decimals=2):
    """Fixed-point value; collapses a rounded-to-zero negative to plain zero."""
    rounded = round(val, decimals)
    if rounded == 0.0:
        rounded = 0.0
    return f"{rounded:.{decimals}f}"


def format_latex_table(rows):
    r"""booktabs table; requires \usepackage{booktabs} in the document."""
    rows.sort(key=lambda r: (r.get("dataset_order", 0), r["r_gal_kpc"],
                             r["zeta"]))

    header = " & ".join([
        r"$r_{\mathrm{hm}}$ [pc]",
        r"$\log_{10}(r_{\mathrm{hm}}/r_{t})$",
        r"$\langle v_{\phi}/\sigma_{\rm local}\rangle$",
        r"$t_{\rm rh}$ [Gyr]",
        r"$r'_{\rm init}$ [kpc]",
        r"$M_{\rm GC}$ [$10^{4}\,M_{\odot}$]",
        r"$\rho_{0}$ [$M_{\odot}\,{\rm pc}^{-3}$]",
    ])

    lines = [
        r"\begin{table}",
        r"  \centering",
        r"  \caption{Summary of TIRO initial conditions, each row averaged "
        r"over its realisations. "
        r"$\rho_{0}$ is the central mass density of the model and "
        r"$M_{\rm GC}$ its total mass; together they set the length scale, "
        r"so $r_{\mathrm{hm}} \propto (M_{\rm GC}/\rho_{0})^{1/3}$ while the "
        r"tidal filling ratio $r_{\mathrm{hm}}/r_{t}$ depends on $\rho_{0}$ "
        r"alone.}",
        r"  \label{tab:tiro_ics}",
        r"  \begin{tabular}{ccccccc}",
        r"    \toprule",
        f"    {header} \\\\",
        r"    \midrule",
    ]

    for r in rows:
        lines.append(f"    {r['r_hm_pc']:.2f} & "
                     f"{r['tid_fill_ratio']:.2f} & "
                     f"{_fixed(r['rot_mean'])} & "
                     f"{r['t_rh_gyr']:.2f} & "
                     f"{r['r_gal_kpc']:.2f} & "
                     f"{r['m_gc_msun'] / 1e4:g} & "
                     f"{r['rho_0_msun_pc3']:g} \\\\")

    lines += [
        r"    \bottomrule",
        r"  \end{tabular}",
        r"\end{table}",
    ]
    return "\n".join(lines)


def main():
    rows = collect_runs(BASE_DIRS)
    if not rows:
        print("No tiro_ics_* directories found.", file=sys.stderr)
        sys.exit(1)

    averaged = average_over_realisations(rows)
    print(f"\n  {len(rows)} realisations -> {len(averaged)} configurations",
          file=sys.stderr)
    for r in sorted(averaged,
                    key=lambda x: (x["dataset_order"], x["r_gal_kpc"],
                                   x["zeta"])):
        print(f"    {r['dataset']:24s} r_gal={r['r_gal_kpc']:5.1f} "
              f"zeta={r['zeta']:+8.2f}  <v_phi/sigma> = "
              f"{r['rot_mean']:+.4f} +/- {r['rot_stdev']:.4f} "
              f"({r['n_seeds']} seeds)",
              file=sys.stderr)
    print(file=sys.stderr)

    print(format_latex_table(averaged))


if __name__ == "__main__":
    main()
