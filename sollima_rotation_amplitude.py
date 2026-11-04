#!/usr/bin/env python3
r"""Sollima et al. (2019) rotation amplitude A and A/sigma for TIRO ICs.

Observed Milky Way globular cluster rotation is usually quoted as A/sigma_0,
with A the Sollima, Baumgardt & Hilker (2019, MNRAS 485, 1460) rotation
amplitude and sigma_0 the central line-of-sight velocity dispersion (in this
project's spinning_gcs_with_streams.csv: A_Rot, sig0, A_over_sigma).  The
sampler and collate_ics_table.py instead report <r_perp omega / sigma_local>,
which is not the same quantity.  This script measures A from the particles
of each realisation exactly the way Sollima et al. measure it from stars, and
divides it by the model's own 1D velocity dispersion, so TIRO models and
observed clusters can be put on the same axis.

Everything is done in the Henon units of particles.csv.  A and sigma share
those units, so A/sigma needs no conversion to km/s and none is attempted.

The Sollima et al. (2019) prescription
--------------------------------------
Sect. 3.1 and Appendix A of the paper.  Observe the cluster along a line of
sight Z with sky axes (X, Y); the rotation axis has position angle theta_0
(anticlockwise from +Y) and inclination i to the line of sight.  Solid-body
rotation with angular velocity omega gives (their Eq. 2)

    v_Z    = omega R sin(theta - theta_0) sin i
    v_par  = omega R sin(theta - theta_0) cos i
    v_perp = omega [R cos(theta - theta_0) cos i + Z sin i]

with R the *projected* distance from the cluster centre, theta the position
angle of the star, and v_par, v_perp the sky-plane velocity components
parallel and perpendicular to the projected rotation axis (their Eq. A3)

    v_par  = -v_X sin theta_0 + v_Y cos theta_0
    v_perp =  v_X cos theta_0 + v_Y sin theta_0 .

To stay model independent they do *not* fit a rotation curve.  They replace
omega R by a single constant, the mean projected rotation velocity
A = <omega R>, drop the unobservable Z term (it has zero mean), and maximise
the likelihood of Eq. 3 over (theta_0, i, A) for the mean-velocity model

    vbar_Z    = A sin(theta - theta_0) sin i
    vbar_par  = A sin(theta - theta_0) cos i
    vbar_perp = A cos(theta - theta_0) cos i ,

each residual weighted by s_j^2 = sigma_j^2(R) + err_j^2, where sigma_j(R)
is the local velocity dispersion of component j at the star's position.  A
is then the amplitude of the sinusoidal modulation of the mean velocity with
position angle, deprojected with the fitted inclination.

What this script does with a model
----------------------------------
Each particle is a star with a perfect 3D velocity, so err = 0, the error
correlation rho = 0, and Eq. 3 collapses to a weighted least-squares problem
that is *linear* in A at fixed (theta_0, i).  The (theta_0, i) surface is
scanned analytically on a grid and polished with Nelder-Mead; the reported
A_unc is the Fisher (1/sqrt(sum w f^2)) error of the linear fit at the
optimum.  The weights use sigma_LOS(R) and sigma_PM(R) profiles measured
from the particles in equal-count bins of projected radius; the two
sky-plane components share one profile so the weights do not depend on
theta_0.  The fitted theta_0 and i are written out next to the true viewing
geometry as a check that the recovery works.

The 1D dispersions
------------------
``sigma_0``    central line-of-sight dispersion: the dispersion of v_Z about
               its mean for particles inside a projected aperture of
               --sigma0-aperture King radii (default 0.25).  This is the
               analogue of the Baumgardt & Hilker (2018) sig0 that the
               observed A/sigma ratios are normalised by, and is the ratio
               written as ``A_over_sigma0``.  sig0 is the CENTRAL (R -> 0)
               value of their best-fit model's dispersion profile, so the
               aperture has to sit inside the flat core of the profile: for
               these Psi = 3 models 1 King radius holds half the cluster and
               reads sigma 5-8% low, while sigma stops changing inside
               ~0.5 r_K.  0.25 r_K is converged and still holds ~10^4
               particles (0.7% counting error).  The King radius in Henon units
               is recovered from the particle half-mass radius and the
               r_king_pc / r_hm_pc ratio in physical_units.csv.
``sigma_1d``   global 1D dispersion, sqrt of the mean over all particles and
               all three components of the squared velocity about the bulk
               velocity.  It includes ordered rotation, so it is
               sqrt(2 K / 3 M) = sqrt(1/6) in Henon units up to the mass
               weighting.  Ratio ``A_over_sigma1d``.

Rotational kinetic-energy fraction xi
-------------------------------------
Sollima et al. argue (their Sect. 4) that A/sigma_0 is a poor indicator of
rotational support and prefer the fraction of kinetic energy in ordered
rotation, defined in their Appendix B as

    xi = < <v_phi>^2 > / < 3 sigma_r^2 >
       = integral rho <v_phi>^2 dV / integral rho <v^2> dV ,

the mass-weighted mean of the squared streaming velocity over the mass-
weighted total second velocity moment.  They get it from a Jeans model
fitted to each cluster; a model with known particles gives it directly and
independently of any viewing direction.  ``xi`` here is

    xi = sum_bins M_b ( vbar_phi,b^2 - var(v_phi)_b / n_b ) / sum_k m_k |v_k|^2

with v_phi the azimuthal velocity about the z rotation axis, the streaming
velocity vbar_phi measured in --xi-bins-r x --xi-bins-z equal-count bins of
(R_cyl, |z|), and the var/n term removing the upward bias that finite-count
means have (a non-rotating model then averages to zero rather than to
sigma^2/n).  Velocities are taken relative to the bulk velocity.

Viewing geometry
----------------
TIRO rotates about z.  The default view is edge-on (--inclination 90) along
the cluster x axis (--azimuth 0), so theta_0 should come back near 0 or 180
degrees and i near 90.  With --n-views > 1 each realisation is instead
observed from that many isotropically random directions, which shows how
much the deprojected A depends on the view for these flattened, triaxial
models.

Usage
-----
    python sollima_rotation_amplitude.py                 # both paper IC sets
    python sollima_rotation_amplitude.py DIR [DIR ...]   # any tiro_ics_*/seed* trees
    python sollima_rotation_amplitude.py --n-views 20 --legacy

Writes ``sollima_amplitude_by_seed.csv`` (one row per realisation and view)
and ``sollima_amplitude_by_config.csv`` (averaged over seeds and views) to
--out-dir (default: the current directory).
"""

import argparse
import csv
import glob
import os
import sys

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.spatial import cKDTree

_PAPER_DIR = "/Users/btcook/system76/Desktop/spinning_gcs_paper/"
DEFAULT_BASE_DIRS = (
    _PAPER_DIR + "sep1_tiro_ics_subdir/",
    _PAPER_DIR + "sep16_tiro_ics_subdir/",
)

# particles.csv is headerless; columns 2 and 4..9 are m, x, y, z, vx, vy, vz.
_PARTICLE_COLS = (2, 4, 5, 6, 7, 8, 9)


# ----------------------------------------------------------------------------
# I/O
# ----------------------------------------------------------------------------
def read_csv_row(path):
    with open(path) as f:
        return next(csv.DictReader(f))


def load_particles(path):
    """(masses, pos, vel) in Henon units, recentred on the mass-weighted mean."""
    df = pd.read_csv(path, header=None, usecols=_PARTICLE_COLS,
                     skipinitialspace=True)
    arr = df.to_numpy(dtype=float)
    m = arr[:, 0]
    pos = arr[:, 1:4]
    vel = arr[:, 4:7]
    w = m / m.sum()
    pos = pos - (w[:, None] * pos).sum(axis=0)
    vel = vel - (w[:, None] * vel).sum(axis=0)
    return m, pos, vel


def half_mass_radius(m, pos):
    r = np.sqrt((pos**2).sum(axis=1))
    order = np.argsort(r)
    cum = np.cumsum(m[order])
    return float(r[order][np.searchsorted(cum, 0.5 * cum[-1])])


# ----------------------------------------------------------------------------
# Projection
# ----------------------------------------------------------------------------
def line_of_sight(inclination_deg, azimuth_deg):
    """Unit LOS vector at angle `inclination` from the rotation (z) axis."""
    i = np.radians(inclination_deg)
    phi = np.radians(azimuth_deg)
    return np.array([np.sin(i) * np.cos(phi), np.sin(i) * np.sin(phi),
                     np.cos(i)])


def random_lines_of_sight(n, rng):
    v = rng.normal(size=(n, 3))
    return v / np.linalg.norm(v, axis=1)[:, None]


def sky_basis(los):
    """Sky axes (e_X, e_Y) for LOS `los`, with +Y along the projected z axis.

    Puts the true projected rotation axis at theta_0 = 0 (or 180), so the
    fitted position angle is an immediate sanity check.  For a LOS along z
    the projection of z vanishes and any sky orientation will do.
    """
    z = np.array([0.0, 0.0, 1.0])
    e_y = z - np.dot(z, los) * los
    if np.linalg.norm(e_y) < 1e-8:
        e_y = np.array([1.0, 0.0, 0.0]) - los[0] * los
    e_y /= np.linalg.norm(e_y)
    e_x = np.cross(e_y, los)
    e_x /= np.linalg.norm(e_x)
    return e_x, e_y


def project(pos, vel, los):
    """Sky coordinates (X, Y), depth Z, velocities (vX, vY, vZ), PA theta."""
    e_x, e_y = sky_basis(los)
    X, Y, Z = pos @ e_x, pos @ e_y, pos @ los
    vX, vY, vZ = vel @ e_x, vel @ e_y, vel @ los
    # Sollima et al. Appendix A: X = -R sin(theta), Y = R cos(theta).
    theta = np.arctan2(-X, Y)
    R = np.sqrt(X**2 + Y**2)
    return X, Y, Z, vX, vY, vZ, R, theta


# ----------------------------------------------------------------------------
# Dispersion profiles used as weights (s_j in Sollima et al. Eq. 3)
# ----------------------------------------------------------------------------
def local_dispersion_profiles(R, vX, vY, vZ, n_bins=20):
    """sigma_LOS(R) and sigma_PM(R) per particle from equal-count radial bins.

    sigma_PM is the 1D dispersion of the two sky-plane components pooled, so
    v_par and v_perp share a weight and the weights are theta_0 independent.
    """
    order = np.argsort(R)
    bins = np.array_split(order, n_bins)
    sig_los = np.empty_like(R)
    sig_pm = np.empty_like(R)
    for idx in bins:
        sig_los[idx] = np.std(vZ[idx])
        sig_pm[idx] = np.sqrt(0.5 * (np.var(vX[idx]) + np.var(vY[idx])))
    return np.maximum(sig_los, 1e-30), np.maximum(sig_pm, 1e-30)


# ----------------------------------------------------------------------------
# The Sollima et al. fit
# ----------------------------------------------------------------------------
class _RotationFit:
    """Weighted least squares for A at fixed (theta_0, i); scan over both.

    With the mean-velocity model linear in A, chi^2(theta_0, i, A) =
    C - 2 A N + A^2 D with

        N = sum_k [ w_Z v_Z sinD sin i + w_P (v_par sinD + v_perp cosD) cos i ]
        D = sum_k [ w_Z sin^2 D sin^2 i + w_P cos^2 i ]

    (D = theta - theta_0; sin^2 D + cos^2 D = 1 removes theta from the PM
    part of D).  Minimising over A gives A = N/D and chi^2_min = C - N^2/D,
    so the best (theta_0, i) maximises N^2/D.
    """

    def __init__(self, theta, vX, vY, vZ, sig_los, sig_pm):
        self.theta = theta
        self.vX, self.vY, self.vZ = vX, vY, vZ
        self.wZ = 1.0 / sig_los**2
        self.wP = 1.0 / sig_pm**2
        self.WP = float(self.wP.sum())

    def _sums(self, theta0):
        d = self.theta - theta0
        s, c = np.sin(d), np.cos(d)
        st0, ct0 = np.sin(theta0), np.cos(theta0)
        v_par = -self.vX * st0 + self.vY * ct0
        v_perp = self.vX * ct0 + self.vY * st0
        n_z = float(np.sum(self.wZ * self.vZ * s))
        d_z = float(np.sum(self.wZ * s * s))
        n_p = float(np.sum(self.wP * (v_par * s + v_perp * c)))
        return n_z, d_z, n_p

    def amplitude(self, theta0, inc):
        n_z, d_z, n_p = self._sums(theta0)
        si, ci = np.sin(inc), np.cos(inc)
        num = n_z * si + n_p * ci
        den = d_z * si * si + self.WP * ci * ci
        return num / den, den

    def _neg_merit(self, params):
        theta0, inc = params
        if not 0.0 <= inc <= 0.5 * np.pi:
            return np.inf
        n_z, d_z, n_p = self._sums(theta0)
        si, ci = np.sin(inc), np.cos(inc)
        num = n_z * si + n_p * ci
        den = d_z * si * si + self.WP * ci * ci
        return -(num * num) / den

    def fit(self, n_theta=180, n_inc=46):
        theta_grid = np.linspace(0.0, 2.0 * np.pi, n_theta, endpoint=False)
        inc_grid = np.linspace(0.0, 0.5 * np.pi, n_inc)
        best = (np.inf, None)
        for th in theta_grid:
            n_z, d_z, n_p = self._sums(th)
            si, ci = np.sin(inc_grid), np.cos(inc_grid)
            num = n_z * si + n_p * ci
            den = d_z * si * si + self.WP * ci * ci
            merit = -(num * num) / den
            k = int(np.argmin(merit))
            if merit[k] < best[0]:
                best = (merit[k], (th, inc_grid[k]))
        res = minimize(self._neg_merit, np.array(best[1]),
                       method="Nelder-Mead",
                       options={"xatol": 1e-6, "fatol": 1e-12})
        theta0, inc = res.x
        A, den = self.amplitude(theta0, inc)
        # Sollima et al. take A > 0; a negative A is the same rotation seen
        # with the axis position angle flipped by 180 degrees.
        if A < 0.0:
            A = -A
            theta0 += np.pi
        theta0 = np.mod(theta0, 2.0 * np.pi)
        return A, 1.0 / np.sqrt(den), theta0, inc


def sollima_amplitude(pos, vel, los, n_bins=20):
    """Return (A, A_unc, theta0_deg, i_deg, sigma_los_per_particle, proj)."""
    X, Y, Z, vX, vY, vZ, R, theta = project(pos, vel, los)
    sig_los, sig_pm = local_dispersion_profiles(R, vX, vY, vZ, n_bins)
    fit = _RotationFit(theta, vX, vY, vZ, sig_los, sig_pm)
    A, A_unc, theta0, inc = fit.fit()
    proj = {"R": R, "vZ": vZ}
    return A, A_unc, np.degrees(theta0), np.degrees(inc), proj


# ----------------------------------------------------------------------------
# Dispersions and reference quantities
# ----------------------------------------------------------------------------
def central_los_dispersion(R, vZ, aperture):
    sel = R < aperture
    if sel.sum() < 50:
        sel = np.argsort(R)[:50]
    return float(np.std(vZ[sel])), int(np.count_nonzero(sel))


def global_1d_dispersion(m, vel):
    w = m / m.sum()
    v0 = (w[:, None] * vel).sum(axis=0)
    return float(np.sqrt((w[:, None] * (vel - v0)**2).sum() / 3.0))


def mean_azimuthal_velocity(m, pos, vel):
    """Mass-weighted <v_phi> about the z axis, the intrinsic streaming speed."""
    R = np.maximum(np.hypot(pos[:, 0], pos[:, 1]), 1e-30)
    v_phi = (pos[:, 0] * vel[:, 1] - pos[:, 1] * vel[:, 0]) / R
    return float(np.average(v_phi, weights=m))


def rotation_energy_fraction(m, pos, vel, n_bins_r=20, n_bins_z=10):
    """xi = ordered rotational KE / total KE (Sollima et al. 2019, App. B).

    Streaming velocity <v_phi> is measured about the z axis in equal-count
    bins of cylindrical radius, each split into equal-count bins of |z|, and
    its square is corrected for the finite-count bias var/n.
    """
    R = np.maximum(np.hypot(pos[:, 0], pos[:, 1]), 1e-30)
    v_phi = (pos[:, 0] * vel[:, 1] - pos[:, 1] * vel[:, 0]) / R
    absz = np.abs(pos[:, 2])

    ordered = 0.0
    for idx_r in np.array_split(np.argsort(R), n_bins_r):
        for idx in np.array_split(idx_r[np.argsort(absz[idx_r])], n_bins_z):
            if idx.size < 2:
                continue
            w = m[idx]
            vbar = np.average(v_phi[idx], weights=w)
            var = np.average((v_phi[idx] - vbar)**2, weights=w)
            ordered += w.sum() * (vbar * vbar - var / idx.size)
    total = float(np.sum(m * (vel**2).sum(axis=1)))
    return ordered / total


def legacy_rot_over_disp(pos, vel, k_nn=30):
    """<r_perp omega / sigma_local>, vectorised copy of the collate measure."""
    X, Y = pos[:, 0], pos[:, 1]
    R = np.maximum(np.hypot(X, Y), 1e-30)
    omega = (X * vel[:, 1] - Y * vel[:, 0]) / R**2
    _, idx = cKDTree(pos).query(pos, k=k_nn + 1)
    nbr = vel[idx[:, 1:]]                       # (N, k, 3)
    dv = nbr - nbr.mean(axis=1, keepdims=True)
    sigma_local = np.maximum(np.sqrt((dv**2).mean(axis=(1, 2))), 1e-30)
    return float(np.mean(R * omega / sigma_local))


# ----------------------------------------------------------------------------
# Driver
# ----------------------------------------------------------------------------
def seed_dirs(base_dirs):
    out = []
    for order, base in enumerate(base_dirs):
        dataset = os.path.basename(os.path.normpath(base))
        for d in sorted(glob.glob(os.path.join(base, "tiro_ics_*", "seed*"))):
            out.append((order, dataset, d))
    return out


def analyse_seed(seed_dir, views, aperture_king, legacy, n_bins, xi_bins):
    params = read_csv_row(os.path.join(seed_dir, "parameters.csv"))
    phys = read_csv_row(os.path.join(seed_dir, "physical_units.csv"))
    m, pos, vel = load_particles(os.path.join(seed_dir, "particles.csv"))

    r_hm = half_mass_radius(m, pos)
    r_king = r_hm * float(phys["r_king_pc"]) / float(phys["r_hm_pc"])
    sigma_1d = global_1d_dispersion(m, vel)
    v_phi = mean_azimuthal_velocity(m, pos, vel)
    xi = rotation_energy_fraction(m, pos, vel, *xi_bins)
    legacy_val = legacy_rot_over_disp(pos, vel) if legacy else np.nan

    base = {
        "dataset": None,
        "run_dir": os.path.basename(os.path.dirname(seed_dir)),
        "seed": int(params["seed"]),
        "n_particles": int(float(params["n_particles"])),
        "psi": float(params["psi_central"]),
        "zeta": float(params["zeta"]),
        "zeta_crit": float(params.get("zeta_crit", "nan")),
        "r_gal_kpc": float(params["r_gal_kpc"]),
        "m_gc_msun": float(phys.get("m_gc_phys_msun") or params["m_gc_msun"]),
        "rho_0_msun_pc3": float(params["rho_0_msun_pc3"]),
        "nu": float(params.get("nu", 3.0)),
        "r_hm_henon": r_hm,
        "r_king_henon": r_king,
        "sigma_1d": sigma_1d,
        "mean_vphi": v_phi,
        "xi": xi,
        "legacy_rot_mean": legacy_val,
    }

    rows = []
    for view_id, (inc_true, az_true, los) in enumerate(views):
        A, A_unc, theta0, inc_fit, proj = sollima_amplitude(pos, vel, los,
                                                            n_bins)
        sigma_0, n_ap = central_los_dispersion(proj["R"], proj["vZ"],
                                               aperture_king * r_king)
        row = dict(base)
        row.update({
            "view": view_id,
            "inc_true_deg": inc_true,
            "az_true_deg": az_true,
            "A": A,
            "A_unc": A_unc,
            "theta0_fit_deg": theta0,
            "inc_fit_deg": inc_fit,
            "sigma_0": sigma_0,
            "n_in_aperture": n_ap,
            "A_over_sigma0": A / sigma_0,
            "A_over_sigma1d": A / sigma_1d,
        })
        rows.append(row)
    return rows


_AVERAGED = ("A", "A_unc", "theta0_fit_deg", "inc_fit_deg", "sigma_0",
             "sigma_1d", "A_over_sigma0", "A_over_sigma1d", "mean_vphi",
             "xi", "legacy_rot_mean", "r_hm_henon", "r_king_henon")


def average_by_config(df):
    keys = ["dataset", "run_dir", "psi", "zeta", "zeta_crit", "r_gal_kpc",
            "m_gc_msun", "rho_0_msun_pc3", "nu", "n_particles"]
    out = []
    for key, g in df.groupby(keys, sort=False):
        row = dict(zip(keys, key))
        row["n_seeds"] = g["seed"].nunique()
        row["n_views"] = g["view"].nunique()
        for col in _AVERAGED:
            row[col] = float(g[col].mean())
            row[col + "_std"] = float(g[col].std(ddof=1)) if len(g) > 1 else 0.0
        out.append(row)
    return pd.DataFrame(out)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="\n".join(__doc__.split("\n\n")[1:]))
    ap.add_argument("base_dirs", nargs="*", default=list(DEFAULT_BASE_DIRS),
                    help="directories holding tiro_ics_*/seed*/ trees "
                    "(default: the two paper IC sets)")
    ap.add_argument("--inclination", type=float, default=90.0,
                    help="viewing inclination of the z rotation axis to the "
                    "line of sight, degrees (default 90 = edge-on)")
    ap.add_argument("--azimuth", type=float, default=0.0,
                    help="azimuth of the line of sight about z, degrees "
                    "(default 0 = along the cluster x axis)")
    ap.add_argument("--n-views", type=int, default=1,
                    help="if > 1, observe each realisation from this many "
                    "random isotropic directions instead of the fixed view")
    ap.add_argument("--view-seed", type=int, default=0,
                    help="RNG seed for the random viewing directions")
    ap.add_argument("--sigma0-aperture", type=float, default=0.25,
                    help="projected aperture for sigma_0 in King radii "
                    "(default 0.25: inside the flat core of the profile, "
                    "the analogue of the central sig0)")
    ap.add_argument("--n-bins", type=int, default=20,
                    help="equal-count radial bins for the dispersion-profile "
                    "weights (default 20)")
    ap.add_argument("--xi-bins-r", type=int, default=20,
                    help="equal-count cylindrical-radius bins for the "
                    "streaming velocity in xi (default 20)")
    ap.add_argument("--xi-bins-z", type=int, default=10,
                    help="equal-count |z| bins per radial bin in xi "
                    "(default 10)")
    ap.add_argument("--legacy", action="store_true",
                    help="also compute the sampler's <r_perp omega / "
                    "sigma_local> (30 nearest neighbours) for comparison")
    ap.add_argument("--out-dir", default=".",
                    help="where to write the two CSVs (default: cwd)")
    args = ap.parse_args()

    if args.n_views > 1:
        rng = np.random.default_rng(args.view_seed)
        los_list = random_lines_of_sight(args.n_views, rng)
        views = []
        for los in los_list:
            # Fold to 0-90 degrees as in the paper: an axis pointing away
            # from the observer is the same axis with the opposite sense.
            inc = np.degrees(np.arccos(abs(np.clip(los[2], -1.0, 1.0))))
            az = np.degrees(np.arctan2(los[1], los[0]))
            views.append((inc, az, los))
    else:
        views = [(args.inclination, args.azimuth,
                  line_of_sight(args.inclination, args.azimuth))]

    dirs = seed_dirs(args.base_dirs)
    if not dirs:
        print("No tiro_ics_*/seed* directories found.", file=sys.stderr)
        sys.exit(1)

    rows = []
    for order, dataset, d in dirs:
        needed = ("parameters.csv", "physical_units.csv", "particles.csv")
        if not all(os.path.exists(os.path.join(d, f)) for f in needed):
            print(f"  skipping {d} (missing files)", file=sys.stderr)
            continue
        print(f"  {dataset}/{os.path.basename(os.path.dirname(d))}/"
              f"{os.path.basename(d)}", file=sys.stderr)
        seed_rows = analyse_seed(d, views, args.sigma0_aperture, args.legacy,
                                 args.n_bins, (args.xi_bins_r, args.xi_bins_z))
        for r in seed_rows:
            r["dataset"] = dataset
            r["dataset_order"] = order
            print(f"      view {r['view']:2d}: A = {r['A']:.4f} +/- "
                  f"{r['A_unc']:.4f}  theta0 = {r['theta0_fit_deg']:6.1f}  "
                  f"i = {r['inc_fit_deg']:5.1f} (true {r['inc_true_deg']:5.1f})"
                  f"  sigma_0 = {r['sigma_0']:.4f}  sigma_1d = "
                  f"{r['sigma_1d']:.4f}  A/sigma_0 = {r['A_over_sigma0']:.4f}"
                  f"  A/sigma_1d = {r['A_over_sigma1d']:.4f}"
                  f"  xi = {r['xi']:.4f}",
                  file=sys.stderr)
        rows.extend(seed_rows)

    df = pd.DataFrame(rows)
    os.makedirs(args.out_dir, exist_ok=True)
    by_seed = os.path.join(args.out_dir, "sollima_amplitude_by_seed.csv")
    by_cfg = os.path.join(args.out_dir, "sollima_amplitude_by_config.csv")
    df.to_csv(by_seed, index=False)
    cfg = average_by_config(df)
    cfg.to_csv(by_cfg, index=False)

    print(f"\n  {len(df)} rows -> {len(cfg)} configurations", file=sys.stderr)
    print(f"  wrote {by_seed}\n  wrote {by_cfg}\n", file=sys.stderr)

    cfg = cfg.sort_values(["dataset", "r_gal_kpc", "zeta"])
    hdr = (f"{'dataset':22s} {'r_gal':>5s} {'zeta':>8s} {'A':>8s} "
           f"{'sigma_0':>8s} {'A/sig0':>8s} {'+/-':>7s} {'A/sig1d':>8s} "
           f"{'+/-':>7s} {'i_fit':>6s} {'<v_phi>':>8s} {'xi':>7s} {'+/-':>7s}")
    if args.legacy:
        hdr += f" {'<vphi/sloc>':>11s}"
    print(hdr)
    for _, r in cfg.iterrows():
        line = (f"{r['dataset']:22s} {r['r_gal_kpc']:5.1f} {r['zeta']:+8.2f} "
                f"{r['A']:8.4f} {r['sigma_0']:8.4f} {r['A_over_sigma0']:8.4f} "
                f"{r['A_over_sigma0_std']:7.4f} {r['A_over_sigma1d']:8.4f} "
                f"{r['A_over_sigma1d_std']:7.4f} {r['inc_fit_deg']:6.1f} "
                f"{r['mean_vphi']:+8.4f} {r['xi']:7.4f} {r['xi_std']:7.4f}")
        if args.legacy:
            line += f" {r['legacy_rot_mean']:+11.4f}"
        print(line)


if __name__ == "__main__":
    main()
