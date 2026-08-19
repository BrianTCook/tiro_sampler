#!/usr/bin/env python3
"""Restricted-three-body escape machinery: Phi_eff, the L1 point, and E_Jacobi.

Shared by ``initial_energy_distributions_jacobi.py`` and
``plot_distance_vs_energy_jacobi.py``, the physically-motivated counterparts of
the Giersz-criterion scripts of the same names.  Where those ask whether a star
clears ``alpha Phi_GC(r_t)`` -- an empirical energy referenced to the cluster
alone -- these ask the restricted-three-body question directly: can the star
reach the inner Lagrange point?

Frames and units
----------------
Everything is in the snapshot's units: kpc, km/s, Msun, so G = 4.30091e-6
kpc (km/s)^2 / Msun and potentials come out in (km/s)^2.

Positions are host-centred, because that is the frame Phi_eff is defined in: the
centrifugal term is about the rotation axis through the host, not the cluster.
Velocities are converted to the co-rotating xyz frame, v_rot = v - Omega x r,
which is the ``v`` of the paper's Eq. (2) -- the frame Phi_eff is static in.
Callers pass raw inertial (snapshot) velocities and ``rotating_frame_velocity``
does the conversion.

The rotating frame turns with the cluster's instantaneous orbital angular
velocity about the host,

    Omega = (r_GC x v_GC) / |r_GC|^2,

which for a circular orbit is exact and for an eccentric one is the osculating
value -- E_Jacobi is then conserved only to the extent that Omega is slowly
varying.  That is the standard restricted-three-body caveat and it is the main
approximation in what follows.

The effective potential and the criterion
-----------------------------------------
At host-centred position r, with the cluster centre at r_GC,

    Phi_eff(r) = Phi_host(|r|) + Phi_GC(|r - r_GC|) - (1/2) |Omega x r|^2,

and the Jacobi energy of a star is measured from the L1 saddle,

    E_J = v^2/2 + Phi_eff(r) - Phi_eff(r_1).

A star's Jacobi constant C = v^2/2 + Phi_eff(r) is conserved, and the
zero-velocity surface it defines opens at L1, so the star can leave the cluster
exactly when C >= Phi_eff(r_1).  That is the whole criterion:

    E_J > 0   the star can reach L1 and escape,
    E_J < 0   it is trapped inside a closed zero-velocity surface,

with no normalisation in between: for a bound star |E_J| is the energy per unit
mass it still needs to get out, in (km/s)^2.

Referencing to Phi_eff(r_1) is also what makes the arithmetic survive double
precision: Phi_eff(L1) ~ -1.4e5 (km/s)^2 at 5 kpc is almost entirely the host,
while every difference that matters is of order 10 (km/s)^2.  Subtracting it
cancels that common offset.

The tidal radius follows from the same solve as r_t = |r_1 - r_c|, the
cluster-to-L1 distance.

For mu = m_cluster/M_host ~ 1e-5 the L1 and L2 saddles are symmetric to high
accuracy, so using L1 alone is the standard simplification and not a further
approximation of consequence.

Earlier versions instead subtracted A = max[Phi_eff(z = 0)], attained at the
L4/L5 equilateral points, and reported the paper's normalised
E_tilde = (E_J - E_crit)/|E_crit|.  That put the escape threshold at a
configuration-dependent negative E_crit and made the L4/L5 level, which no star
in these runs comes near, the unit of energy.  L4/L5 are gone; the threshold is
now fixed at zero by construction.
"""

import glob
import os

import numpy as np
from scipy.optimize import brentq

# kpc (km/s)^2 / Msun -- positions in kpc, velocities in km/s, masses in Msun.
NEWTON_G = 4.30091e-6

# The host is the spherical part of gala's MilkyWayPotential2022 -- NFW halo,
# Hernquist bulge, Hernquist nucleus -- mirroring
# ``host_potential.SphericalMW2022Host``, which is what the ICs are built in.
# The flattened MN3 exponential disk is the one component both drop: Phi_eff and
# the L1 solve here assume a spherical host, and so does the TIRO nu.
#
# Written out analytically rather than imported from gala, which keeps this
# module a float-only dependency of the analysis scripts.  Values are gala
# 1.7.1's; ``host_potential`` reads the same numbers off the gala object, so
# compare against it if gala is ever updated.
#
# NFW: ``NFW_M`` is the characteristic mass M_s = 4 pi rho_s r_s^3, gala's ``m``
# parameter, so that
#   Phi(r)  = -G M_s ln(1 + r/r_s) / r          (gala sphericalnfw_value)
#   M(<r)   =  M_s [ln(1 + r/r_s) - (r/r_s)/(1 + r/r_s)]
NFW_M = 5.5427e11
NFW_RS = 15.626
# Hernquist components as (m [Msun], c [kpc]): bulge, then nucleus.
#   Phi(r)  = -G m / (r + c)
#   M(<r)   =  m r^2 / (r + c)^2
HERNQUIST_COMPONENTS = ((5.0e9, 1.0), (1.8142e9, 0.0688867))


def galaxy_potential(r):
    """Phi_host(r) in (km/s)^2: NFW halo + Hernquist bulge + nucleus."""
    r = np.maximum(np.asarray(r, dtype=float), 1e-12)
    x = r / NFW_RS
    phi = -NEWTON_G * NFW_M * np.log1p(x) / r
    for m, c in HERNQUIST_COMPONENTS:
        phi -= NEWTON_G * m / (r + c)
    return phi


def galaxy_enclosed_mass(r):
    """M_host(<r) in Msun, summed over the same three components."""
    r = np.maximum(np.asarray(r, dtype=float), 1e-12)
    x = r / NFW_RS
    mass = NFW_M * (np.log1p(x) - x / (1.0 + x))
    for m, c in HERNQUIST_COMPONENTS:
        mass = mass + m * r**2 / (r + c)**2
    return mass


def orbital_frequency(r_gc, v_gc):
    """Omega vector of the rotating frame, (km/s)/kpc.

    The instantaneous orbital angular velocity of the cluster about the host,
    ``(r x v) / |r|^2``.  Exact for a circular orbit; osculating otherwise.
    """
    r_gc = np.asarray(r_gc, dtype=float)
    v_gc = np.asarray(v_gc, dtype=float)
    return np.cross(r_gc, v_gc) / float(np.dot(r_gc, r_gc))


def centrifugal_potential(pos, omega_vec):
    """-(1/2) |Omega x r|^2 in (km/s)^2, for host-centred ``pos`` (..., 3)."""
    return -0.5 * np.sum(np.cross(np.asarray(pos, dtype=float), omega_vec)**2,
                         axis=-1)


def l1_distance(r_gc_mag, m_cluster, omega_mag):
    """Distance from the cluster centre to L1, in kpc.

    L1 sits on the host-cluster line where the net radial force in the rotating
    frame vanishes.  Writing ``s`` for the host-centred radius and
    ``ell = r_gc_mag - s`` for the distance from the cluster,

        F(ell) = G M_host(<s)/s^2 - G m_cluster/ell^2 - Omega^2 s

    is -inf as ell -> 0 (the cluster's own pull diverges) and positive as
    ell -> r_gc_mag (the host term survives while the others vanish), so it has
    exactly one root, which brentq finds.
    """
    if not (r_gc_mag > 0.0 and m_cluster > 0.0 and omega_mag > 0.0):
        raise ValueError("r_gc_mag, m_cluster and omega_mag must be positive")

    def force_balance(ell):
        s = r_gc_mag - ell
        return (NEWTON_G * float(galaxy_enclosed_mass(s)) / s**2
                - NEWTON_G * m_cluster / ell**2
                - omega_mag**2 * s)

    # Bracket on a Hill-radius scale and widen until the sign flips; the upper
    # bound is capped short of the host centre, where s -> 0 is singular.
    lo = 1e-8 * r_gc_mag
    hi = r_gc_mag * (m_cluster / (3.0 * float(galaxy_enclosed_mass(r_gc_mag))))**(1.0 / 3.0)
    for _ in range(60):
        if force_balance(hi) > 0.0:
            break
        hi = min(hi * 1.5, 0.999999 * r_gc_mag)
        if hi >= 0.999999 * r_gc_mag:
            break
    else:  # pragma: no cover - defensive
        raise RuntimeError("could not bracket L1")

    return float(brentq(force_balance, lo, hi, xtol=1e-14, rtol=1e-14))


def l1_effective_potential(r_gc_mag, m_cluster, omega_mag, ell=None):
    """(Phi_eff(L1) in (km/s)^2, distance from the cluster to L1 in kpc).

    The cluster is treated as a point mass here: L1 lies at roughly the tidal
    radius, outside essentially all of the bound mass, so Phi_GC there is
    Keplerian to the same accuracy the criterion itself has.
    """
    if ell is None:
        ell = l1_distance(r_gc_mag, m_cluster, omega_mag)
    s = r_gc_mag - ell
    phi = (float(galaxy_potential(s)) - NEWTON_G * m_cluster / ell
           - 0.5 * omega_mag**2 * s**2)
    return phi, ell


def load_tidal_radius(radii_info_csv):
    """``(t_Myr, r_tidal_kpc)`` from a KRIOS ``radii_info_*`` time series.

    KRIOS's own tidal radius, from ``System::compute_tidal_radius``: the
    fixed-point solve of ``r_t = (G M_bound(<r_t) / lambda1_eff)^(1/3)`` with
    ``lambda1_eff = lambda1 - (lambda2 + lambda3)/2``, the Pfeffer et al. (2018)
    effective eigenvalue of the *host* tidal tensor.  That is not the
    restricted-three-body result: the ``-(lambda2+lambda3)/2`` construction
    folds a rotational contribution into the eigenvalues as a proxy for the
    centrifugal term, and it is built from the host tensor alone, so it is blind
    to the cluster's own shape.  It agrees with :func:`l1_distance` to ~0.05%.

    UNITS: with ``IO_IN_PHYSICAL_UNITS=true`` (all the spinning_gcs runs) ``t``
    is in Myr and every radius column is in **pc**, while this module works in
    kpc -- hence the 1e-3.  A factor-1000 discrepancy downstream means this
    conversion was skipped somewhere.  With ``IO_IN_PHYSICAL_UNITS=false`` the
    columns are Henon units and none of this applies.

    Rows are one per integration *substep*, not per snapshot, so ``t`` is
    irregular and will not equal snapshot times: interpolate with
    :func:`tidal_radius_at` rather than float-matching.
    """
    d = np.genfromtxt(radii_info_csv, delimiter=",", names=True)
    # names=True mangles the "r_0.001"-style headers; t and r_tidal survive,
    # which is all that is needed here.
    return np.atleast_1d(d["t"]), np.atleast_1d(d["r_tidal"]) * 1e-3


def tidal_radius_at(t_query_myr, t_myr, r_tidal_kpc):
    """``r_tidal`` [kpc] at an arbitrary time, linearly interpolated.

    r_tidal drifts slowly and smoothly (it tracks the shrinking bound mass), so
    linear interpolation between substeps is well inside the quantity's own
    accuracy.

    Returns ``None`` when the run has no tidal boundary: an isolated run has no
    host potential, ``set_tidal_radius`` is never called, and the member keeps
    its ``+inf`` default.  That is the coherent value, but it must not be
    propagated into a division, so it is surfaced as ``None`` here.
    """
    t_myr = np.asarray(t_myr, dtype=float)
    r_tidal_kpc = np.asarray(r_tidal_kpc, dtype=float)
    good = np.isfinite(t_myr) & np.isfinite(r_tidal_kpc) & (r_tidal_kpc > 0.0)
    if not good.any():
        return None
    value = float(np.interp(float(t_query_myr), t_myr[good], r_tidal_kpc[good]))
    return value if np.isfinite(value) and value > 0.0 else None


def tidal_radius_for_run(run_dir, time_myr):
    """KRIOS's tabulated ``r_tidal`` [kpc] for ``run_dir`` at ``time_myr``.

    Convenience wrapper over :func:`load_tidal_radius` / :func:`tidal_radius_at`
    so call sites need one line and cannot get the pc->kpc conversion wrong.
    Returns ``None`` if the run has no ``radii_info`` series or no finite tidal
    boundary (an isolated run), which callers should read as "fall back to
    solving for L1".
    """
    matches = sorted(glob.glob(os.path.join(run_dir, "radii_info_*.csv")))
    if not matches:
        return None
    try:
        t_myr, r_tidal_kpc = load_tidal_radius(matches[0])
    except (OSError, ValueError, KeyError):
        return None
    return tidal_radius_at(time_myr, t_myr, r_tidal_kpc)


def escape_reference(r_gc, v_gc, m_cluster, ell=None):
    """The escape zero-point for one cluster state.

    Returns ``(phi_l1, ell_l1, omega_vec)``: the effective potential at L1,
    which :func:`jacobi_energy` subtracts so that E_J = 0 is the escape
    threshold; the cluster-to-L1 distance; and the rotating frame's Omega.

    ``ell``: cluster-to-L1 distance in kpc.  ``None`` (the default, and the
    historical behaviour) re-derives it from the point-mass force balance in
    :func:`l1_distance`.  Pass KRIOS's tabulated ``r_tidal`` -- via
    :func:`load_tidal_radius` / :func:`tidal_radius_at` -- to use the run's own
    Pfeffer et al. (2018) boundary instead.

    Supplying ``ell`` changes TWO things, and they are not equally safe:

    1. ``ell_l1``, the radius the figures divide by.  Cosmetic and
       self-consistent -- it relabels r/r_L1 against the run's own boundary.
    2. ``phi_l1``, and therefore the zero-point of every ``E_J`` derived from
       it: :func:`l1_effective_potential` evaluates ``-G m/ell`` and
       ``Phi_host(r_gc - ell)`` at whatever ``ell`` it is handed.  This moves
       the escape criterion itself.

    So a caller that passes ``ell`` is asserting that escape should be
    referenced to a Pfeffer boundary rather than a force-balance L1.  Those are
    different definitions that agree closely (~0.05%) rather than the same
    quantity; if only (1) is wanted, take ``ell_l1`` from here for the physics
    and use the tabulated radius purely as a plotting normalisation.
    """
    r_gc = np.asarray(r_gc, dtype=float)
    omega_vec = orbital_frequency(r_gc, v_gc)
    omega_mag = float(np.linalg.norm(omega_vec))
    radius = float(np.linalg.norm(r_gc))

    phi_l1, ell_l1 = l1_effective_potential(radius, m_cluster, omega_mag,
                                            ell=ell)
    return phi_l1, ell_l1, omega_vec


def effective_potential(pos_host, phi_cluster, omega_vec):
    """Phi_eff at host-centred positions, given the cluster potential there.

    ``phi_cluster`` is the *specific* cluster potential at each position, in
    (km/s)^2 -- from KRIOS's SCF column for snapshots, or from the IC's own
    self-potential.  Using the real cluster potential rather than a point mass
    matters inside the cluster, which is where most of the particles are.
    """
    return (galaxy_potential(np.linalg.norm(pos_host, axis=-1))
            + np.asarray(phi_cluster, dtype=float)
            + centrifugal_potential(pos_host, omega_vec))


def rotating_frame_velocity(vel_inertial, pos_host, omega_vec):
    """Velocity in the co-rotating xyz frame: ``v_inertial - Omega x r``.

    This is the ``v`` of the paper's Eq. (2) -- the speed in the frame Phi_eff
    is static in.  It is not the same as the cluster-centric velocity
    ``v - v_SCF``: for a circular orbit the two differ by ``Omega x r_cc``,
    which is ~0.2 km/s at the median cluster radius against a ~5 km/s
    dispersion, but grows to a few km/s out at the tidal boundary where the
    escape question is actually decided.
    """
    return (np.asarray(vel_inertial, dtype=float)
            - np.cross(omega_vec, np.asarray(pos_host, dtype=float)))


def jacobi_energy(pos_host, vel_inertial, phi_cluster, omega_vec, phi_l1):
    """E_J = v^2/2 + Phi_eff(r) - Phi_eff(L1), in (km/s)^2.

    ``phi_l1`` is the L1 effective potential from :func:`escape_reference`.
    Subtracting it is part of the definition, not a normalisation applied
    afterwards: it puts the escape threshold at E_J = 0, so the sign of the
    return value *is* the escape criterion, and the magnitude of a negative one
    is the energy per unit mass the star still needs to get out through L1.

    ``vel_inertial`` is the velocity in the fixed frame; it is converted to the
    rotating frame here, so callers pass the raw snapshot velocities.
    """
    v_rot = rotating_frame_velocity(vel_inertial, pos_host, omega_vec)
    kinetic = 0.5 * np.sum(v_rot**2, axis=-1)
    return (kinetic + effective_potential(pos_host, phi_cluster, omega_vec)
            - phi_l1)


def circular_frequency(r_gal):
    """Omega for a circular orbit at ``r_gal`` kpc, in (km/s)/kpc.

    Used by the IC script, where the cluster is placed on a circular orbit
    rather than read from a run's SCF frame.
    """
    return np.sqrt(NEWTON_G * galaxy_enclosed_mass(r_gal) / r_gal**3)
