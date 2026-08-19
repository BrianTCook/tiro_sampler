#!/usr/bin/env python3
"""Restricted-three-body escape machinery for an *extended* spherical host.

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

Why the host being extended changes the answer
----------------------------------------------
Every departure from the textbook Kepler result traces back to one fact: for an
extended host the mass that sets the orbital frequency and the mass that carries
the host's inertia are different numbers,

    M(<R)  !=  M_tot.

Omega^2 R = Phi_h'(R) = G M(<R)/R^2 is set by the *enclosed* mass; the host's
recoil about the barycentre is set by the *total* mass.  Kepler is the
degenerate case where they coincide, and several familiar results are artifacts
of that degeneracy.

**L1/L2.** Expanding the host's attraction about R and balancing it against the
cluster's pull gives the generalised Jacobi radius

    r_J^3 = G m / (Omega^2 - Phi_h''(R))
          = G m / (4 Omega^2 - kappa^2)
          = R^3 m / [ (3 - dlnM/dlnR) M(<R) ],

i.e. Binney & Tremaine eq. (8.91).  The familiar factor of 3 is really
3 - dlnM/dlnR = 3(1 - rho/rho_bar), which is exactly TIRO's nu (see
``host_potential.nu``, and :func:`nu_tidal` here) -- the same number, arrived at
from the tidal tensor rather than the mass profile.  For the spherical
MilkyWayPotential2022 used below nu is 1.6-1.9 over 3-20 kpc, so the naive
Keplerian (m/3M)^(1/3) *understates* the tidal radius by ~20%: at R = 5 kpc with
m = 2e5 Msun it gives 69.6 pc against the correct 85.2 pc.

Nothing in the criterion actually uses that expansion.  :func:`l1_distance`
solves the exact radial force balance against the true M(<r) profile, so
Phi_eff(r_1) is the real saddle of this Phi_eff and not a leading-order stand-in;
:func:`jacobi_radius` is provided for comparison and for the analytic scalings.

A limiting case worth knowing: nu -> 0 for a homogeneous core (dlnM/dlnR -> 3),
where r_J diverges.  That is physical, not a bug -- solid-body rotation has no
shear, the radial tidal tensor vanishes, and there is no L1 at this order.
Inside a constant-density core a satellite is tidally *protected*.
:func:`jacobi_radius` returns ``inf`` there rather than pretending otherwise.

**L1 vs L2.** They are equidistant only at leading order; the asymmetry enters
at O(r_J/R) through Phi_h'''(R), *not* at O(m/M_host) as previously claimed
here.  For the MW host with m = 2e5 Msun that is a real but small effect --
r_J/R = 1.7% at 5 kpc, and the solved offset (L2 - L1)/r_J is 0.40% there
(0.64% at 3 kpc, 0.23% at 10 kpc), a few tenths of a parsec.  Using L1 alone
remains the right simplification, now for the right reason.
:func:`l2_distance` solves the outer saddle exactly if the difference is ever
wanted.

**L4/L5 do not exist here.**  With the host centre held rigidly fixed the
tangential component of grad Phi_eff = 0 reduces to sin(theta) G m / s^3 = 0,
whose only roots are theta = 0, pi: there are no off-axis equilibria at all.
Letting the host recoil rigidly about the barycentre -- the consistent
treatment, since the barycentre offset is O(m) just as the cluster's transverse
pull is -- puts them back on the orbital circle at

    sin(theta/2) = (1/2) [ (M_tot + m) / M(<R) ]^(1/3),

which is equilateral only when M_tot = M(<R), and which has **no solution at all**
once (M_tot + m)/M(<R) > 8.  The NFW halo's mass diverges logarithmically, and
even truncating it at 200 kpc the condition demands R > 16.2 kpc; every cluster
in these runs sits at 3-10 kpc, where M_tot/M(<R) is 38 and 15.  So for this host
the triangular points are absent, and were the orbit wide enough for them to
exist they would not be equilateral.  A live halo would not admit them in any
case (it deforms and exerts dynamical friction rather than translating rigidly).

The energy scale, and what replaces A
-------------------------------------
Earlier versions of this module subtracted A = max[Phi_eff(z = 0)], *asserted* to
sit at equilateral L4/L5 points.  Those coordinates are not equilibria of this
Phi_eff -- they are not equilibria of any extended-host Phi_eff -- so A was a
number read off a dynamically meaningless position.

The zero point does not need A at all: E_J here is measured from Phi_eff(L1), so
the escape threshold is fixed at zero by construction and the unit is (km/s)^2.
A only ever set the *scale*, because it cancels from the numerator of the
normalised coordinate:

    E_tilde = (E_J^A - E_crit)/|E_crit| = E_J^L1 / |E_crit|.

The right scale is the corotation level U(R) = Phi_h(R) - (1/2)Omega^2 R^2 (see
:func:`corotation_potential`), giving

    E_crit = Phi_eff(L1) - U(R) = -(3/2) G m / r_J,

with the 3/2 universal rather than Keplerian -- nu cancels.  U(R) is a genuine
stationary level of the rotating frame, needs no equilateral assumption, and is
the circle L4/L5 lie on whenever they exist.  See :func:`escape_energy_scale`
and :func:`tilde_energy`.

Do **not** normalise by |Phi_eff(L1)| itself: at ~1.4e5 (km/s)^2 it is almost
entirely the host potential, so it carries the arbitrary zero point of Phi_h and
differs between orbital radii for reasons unrelated to escape.
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
# compare against it if gala is ever updated.  The check is cheap: nu computed
# here from the mass profile and host_potential.nu computed by gala from the
# Hessian agree to 1e-14 over 0.2-100 kpc, and the two Omega^2 to 2e-6 (the
# residual is the differing G).
#
# NFW: ``NFW_M`` is the characteristic mass M_s = 4 pi rho_s r_s^3, gala's ``m``
# parameter, so that
#   Phi(r)  = -G M_s ln(1 + r/r_s) / r          (gala sphericalnfw_value)
#   M(<r)   =  M_s [ln(1 + r/r_s) - (r/r_s)/(1 + r/r_s)]
#   rho(r)  =  M_s / (4 pi r (r + r_s)^2)
NFW_M = 5.5427e11
NFW_RS = 15.626
# Hernquist components as (m [Msun], c [kpc]): bulge, then nucleus.
#   Phi(r)  = -G m / (r + c)
#   M(<r)   =  m r^2 / (r + c)^2
#   rho(r)  =  m c / (2 pi r (r + c)^3)
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


def galaxy_density(r):
    """rho_host(r) in Msun/kpc^3, the same three components.

    Only needed for the logarithmic mass slope, but it is the honest way to get
    it: dlnM/dlnR = 4 pi r^3 rho / M(<r) uses the analytic derivative of the
    mass profile rather than a finite difference of it.
    """
    r = np.maximum(np.asarray(r, dtype=float), 1e-12)
    rho = NFW_M / (4.0 * np.pi * r * (r + NFW_RS)**2)
    for m, c in HERNQUIST_COMPONENTS:
        rho = rho + m * c / (2.0 * np.pi * r * (r + c)**3)
    return rho


def galaxy_total_mass(r_trunc=None):
    """M_tot of the host in Msun -- ``inf`` unless a truncation radius is given.

    The NFW mass diverges logarithmically, so the host as written here has no
    finite total mass.  That is not a technicality to be waved away: M_tot is
    what sets the host's recoil about the barycentre, and it is the quantity
    (not M(<R)) that decides whether the triangular Lagrange points exist --
    see :func:`triangular_point_angle`.  Pass ``r_trunc`` [kpc] to evaluate the
    condition for a halo truncated somewhere sensible.
    """
    if r_trunc is None:
        return np.inf
    return float(galaxy_enclosed_mass(r_trunc))


def logarithmic_mass_slope(r):
    """dlnM/dlnR = 4 pi r^3 rho(r) / M(<r) = 3 rho / rho_bar, dimensionless.

    0 outside all the mass (Kepler), 2 for a singular isothermal sphere, 3 for a
    homogeneous core.  It is the single number by which an extended host departs
    from a point mass in the tidal problem.
    """
    r = np.maximum(np.asarray(r, dtype=float), 1e-12)
    return 4.0 * np.pi * r**3 * galaxy_density(r) / galaxy_enclosed_mass(r)


def nu_tidal(r):
    """nu = 3 - dlnM/dlnR, the generalised Jacobi denominator.

    Identically TIRO's galactic potential coefficient (``host_potential.nu``,
    defined there as 1 - R Phi''/Phi'): both equal (Omega^2 - Phi'')/Omega^2,
    one written from the mass profile and one from gala's Hessian.  They agree
    to 1e-14 over 0.2-100 kpc -- the same number by two independent routes, so
    a disagreement means one of the two potentials has drifted.

    3 for a point mass, 1 for a flat rotation curve, 0 for a homogeneous core --
    where the radial tidal field vanishes and L1/L2 cease to exist.
    """
    return 3.0 - logarithmic_mass_slope(r)


def galaxy_potential_curvature(r):
    """Phi_host''(r) in (km/s)^2/kpc^2.

    From Phi' = G M(<r)/r^2, so Phi'' = Omega^2 (dlnM/dlnR - 2) and
    Omega^2 - Phi'' = nu Omega^2 -- the denominator of the Jacobi radius.
    """
    r = np.maximum(np.asarray(r, dtype=float), 1e-12)
    omega_sq = NEWTON_G * galaxy_enclosed_mass(r) / r**3
    return omega_sq * (logarithmic_mass_slope(r) - 2.0)


def epicyclic_frequency(r):
    """kappa in (km/s)/kpc, for a circular orbit at ``r`` kpc.

    kappa^2 = R dOmega^2/dR + 4 Omega^2 = (1 + dlnM/dlnR) Omega^2, so the Jacobi
    denominator can equivalently be written 4 Omega^2 - kappa^2 = nu Omega^2.
    Provided because that form is the one that appears in the epicyclic
    literature, and because kappa/Omega is the cleanest statement of how much
    shear the orbit actually sees.
    """
    r = np.maximum(np.asarray(r, dtype=float), 1e-12)
    omega_sq = NEWTON_G * galaxy_enclosed_mass(r) / r**3
    return np.sqrt((1.0 + logarithmic_mass_slope(r)) * omega_sq)


def jacobi_radius(r_gal, m_cluster):
    """Leading-order Jacobi radius r_J [kpc] for a circular orbit at ``r_gal``.

        r_J^3 = G m / (Omega^2 - Phi'') = R^3 m / (nu M(<R)),   nu = 3 - dlnM/dlnR

    the generalised (Binney & Tremaine eq. 8.91) tidal radius.  This is the
    analytic scaling, valid to O(r_J/R); the criterion in this module does not
    use it, and :func:`l1_distance` solves the balance exactly instead (they
    agree to ~0.2% for the MW host at 3-10 kpc).  Use this for scalings, for
    seeding a bracket, and for the comparison with the Keplerian (m/3M)^(1/3).

    Returns ``inf`` where nu <= 0 -- a homogeneous or hollow core, in which the
    radial tidal field does not shear and there is no L1 at this order.  The
    satellite is tidally protected there; the divergence is the physics, not a
    failure of the formula.
    """
    r_gal = float(r_gal)
    nu = float(nu_tidal(r_gal))
    if nu <= 0.0:
        return np.inf
    return r_gal * (m_cluster / (nu * float(galaxy_enclosed_mass(r_gal))))**(
        1.0 / 3.0)


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


def _collinear_root(force_balance, r_gc_mag, guess):
    """Bracket ``force_balance`` from ``guess`` outward and brentq the root.

    Shared by the L1 and L2 solves: both are one-sided, both have a root of
    order the Jacobi radius, and both are singular at the ends of their range.
    """
    lo = 1e-8 * r_gc_mag
    hi = min(max(guess, 10.0 * lo), 0.5 * r_gc_mag)
    for _ in range(80):
        if force_balance(hi) > 0.0:
            break
        if hi >= 0.999999 * r_gc_mag:
            break
        hi = min(hi * 1.5, 0.999999 * r_gc_mag)
    if not force_balance(hi) > 0.0:
        raise RuntimeError(
            f"could not bracket a collinear Lagrange point out to "
            f"{hi:g} kpc of a cluster at {r_gc_mag:g} kpc: the radial force "
            f"balance does not change sign.  For a host with nu <= 0 near this "
            f"radius (a homogeneous core) that is the correct answer -- the "
            f"saddle does not exist -- rather than a solver failure.")
    return float(brentq(force_balance, lo, hi, xtol=1e-14, rtol=1e-14))


def l1_distance(r_gc_mag, m_cluster, omega_mag):
    """Distance from the cluster centre to L1 (the inner saddle), in kpc.

    L1 sits on the host-cluster line where the net radial force in the rotating
    frame vanishes.  Writing ``s`` for the host-centred radius and
    ``ell = r_gc_mag - s`` for the distance from the cluster,

        F(ell) = G M_host(<s)/s^2 - G m_cluster/ell^2 - Omega^2 s

    is -inf as ell -> 0 (the cluster's own pull diverges) and positive as
    ell -> r_gc_mag (the host term survives while the others vanish), so it has
    exactly one root, which brentq finds.

    This is the *exact* balance against the true M(<s) profile, not the O(r_J/R)
    expansion of :func:`jacobi_radius`: the enclosed mass is re-evaluated at the
    trial radius, so no assumption about the host's mass distribution enters.
    The bracket is seeded with r_J because that is the right scale, but nothing
    downstream depends on the seed.

    The host centre is held fixed, which for the collinear points is an
    O(m/M_tot) ~ 1e-5 effect.  It is emphatically *not* negligible for the
    triangular points, which is why they are handled separately and why they do
    not exist for this host at all -- see :func:`triangular_point_angle`.
    """
    if not (r_gc_mag > 0.0 and m_cluster > 0.0 and omega_mag > 0.0):
        raise ValueError("r_gc_mag, m_cluster and omega_mag must be positive")

    def force_balance(ell):
        s = r_gc_mag - ell
        return (NEWTON_G * float(galaxy_enclosed_mass(s)) / s**2
                - NEWTON_G * m_cluster / ell**2
                - omega_mag**2 * s)

    return _collinear_root(force_balance, r_gc_mag,
                           jacobi_radius(r_gc_mag, m_cluster))


def l2_distance(r_gc_mag, m_cluster, omega_mag):
    """Distance from the cluster centre to L2 (the outer saddle), in kpc.

    The mirror of :func:`l1_distance` with ``s = r_gc_mag + ell``,

        F(ell) = Omega^2 s - G M_host(<s)/s^2 - G m_cluster/ell^2,

    negative at small ell and positive well outside.

    L1 and L2 are equidistant only at leading order.  The asymmetry is
    O(r_J/R), entering through Phi_h'''(R): for the MW host and m = 2e5 Msun,
    (L2 - L1)/r_J is 0.64%, 0.40% and 0.23% at R = 3, 5 and 10 kpc -- a few
    tenths of a parsec, far below the accuracy of the escape criterion itself.
    Provided so that the claim can be checked rather than asserted; the
    criterion uses L1, the lower of the two saddles.
    """
    if not (r_gc_mag > 0.0 and m_cluster > 0.0 and omega_mag > 0.0):
        raise ValueError("r_gc_mag, m_cluster and omega_mag must be positive")

    def force_balance(ell):
        s = r_gc_mag + ell
        return (omega_mag**2 * s
                - NEWTON_G * float(galaxy_enclosed_mass(s)) / s**2
                - NEWTON_G * m_cluster / ell**2)

    return _collinear_root(force_balance, r_gc_mag,
                           jacobi_radius(r_gc_mag, m_cluster))


def l3_radius(r_gc_mag, m_cluster, omega_mag):
    """Host-centred radius of L3 [kpc], on the far side of the host.

    Solves Omega^2 r = Phi_h'(r) + G m/(R + r)^2, which to leading order in m
    gives r = R [1 + (1/4)(r_J/R)^3].

    Verified against that expansion: the solved and expanded radii agree to 10
    significant figures at R = 3, 5 and 10 kpc.

    Caveat, and it is the reason this is not used anywhere: L3 sits at
    essentially the same radius as the cluster, so the answer is sensitive to
    convention -- specifically to whether the host is allowed to recoil, which
    :func:`l1_distance` and this function both decline to let it do.  The
    coefficient shifts with that choice.  L1/L2 are insensitive to it because
    they are displaced from the cluster by r_J, not by O(m/M) R.
    """
    if not (r_gc_mag > 0.0 and m_cluster > 0.0 and omega_mag > 0.0):
        raise ValueError("r_gc_mag, m_cluster and omega_mag must be positive")

    def force_balance(r):
        return (omega_mag**2 * r
                - NEWTON_G * float(galaxy_enclosed_mass(r)) / r**2
                - NEWTON_G * m_cluster / (r_gc_mag + r)**2)

    lo, hi = 0.5 * r_gc_mag, 2.0 * r_gc_mag
    if not (force_balance(lo) < 0.0 < force_balance(hi)):
        raise RuntimeError(
            f"could not bracket L3 on [{lo:g}, {hi:g}] kpc for a cluster at "
            f"{r_gc_mag:g} kpc.")
    return float(brentq(force_balance, lo, hi, xtol=1e-14, rtol=1e-14))


def triangular_point_angle(r_gc_mag, m_cluster, m_host_total=None):
    """Angular separation of L4/L5 from the cluster [rad], or ``None``.

    For a rigidly recoiling host of *total* mass M_tot the triangular equilibria
    lie on the cluster's own orbital circle at

        sin(theta/2) = (1/2) [ (M_tot + m) / M(<R) ]^(1/3),

    which returns theta = 60 degrees only in the Keplerian case M_tot = M(<R).
    Otherwise they sit at larger separation, and once

        (M_tot + m) / M(<R) > 8

    the equation has no solution and the triangular points do not exist at all.
    ``None`` is returned in that case -- the honest answer, not an error.

    ``m_host_total`` defaults to :func:`galaxy_total_mass`, i.e. ``inf`` for the
    NFW halo as written, so the default result is ``None``.  Pass a truncated
    mass to test a specific halo: even at r_trunc = 200 kpc the condition needs
    R > 16.2 kpc, while these runs use 3-10 kpc.

    Two things this function is *not*.  It is not a route to a reference energy:
    with the host centre fixed, as everywhere else in this module, there are no
    off-axis equilibria whatsoever, so evaluating Phi_eff at "the L4/L5 points"
    means evaluating it at coordinates that are not stationary points of it.
    And it is not a description of a real halo, which deforms and exerts
    dynamical friction rather than translating rigidly -- the triangular points
    are not sharply defined in any live galactic setting, which is why the
    tidal-stripping literature uses the L1/L2 result exclusively.
    """
    if m_host_total is None:
        m_host_total = galaxy_total_mass()
    ratio = (float(m_host_total) + float(m_cluster)) / float(
        galaxy_enclosed_mass(r_gc_mag))
    if not np.isfinite(ratio) or ratio > 8.0:
        return None
    return 2.0 * float(np.arcsin(0.5 * ratio**(1.0 / 3.0)))


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


def corotation_potential(r_gc_mag, omega_mag):
    """U(R) = Phi_h(R) - (1/2) Omega^2 R^2 in (km/s)^2 -- the corotation level.

    The host-plus-centrifugal effective potential on the cluster's own
    guiding-centre circle, with the cluster's own term left out.  This is the
    replacement for the old L4/L5 reference ``A``, and it is the right one for
    an extended host:

      * Circularity makes R a stationary point of U: U'(R) = Phi'(R) - Omega^2 R
        = 0, with U''(R) = Phi'' - Omega^2 = -nu Omega^2 < 0.  So U(R) is a
        genuine maximum -- the corotation ridge, constant around the circle.
      * It is the circle the triangular points sit on *when they exist*: the
        notes' general-host treatment forces r = R exactly, for any spherical
        host.  The old equilateral A got the circle right and the angle wrong,
        and the angle enters only through the cluster's own -G m/|x - r_GC|
        term -- precisely the piece that is undefined when the points do not
        exist.  Dropping that term is what makes U(R) well posed.
      * It is the common leading-order level of L3, L4 and L5, which differ
        from it only by O(G m/R) cluster terms.  Preferable to Phi_eff(L3) as a
        definition because L3 carries the host-recoil convention sensitivity
        documented in :func:`l3_radius`, and U(R) carries none.

    Numerically, the old equilateral A was exactly U(R) - G m/R for any
    spherical host, so switching to U(R) shifts the energy scale by the
    fraction (2/3)(r_J/R) -- 1.4%, 1.1%, 0.8% at R = 3, 5, 10 kpc.
    """
    return (float(galaxy_potential(r_gc_mag))
            - 0.5 * omega_mag**2 * r_gc_mag**2)


def escape_energy_scale(phi_l1, r_gc_mag, omega_mag):
    """|E_crit| = |Phi_eff(L1) - U(R)| in (km/s)^2: the escape energy unit.

    The depth of the L1 saddle below corotation, and the natural unit of the
    escape problem -- the energy per unit mass a star at the corotation level
    would have to give up to be trapped inside the L1 neck.

    Expanding U about R and using r_J^3 = G m/(nu Omega^2),

        Phi_eff(L1) = U(R) - (1/2) nu Omega^2 r_J^2 - G m/r_J
                    = U(R) - (3/2) G m / r_J,

    so |E_crit| = (3/2) G m / r_J to O(r_J/R).  The nu cancels between the two
    terms, which is why the 3/2 is universal rather than Keplerian: it holds for
    any spherical host.  Verified against the exact solve to 0.1% at 3-10 kpc.

    This is the denominator E_tilde should be normalised by.  |Phi_eff(L1)|
    itself must NOT be used: it is ~1.4e5 (km/s)^2, almost entirely the host
    potential, so it carries the arbitrary zero-point of Phi_h and varies
    between orbital radii for reasons that have nothing to do with escape.
    """
    return abs(float(phi_l1) - corotation_potential(r_gc_mag, omega_mag))


def load_tidal_radius(radii_info_csv):
    """``(t_Myr, r_tidal_kpc)`` from a KRIOS ``radii_info_*`` time series.

    KRIOS's own tidal radius, from ``System::compute_tidal_radius``: the
    fixed-point solve of ``r_t = (G M_bound(<r_t) / lambda1_eff)^(1/3)`` with
    ``lambda1_eff = lambda1 - (lambda2 + lambda3)/2``, the Pfeffer et al. (2018)
    effective eigenvalue of the *host* tidal tensor.  That is a third route to
    the same scale: for a spherical host the leading eigenvalue of the effective
    tidal tensor is nu Omega^2, so lambda1_eff plays the role of the
    Omega^2 - Phi'' in :func:`jacobi_radius`.  It is not the
    restricted-three-body result, though -- the ``-(lambda2+lambda3)/2``
    construction folds a rotational contribution into the eigenvalues as a proxy
    for the centrifugal term, and it is built from the host tensor alone, so it
    is blind to the cluster's own shape.  It agrees with :func:`l1_distance` to
    ~0.05%.

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


def tilde_energy(e_jacobi_a, e_crit):
    """E_tilde = (E_J - E_crit)/|E_crit|, the paper's Eq. (3).

    ``e_jacobi_a`` must be the **A-referenced** Jacobi energy,
    v^2/2 + Phi_eff(r) - U(R).  ``e_crit`` is Phi_eff(L1) - U(R), negative.

    Passing the L1-referenced E_J that :func:`jacobi_energy` returns instead
    yields E_tilde - 1: the two zero points differ by exactly E_crit.  That is
    the worst possible size of error -- exactly the width of the conventional
    E_tilde >= 1 cut -- and it will not look obviously wrong.  Use
    :func:`tilde_energy_from_l1` for that input.

    Anchors:
        E_tilde = 0   C = Phi_eff(L1): the star can reach L1 and leave.
        E_tilde = 1   C = U(R): it can reach the corotation circle and is no
                      longer confined to the neighbourhood of the orbit.
    """
    return (np.asarray(e_jacobi_a, dtype=float) - e_crit) / abs(e_crit)


def tilde_energy_from_l1(e_jacobi_l1, e_crit):
    """The same E_tilde from the L1-referenced E_J of :func:`jacobi_energy`.

    Since E_J^A = E_J^L1 + E_crit, Eq. (3) collapses to a plain rescaling,

        E_tilde = E_J^L1 / |E_crit|,

    which is why the zero point is unaffected by the choice of A and only the
    unit of the axis depends on it.  Verified equal to :func:`tilde_energy` to
    machine zero.  ``e_crit`` may be passed as the signed value or as the
    positive scale from :func:`escape_energy_scale`; only its magnitude is used.
    """
    return np.asarray(e_jacobi_l1, dtype=float) / abs(e_crit)


def circular_frequency(r_gal):
    """Omega for a circular orbit at ``r_gal`` kpc, in (km/s)/kpc.

    Used by the IC script, where the cluster is placed on a circular orbit
    rather than read from a run's SCF frame.
    """
    return np.sqrt(NEWTON_G * galaxy_enclosed_mass(r_gal) / r_gal**3)


def _report(m_cluster=2.0e5, radii=(1.0, 3.0, 5.0, 8.0, 10.0, 20.0)):
    """Print the Lagrange-point structure of this host, for eyeballing.

    Every claim in the module docstring is a column here, so ``python -m
    jacobi_escape`` is the way to check them rather than trust them.
    """
    print(f"Spherical MilkyWayPotential2022 host, m_cluster = {m_cluster:g} "
          f"Msun\n")
    header = ("  R      M(<R)     dlnM/dlnR   nu     r_J      L1       L2    "
              "(L2-L1)   Kepler   Mtot/M    L4/L5")
    print(header)
    print(f"{'kpc':>4}  {'Msun':>9}                      {'pc':>6}   "
          f"{'pc':>6}   {'pc':>6}    {'/r_J':>6}   {'pc':>6}  {'(200kpc)':>8}")
    print("-" * len(header))
    for r in radii:
        mass = float(galaxy_enclosed_mass(r))
        omega = float(circular_frequency(r))
        r_j = jacobi_radius(r, m_cluster)
        l1 = l1_distance(r, m_cluster, omega)
        l2 = l2_distance(r, m_cluster, omega)
        kepler = r * (m_cluster / (3.0 * mass))**(1.0 / 3.0)
        ratio = galaxy_total_mass(200.0) / mass
        theta = triangular_point_angle(r, m_cluster, galaxy_total_mass(200.0))
        theta_str = "none" if theta is None else f"{np.degrees(theta):.1f} deg"
        print(f"{r:5.1f} {mass:10.3e} {float(logarithmic_mass_slope(r)):9.4f} "
              f"{float(nu_tidal(r)):7.4f} {r_j * 1e3:7.2f} {l1 * 1e3:8.2f} "
              f"{l2 * 1e3:8.2f} {(l2 - l1) / r_j:8.4f} {kepler * 1e3:8.2f} "
              f"{ratio:8.2f} {theta_str:>9}")
    print("\nnu = 3 - dlnM/dlnR is TIRO's galactic potential coefficient; "
          "compare\nhost_potential.nu(R) (gala-derived) against the nu column.")
    print("The L4/L5 column is 'none' wherever (M_tot + m)/M(<R) > 8; with the "
          "halo\nuntruncated M_tot diverges and they never exist.")


if __name__ == "__main__":
    _report()
