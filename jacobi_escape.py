#!/usr/bin/env python3
"""Restricted-three-body escape machinery for an *extended* spherical host.

Shared by ``initial_energy_distributions_jacobi.py`` and
``plot_distance_vs_energy_jacobi.py``, the restricted-three-body counterparts of
the cluster-only scripts of the same names.  Where those test a star's
cluster-frame energy against an empirically calibrated boundary, these ask the
dynamical question directly -- can the star reach the inner Lagrange point? --
and answer it from the Jacobi integral.  No empirical constant, and no
leading-order stand-in for one, appears anywhere below: every scale here is a
difference of Phi_eff values evaluated at points that are solved for.

Coordinates
-----------
The KRIOS convention, shared with ``host_potential``, ``critical_spin`` and
``generate_ics_for_spinning_clusters``: **primed for the host, unprimed for the
cluster.**

    r'              host-centred.  The host sits at r' = 0 and the cluster
                    centre at r'_GC = (R, 0, 0), R = |r'_GC| the galactocentric
                    radius.  ``pos_host`` in every signature below is r'.
    r = r' - r'_GC  cluster-centred.  The cluster sits at r = 0, and these are
                    the (x, y, z) of TIRO's perturbation a_pert.

Which of the two a quantity is written in is not bookkeeping here.  Phi_host is
a function of |r'| and Phi_GC of |r|, so far so obvious; the one that actually
bites is the centrifugal term, which is a function of r' and *not* of r.  See
below.

Units
-----
Everything is in the snapshot's units: kpc, km/s, Msun, so
G = 4.3009173e-6 kpc (km/s)^2 / Msun -- astropy's CODATA value, shared with
every other G in this project -- and potentials come out in (km/s)^2.

The rotating frame
------------------
The frame turns with the cluster's instantaneous orbital angular velocity about
the host,

    Omega = (r'_GC x v_GC) / |r'_GC|^2,

which for a circular orbit is exact and for an eccentric one is the osculating
value -- E_Jacobi is then conserved only to the extent that Omega is slowly
varying.  That is the standard restricted-three-body caveat and it is the main
approximation in what follows.

Velocities are converted to the co-rotating xyz frame, v_rot = v - Omega x r',
which is the ``v`` of the paper's Eq. (2) -- the frame Phi_eff is static in.
Callers pass raw inertial (snapshot) velocities and ``rotating_frame_velocity``
does the conversion.

The centrifugal term
--------------------
Binney & Tremaine (2008) Sec. 3.3.2, eqs. (3.112)-(3.114): in a frame rotating
at angular velocity Omega_b a star conserves the Jacobi integral

    E_J = (1/2)|v|^2 + Phi_eff(x),   Phi_eff(x) = Phi(x) - (1/2)|Omega_b x x|^2,

with **x measured from a point on the rotation axis**.  That last clause is the
entire content of the primed/unprimed distinction.  The axis of this frame runs
through the *host* centre -- that is what the cluster orbits -- so the argument
of the centrifugal term is r', giving

    Phi_eff(r') = Phi_host(|r'|) + Phi_GC(|r' - r'_GC|) - (1/2)|Omega x r'|^2.

Writing -(1/2)|Omega x r|^2 with the unprimed r is not the same potential
shifted by a constant; it is a frame rotating about an axis through the
*cluster*.  With Omega along z and r'_GC = (R, 0, 0) the two differ by

    -(1/2)|Omega x r'|^2 = -(1/2)|Omega x r|^2 - Omega^2 R x - (1/2)Omega^2 R^2,

and the linear piece -Omega^2 R x is precisely the pseudo-potential of the
cluster's own orbital acceleration -- the term that cancels Phi_host'(R) x and
makes the cluster centre a free-fall origin.  Drop it and a uniform residual
force Omega^2 R survives, some 36 times the tidal force nu Omega^2 r_J at the
saddle for the 5 kpc reference case, and the collinear points stop being
anywhere near symmetric.

Two centrifugal terms exist in this project and they are about different axes.
This module carries only the *orbital* one, at Omega about the host axis: KRIOS
integrates in the inertial frame and its snapshots are inertial, so a cluster's
own figure rotation lives in the particle velocities rather than in a potential.
The other is the cluster's *spin*, at omega^2 = (1 + zeta) Omega^2 about the
cluster axis, and it appears only in TIRO's perturbation, as the
-zeta (x^2 + y^2) term written in unprimed coordinates (see
``generate_ics_for_spinning_clusters.calc_external_pe_numba`` and
``assert_l1_saddle_exists``).  At zeta = 0 the cluster is synchronous, the two
axes turn at the same rate, and the distinction collapses.

Coriolis has no place in Phi_eff for the reason BT08 give in the same section:
it acts perpendicular to v and does no work, so it drops out of the Jacobi
integral -- which is not to say it drops out of the orbits.

The criterion
-------------
The Jacobi energy of a star is measured from the L1 saddle at r'_1,

    E_J = (1/2)|v_rot|^2 + Phi_eff(r') - Phi_eff(r'_1).

A star's Jacobi constant C = (1/2)|v_rot|^2 + Phi_eff(r') is conserved, and the
zero-velocity surface it defines opens at L1, so the star can leave the cluster
exactly when C >= Phi_eff(r'_1).  That is the whole criterion:

    E_J > 0   the star can reach L1 and escape,
    E_J < 0   it is trapped inside a closed zero-velocity surface,

with no normalisation in between: for a bound star |E_J| is the energy per unit
mass it still needs to get out, in (km/s)^2.

Referencing to Phi_eff(r'_1) is also what makes the arithmetic survive double
precision: Phi_eff(L1) ~ -1.4e5 (km/s)^2 at 5 kpc is almost entirely the host,
while every difference that matters is of order 10 (km/s)^2.  Subtracting it
cancels that common offset.

The tidal radius follows from the same solve as r_t = |r'_1 - r'_GC|, the
cluster-to-L1 distance -- i.e. |r| at the saddle, the unprimed radius of L1.

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

The reference level A, and E_tilde
----------------------------------
The dimensionless Jacobi integral is Weatherford et al. (2024),

    E_tilde = (E_J - E_crit)/|E_crit|,
    E_J     = (1/2)|v_rot|^2 + Phi_eff(r') - A,
    E_crit  = Phi_eff(r'_L1) - A,
    A       = max[Phi_eff(z' = 0)],

with the maximum taken over the orbital plane through the host.  The anchors are

    E_tilde = 0   E_J = E_crit: the star can reach L1 and leave,
    E_tilde = 1   E_J = 0:      it can reach the A level itself.

**How A is evaluated here.**  The usual shortcut is to assert that the maximum
sits at the equilateral L4/L5 points and to evaluate Phi_eff there.  Sometimes
that is right and sometimes it is not, and which case you are in depends on the
host, so :func:`plane_potential_maximum` does not assume a location at all: it
grids Phi_eff over the z' = 0 plane and refines the best cell with a local
optimiser.  Brute force costs about a second, it is the definition rather than a
proxy for it, and it cannot be wrong about a geometry it never assumed.

**Where the maximum turns out to be** depends on one thing: whether the host
recoils.  The host's displacement from the rotation axis produces a tangential
force of order G m/d^2 -- the same order as the cluster's own, and independent
of the host mass (:func:`barycentre_fraction` has the arithmetic).  So this is
not a detail that washes out for small mass ratios; it decides whether there are
any off-axis equilibria at all.

  *Host pinned to the rotation axis* (``m_host_total=None``, the default, and
  the right choice for an extended live halo, which deforms rather than
  translating).  Phi_host and the centrifugal term are then both axisymmetric
  about the origin and contribute nothing azimuthally, so the only tangential
  force is the cluster's and it vanishes only at phi = 0, pi.  **Every**
  stationary point of Phi_eff on the plane is collinear.  Since Phi_eff ->
  -infinity at the cluster and (for nu > 0) outward, the interior maximum is one
  of L1, L2, L3, and L3 -- roughly 2R from the cluster rather than r_J -- is the
  highest.  The search returns phi_max = 180 degrees, matching the independent
  1-D solve of :func:`l3_radius` to ~1e-10 (km/s)^2, and

      A = Phi_eff(L3) = U(R) - G m/(2R),

  with U(R) = Phi_h(R) - (1/2)Omega^2 R^2 the corotation level
  (:func:`corotation_potential`).

  *Host recoiling* (``m_host_total`` given; the right choice for a point-mass
  host, which has nowhere else to put the momentum).  This is the textbook
  restricted three-body problem and the triangular points are back: the search
  returns the **exact equilateral L4**, reproducing its angle to ~1e-6 degrees
  and its radius to nine significant figures at every mass ratio from
  m/M = 1e-5 to 0.5.  For small m/M that is phi_max -> 60 degrees and
  A = Phi_eff(L4), the reference now sitting a distance R from the cluster
  rather than 2R.  No tidy A-minus-U(R) form is quoted for this case, because
  U(R) is itself measured from the rotation axis and so moves with mu as well;
  the two shifts are the same order and the difference is not the clean
  G m/(2R) of the pinned case.  E_crit is exact regardless -- it is
  Phi_eff(L1) - A with both terms solved for, and ``python -m jacobi_escape``
  checks that identity rather than any of these analytic forms.

Getting this wrong is a ~1% error in E_crit, all of it in A -- L1 is unmoved,
being displaced from the cluster by r_J rather than by O(m/M) R.  Note that the
closed form of :func:`triangular_point_angle` tracks the brute force only in the
small-m limit; the search does not depend on it, which is the point.

Either way,

    E_crit = Phi_eff(L1) - A,

which is the *only* energy scale this module defines.  Both terms in it are
solved for -- the saddle by the exact force balance of :func:`l1_distance`, the
maximum by the plane search -- so E_crit needs no expansion, no calibration and
no assumption about L4/L5, and there is deliberately no leading-order stand-in
offered as an alternative to normalise by.  The analytic forms above are here to
make the number interpretable and to give :func:`_report` something to check
against; nothing computes with them.

Do **not** normalise by |Phi_eff(L1)| itself: at ~1.4e5 (km/s)^2 it is almost
entirely the host potential, so it carries the arbitrary zero point of Phi_h and
differs between orbital radii for reasons unrelated to escape.  A is safe from
that because it enters only as a difference, in E_J and E_crit alike.
"""

import glob
import os

import numpy as np
from astropy import units as u
from astropy.constants import G as _G_astropy
from scipy.optimize import brentq, minimize

# kpc (km/s)^2 / Msun -- positions in kpc, velocities in km/s, masses in Msun.
#
# Taken from astropy's CODATA constant rather than written out as a literal, so
# that every G in this project is the same number.  It used to be the rounded
# 4.30091e-6, which differs from CODATA by 1.7e-6 relative -- nothing on its own,
# but ``generate_ics_for_spinning_clusters`` builds Phi_GC with astropy's G and
# adds it to a Phi_host and an A built with this one, and a single Phi_eff
# should not be assembled from two different gravitational constants.  With them
# unified, that module's round-trip check against the helpers here comes back at
# machine zero instead of 5e-4 (km/s)^2.
#
# astropy is imported here for this constant alone; the module is otherwise
# float-only and does not need units at runtime.
NEWTON_G = float(
    _G_astropy.to_value(u.kpc * (u.km / u.s)**2 / u.Msun))

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
# Hessian agree to 1e-14 over 0.2-100 kpc, and the two Omega agree to 1e-15 --
# they used to differ by 2e-6, which was this module's rounded G against
# astropy's, and is gone now that NEWTON_G is taken from astropy.
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


def barycentre_fraction(m_cluster, m_host_total=None):
    """mu = m/(M_tot + m), the fraction of the separation the host sits off-axis.

    ``m_host_total`` is the host mass that carries the system's *inertia*, which
    for an extended host is not the M(<R) that sets the orbital frequency --
    that distinction is the theme of this module.  ``None`` means infinite, i.e.
    the host is pinned to the rotation axis and mu = 0.

    Which to pass is a physical question with an O(1) consequence, not a
    refinement.  The host's displacement produces a tangential force
    G M_tot (mu d)/d^3 = G m/d^2 for M_tot >> m -- **independent of M_tot**, and
    exactly the order of the cluster's own tangential pull.  So fixing the host
    is not a small approximation off the host-cluster axis: it removes one of
    the two competing terms entirely, and with it the triangular points.

      * Point-mass host (``KeplerHost``, nu = 3): it recoils, because a point
        mass has nowhere else to put the momentum.  Pass ``m_host_total = M``.
        This is the textbook restricted three-body problem, L4/L5 exist at 60
        degrees and A sits there.
      * Extended live halo (``SphericalMW2022Host``): leave it ``None``.  Not
        because mu is small -- the argument above shows that does not save you
        -- but because a halo does not translate rigidly at all; it deforms and
        exerts dynamical friction.  And even granting rigid recoil, the
        triangular points need (M_tot + m)/M(<R) <= 8 to exist
        (:func:`triangular_point_angle`), which for this host means R > 16 kpc.
        Below that they are absent either way and A sits at L3.

    Getting this wrong moves E_crit by about 1%: for a 2e5 Msun cluster at
    5 kpc around a point-mass host, the barycentric E_crit is -18.346 (km/s)^2
    against -18.518 fixed, a 0.94% error that lives entirely in A -- L1 comes
    out at 69.245 pc either way.
    """
    if m_host_total is None or not np.isfinite(m_host_total):
        return 0.0
    return float(m_cluster) / (float(m_host_total) + float(m_cluster))


def _resolve_host(phi_host, mass_host):
    """Fill in the built-in MW2022 host for whichever callable is None.

    Every function below that needs the host takes ``phi_host`` (Phi_host(r) in
    (km/s)^2) and/or ``mass_host`` (M(<r) in Msun) and defaults to
    :func:`galaxy_potential` / :func:`galaxy_enclosed_mass`.  Both must be
    vectorised over a numpy array of radii in kpc.

    The point of the hook is that nothing in the Lagrange-point or A machinery
    is specific to MilkyWayPotential2022 -- it needs a spherical Phi(r) and the
    M(<r) that goes with it, and nothing else.  Passing a different pair covers
    any host, and with it any nu: nu = 3 - dlnM/dlnR is a property of
    ``mass_host`` alone, so a caller wanting nu = 3 passes a point mass, nu = 1
    a flat rotation curve, and so on.  ``host_potential.py``'s gala hosts are
    wrapped this way by the IC generator, which is what makes ``--host kepler``
    come out Keplerian here instead of silently Milky Way.

    Pass the two together or not at all: a Phi and an M(<r) that are not each
    other's derivative-integral pair describe no potential, and the collinear
    solves would balance forces from one against a potential from the other.
    """
    if (phi_host is None) != (mass_host is None):
        raise ValueError(
            "pass phi_host and mass_host together or neither: Phi_host(r) and "
            "M_host(<r) must describe the same potential, since the Lagrange "
            "solves use the mass and the energies use the potential.")
    if phi_host is None:
        return galaxy_potential, galaxy_enclosed_mass
    return phi_host, mass_host


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
    ``(r'_GC x v_GC) / |r'_GC|^2``.  ``r_gc`` is the cluster's *primed*
    (host-centred) position, which is what makes this Omega the angular velocity
    about the host axis rather than about the cluster.  Exact for a circular
    orbit; osculating otherwise.
    """
    r_gc = np.asarray(r_gc, dtype=float)
    v_gc = np.asarray(v_gc, dtype=float)
    return np.cross(r_gc, v_gc) / float(np.dot(r_gc, r_gc))


def centrifugal_potential(pos, omega_vec):
    """-(1/2)|Omega x r'|^2 in (km/s)^2, for primed ``pos`` (..., 3).

    The centrifugal part of Binney & Tremaine (2008) eq. (3.114), and the one
    term in Phi_eff whose argument must be the *primed*, host-centred position:
    the rotation axis of this frame passes through the host, and BT's x is
    measured from a point on that axis.  Passing cluster-centred positions here
    silently rotates the frame about the wrong axis -- see "The centrifugal
    term" in the module docstring for what that costs.

    Only the orbital rotation Omega enters.  A cluster's own spin omega is not
    represented by any potential in this module; it lives in the inertial
    particle velocities that :func:`rotating_frame_velocity` is handed.
    """
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


def l1_distance(r_gc_mag,
                m_cluster,
                omega_mag,
                phi_host=None,
                mass_host=None,
                m_host_total=None):
    """Distance from the cluster centre to L1 (the inner saddle), in kpc.

    L1 sits on the host-cluster line where the net radial force in the rotating
    frame vanishes.  Writing ``s`` for the primed (host-centred) radius |r'| and
    ``ell = r_gc_mag - s`` for the unprimed distance |r| from the cluster,

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

    _, mass_host = _resolve_host(phi_host, mass_host)
    mu = barycentre_fraction(m_cluster, m_host_total)

    def force_balance(ell):
        # s is the host-to-point distance, which is separation minus
        # cluster-to-point whatever the origin.  Only the centrifugal arm knows
        # about the barycentre: it is measured from the rotation axis, which
        # sits mu * r_gc_mag on the far side of the host.
        s = r_gc_mag - ell
        x_from_axis = (1.0 - mu) * r_gc_mag - ell
        return (NEWTON_G * float(mass_host(s)) / s**2
                - NEWTON_G * m_cluster / ell**2
                - omega_mag**2 * x_from_axis)

    # Seeded with a Jacobi-radius-sized guess; _collinear_root brackets outward
    # from it, so a seed that is wrong for an exotic host costs iterations and
    # not correctness.
    return _collinear_root(force_balance, r_gc_mag,
                           0.1 * r_gc_mag)


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


def l3_radius(r_gc_mag, m_cluster, omega_mag, phi_host=None, mass_host=None):
    """Host-centred radius of L3 [kpc], on the far side of the host.

    Solves Omega^2 r = Phi_h'(r) + G m/(R + r)^2, which to leading order in m
    gives r = R [1 + (1/4)(r_J/R)^3].

    Verified against that expansion: the solved and expanded radii agree to 10
    significant figures at R = 3, 5 and 10 kpc.

    L3 is where the reference level A lives.  With the host on the rotation
    axis every stationary point of Phi_eff on the z' = 0 plane is collinear (see
    the module docstring), and L3 is the highest of the three, so
    A = max[Phi_eff(z' = 0)] = Phi_eff(L3).  This function is therefore the
    independent check on :func:`plane_potential_maximum`: a 1-D root solve
    against a 2-D grid search, agreeing to ~1e-10 (km/s)^2 over R = 1-20 kpc and
    m = 2e5-2e10 Msun.  The brute force stays the definition -- it assumes
    nothing about where the maximum is -- and this stays the check.

    Caveat, unchanged by that promotion: L3 sits at essentially the same radius
    as the cluster, so its position is sensitive to convention -- specifically
    to whether the host is allowed to recoil, which :func:`l1_distance` and this
    function both decline to let it do.  The coefficient shifts with that
    choice.  L1/L2 are insensitive because they are displaced from the cluster
    by r_J, not by O(m/M) R.  A inherits that sensitivity, but only through the
    G m/(2R) term, which is 1% of |E_crit| -- and E_crit is a *difference* of
    Phi_eff values, so the host's own zero point cancels out of it entirely.
    """
    if not (r_gc_mag > 0.0 and m_cluster > 0.0 and omega_mag > 0.0):
        raise ValueError("r_gc_mag, m_cluster and omega_mag must be positive")

    _, mass_host = _resolve_host(phi_host, mass_host)

    def force_balance(r):
        return (omega_mag**2 * r
                - NEWTON_G * float(mass_host(r)) / r**2
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

    Two caveats, which together are why :func:`plane_potential_maximum` brute-
    forces A instead of calling this.  First, the formula assumes the host
    recoils rigidly: with the host pinned to the rotation axis, as is the
    default there and the right choice for a live halo, there are no off-axis
    equilibria whatsoever and "the L4/L5 points" would be coordinates that are
    not stationary points of Phi_eff at all.  Second, even where they do exist
    this closed form is a small-m approximation -- checked against the exact
    equilateral geometry for a point-mass host it is right to 3e-4 degrees at
    m/M = 1e-5 but 9 degrees out by m/M = 0.5, where the true L4 is still
    exactly equilateral with the two masses.

    Nor is rigid recoil a description of a real halo, which deforms and exerts
    dynamical friction rather than translating -- the triangular points are not
    sharply defined in any live galactic setting, which is why the
    tidal-stripping literature uses the L1/L2 result exclusively.
    """
    if m_host_total is None:
        m_host_total = galaxy_total_mass()
    ratio = (float(m_host_total) + float(m_cluster)) / float(
        galaxy_enclosed_mass(r_gc_mag))
    if not np.isfinite(ratio) or ratio > 8.0:
        return None
    return 2.0 * float(np.arcsin(0.5 * ratio**(1.0 / 3.0)))


def l1_effective_potential(r_gc_mag,
                           m_cluster,
                           omega_mag,
                           ell=None,
                           phi_host=None,
                           mass_host=None,
                           m_host_total=None):
    """(Phi_eff(L1) in (km/s)^2, distance from the cluster to L1 in kpc).

    The cluster is treated as a point mass here: L1 lies at roughly the tidal
    radius, outside essentially all of the bound mass, so Phi_GC there is
    Keplerian to the same accuracy the criterion itself has.

    Note the two radii in the one expression, and that they are not
    interchangeable: ``s = r_gc_mag - ell`` is the primed |r'_1| and carries the
    host and centrifugal terms, while ``ell`` is the unprimed |r_1| and carries
    the cluster term.
    """
    phi_host, mass_host = _resolve_host(phi_host, mass_host)
    mu = barycentre_fraction(m_cluster, m_host_total)
    if ell is None:
        ell = l1_distance(r_gc_mag,
                          m_cluster,
                          omega_mag,
                          phi_host=phi_host,
                          mass_host=mass_host,
                          m_host_total=m_host_total)
    s = r_gc_mag - ell
    x_from_axis = (1.0 - mu) * r_gc_mag - ell
    phi = (float(phi_host(s)) - NEWTON_G * m_cluster / ell
           - 0.5 * omega_mag**2 * x_from_axis**2)
    return phi, ell


def corotation_potential(r_gc_mag,
                         omega_mag,
                         phi_host=None,
                         mass_host=None,
                         m_cluster=0.0,
                         m_host_total=None):
    """U(R) = Phi_h(R) - (1/2) Omega^2 R^2 in (km/s)^2 -- the corotation level.

    Phi_eff evaluated at |r'| = R with the cluster's own term left out: the
    host-plus-centrifugal effective potential on the cluster's own
    guiding-centre circle.  Dropping the cluster term is what makes it a
    function of the primed radius alone, hence constant around the circle.

    This is *not* the reference level: that is A = max[Phi_eff(z' = 0)], which
    :func:`plane_potential_maximum` brute-forces.  U(R) is the analytic anchor
    that makes A interpretable, and the two are related exactly by the cluster's
    own potential at the maximum,

        A = Phi_eff(L3) = U(R) - G m/(2R),

    L3 lying a distance 2R from the cluster.  The equilateral assumption would
    have put the reference a distance R away instead, giving U(R) - G m/R, so
    the three candidate levels are ordered U(R) > A > A_equilateral and span
    G m/R -- 1.1% of |E_crit| at 5 kpc.

    Why U(R) is the right anchor to quote:

      * Circularity makes R a stationary point of U: U'(R) = Phi'(R) - Omega^2 R
        = 0, with U''(R) = Phi'' - Omega^2 = -nu Omega^2 < 0.  So U(R) is a
        genuine maximum -- the corotation ridge, constant around the circle.
      * It is the circle the triangular points sit on *when they exist*: the
        general-host treatment forces r = R exactly, for any spherical host.
      * It is the common leading-order level of L3, L4 and L5, which differ from
        it only by O(G m/R) cluster terms.  It is what survives when the angle
        -- the piece that is undefined if the triangular points do not exist --
        is dropped, which is what makes it well posed for any host.

    Still exported in its own right because it carries no dependence on
    m_cluster and none on the host-recoil convention that
    :func:`l3_radius` documents, so it is the stable thing to compare orbital
    radii against.
    """
    phi_host, _ = _resolve_host(phi_host, mass_host)
    mu = barycentre_fraction(m_cluster, m_host_total)
    return (float(phi_host(r_gc_mag))
            - 0.5 * omega_mag**2 * ((1.0 - mu) * r_gc_mag)**2)


def plane_potential_maximum(r_gc_mag,
                            m_cluster,
                            omega_mag,
                            phi_host=None,
                            mass_host=None,
                            m_host_total=None,
                            n_radial=1200,
                            n_azimuth=901,
                            r_outer=4.0,
                            max_expansions=12):
    """``(A, r_max, phi_max)``: A = max[Phi_eff(z' = 0)], by brute force.

    The reference level of Weatherford et al. (2024) Eq. (5), evaluated without
    assuming where the maximum is.  ``A`` is in (km/s)^2; ``r_max`` is the
    host-centred radius |r'| of the maximum in kpc and ``phi_max`` its azimuth
    from the host-cluster axis in radians, both returned so the caller can check
    *where* the answer came from rather than only what it was.

    Method: evaluate Phi_eff on a log-spaced radial by uniform azimuthal grid
    over the half-plane (Phi_eff is symmetric in phi -> -phi), take the best
    cell, then refine it with Nelder-Mead in Cartesian (x', y').

    Why not just use the equilateral points.  Because they are not equilibria of
    this Phi_eff -- see :func:`triangular_point_angle`, and the module docstring
    -- so evaluating there would be reading a number off a position with no
    dynamical meaning.  Brute force costs about a second, which is nothing next
    to the snapshot loop that follows, and it cannot be wrong about a geometry
    it never assumed.

    Any host, and hence any nu
    --------------------------
    ``phi_host`` and ``mass_host`` (see :func:`_resolve_host`) select the host;
    the default is this module's MilkyWayPotential2022, where nu runs 1.6-2.1.
    Nothing in the method is specific to it.  Verified against power-law hosts
    M(<r) ~ r^(3-nu) over nu = 0.5 to 3 and against a point mass (nu = 3): the
    maximum is L3 at phi = pi in every case, matching :func:`l3_radius` to ~1e-9
    (km/s)^2 relative.

    The one case with no answer is nu <= 0 -- a homogeneous or hollow core, where
    the host's attraction and the centrifugal term cancel to quadratic order and
    Phi_eff has no interior maximum at all; its supremum is approached only as
    the cluster's own -G m/|r| dies away at large separation.  That is the same
    degeneracy that makes r_J diverge (:func:`jacobi_radius` returns ``inf``)
    and removes L1.  Rather than return whatever sits on the edge of the grid,
    the search notices that the maximum has run to the outer boundary, expands
    the grid, and after ``max_expansions`` doublings raises.  A caller seeing
    that error should check nu before checking this function.

    Where the maximum lands, and what checks it
    -------------------------------------------
    ``m_host_total`` decides, via :func:`barycentre_fraction`, and the two
    answers are genuinely different points rather than a small displacement:

    ``None`` (default), host pinned to the rotation axis
        Phi_host and the centrifugal term are both axisymmetric about the
        origin, so neither contributes azimuthally and the only azimuthal force
        is the cluster's, vanishing at phi = 0, pi.  Every stationary point of
        Phi_eff on the plane is collinear, for any spherical host whatsoever,
        and the highest is L3.  Returns phi_max = pi, checkable against the
        independent 1-D solve of :func:`l3_radius` (~1e-10 (km/s)^2 over
        R = 1-20 kpc, m = 2e5-2e10 Msun, and nu = 0.5-3).
    ``m_host_total`` given, host recoiling about the barycentre
        The textbook restricted three-body problem.  The host now does exert an
        azimuthal force, of the same order as the cluster's, and the triangular
        points exist: returns the **exact equilateral L4**, verified to ~1e-6
        degrees in angle and nine significant figures in radius against the
        equilateral construction, for m/M from 1e-5 to 0.5.  For small m/M that
        is phi_max -> 60 degrees.

    So a phi_max of pi with a finite ``m_host_total``, or of 60 degrees with
    ``None``, means the geometry and the recoil argument have come apart and
    something upstream is inconsistent.

    :func:`triangular_point_angle` gives a closed form for the second case, but
    it agrees with this search only for small m/M -- by m/M ~ 0.5 it is 9
    degrees out.  The search does not use it, which is the point of brute
    force.
    """
    if not (r_gc_mag > 0.0 and m_cluster > 0.0 and omega_mag > 0.0):
        raise ValueError("r_gc_mag, m_cluster and omega_mag must be positive")
    phi_host, mass_host = _resolve_host(phi_host, mass_host)

    # Origin on the rotation axis.  With m_host_total given that is the
    # barycentre and the host sits mu * r_gc_mag behind it; with it None the
    # host is the origin, mu = 0, and this reduces to the pinned-host case.
    # Which is right is a physical choice with an O(1) effect off-axis -- see
    # :func:`barycentre_fraction`.
    mu = barycentre_fraction(m_cluster, m_host_total)
    x_host = -mu * r_gc_mag
    x_cluster = (1.0 - mu) * r_gc_mag

    def phi_eff(x, y):
        to_host = np.maximum(np.hypot(x - x_host, y), 1e-12)
        to_cluster = np.maximum(np.hypot(x - x_cluster, y), 1e-12)
        return (phi_host(to_host) - NEWTON_G * m_cluster / to_cluster
                - 0.5 * omega_mag**2 * (x**2 + y**2))

    azimuths = np.linspace(0.0, np.pi, int(n_azimuth))
    r_hi = float(r_outer) * r_gc_mag
    for _ in range(int(max_expansions) + 1):
        radii = np.geomspace(0.02 * r_gc_mag, r_hi, int(n_radial))
        r_grid, phi_grid = np.meshgrid(radii, azimuths, indexing="ij")
        values = phi_eff(r_grid * np.cos(phi_grid), r_grid * np.sin(phi_grid))
        best = np.unravel_index(np.nanargmax(values), values.shape)
        # Running to the outer edge means there is no interior maximum inside
        # this grid: either the grid is too small or nu <= 0 and there is none
        # at any size.  Either way the fix is to look further out first.
        if best[0] < len(radii) - 1:
            break
        r_hi *= 2.0
    else:
        raise RuntimeError(
            f"max[Phi_eff(z' = 0)] ran to the outer edge of the search grid "
            f"out to {r_hi:g} kpc for a cluster of {m_cluster:g} Msun at "
            f"{r_gc_mag:g} kpc.  Phi_eff has no interior maximum on the plane, "
            f"which is the nu <= 0 case: inside a homogeneous core the host's "
            f"attraction and the centrifugal term cancel to quadratic order, "
            f"there is no L1 either (jacobi_radius returns inf), and the "
            f"satellite is tidally protected rather than stripped.  Check nu "
            f"at this radius before looking any further at this function.")

    result = minimize(lambda q: -float(phi_eff(q[0], q[1])),
                      [float(r_grid[best] * np.cos(phi_grid[best])),
                       float(r_grid[best] * np.sin(phi_grid[best]))],
                      method="Nelder-Mead",
                      options={
                          "xatol": 1e-12,
                          "fatol": 1e-14,
                          "maxiter": 20000
                      })
    x, y = float(result.x[0]), float(result.x[1])
    return -float(result.fun), float(np.hypot(x, y)), float(np.arctan2(y, x))


def critical_energy(r_gc_mag,
                    m_cluster,
                    omega_mag,
                    ell=None,
                    phi_host=None,
                    mass_host=None,
                    m_host_total=None):
    """``(E_crit, A, phi_l1, ell_l1)`` for one cluster configuration.

    Weatherford et al. (2024) Eq. (4), E_crit = Phi_eff(r'_L1) - A, together
    with the pieces it is built from so a caller can reference E_J to the same
    A (Eq. 2) without recomputing anything:

        ``E_crit``   the critical energy, negative, in (km/s)^2.
        ``A``        max[Phi_eff(z' = 0)] from :func:`plane_potential_maximum`.
        ``phi_l1``   Phi_eff at the L1 saddle, in (km/s)^2.
        ``ell_l1``   cluster-to-L1 distance in kpc.

    ``ell`` is passed through to :func:`l1_effective_potential`; see
    :func:`escape_reference` for what supplying it changes and why it is not a
    free choice.

    E_crit < 0 always, since L1 is a saddle below the plane's maximum.  A
    non-negative return would mean the L1 solve and the plane search have found
    inconsistent configurations and is worth treating as an error upstream
    rather than propagating into a division.
    """
    phi_host, mass_host = _resolve_host(phi_host, mass_host)
    phi_l1, ell_l1 = l1_effective_potential(r_gc_mag,
                                            m_cluster,
                                            omega_mag,
                                            ell=ell,
                                            phi_host=phi_host,
                                            mass_host=mass_host,
                                            m_host_total=m_host_total)
    a_ref, _, _ = plane_potential_maximum(r_gc_mag,
                                          m_cluster,
                                          omega_mag,
                                          phi_host=phi_host,
                                          mass_host=mass_host,
                                          m_host_total=m_host_total)
    return phi_l1 - a_ref, a_ref, phi_l1, ell_l1


def escape_energy_scale(phi_l1,
                        r_gc_mag,
                        m_cluster,
                        omega_mag,
                        phi_host=None,
                        mass_host=None):
    """|E_crit| = |Phi_eff(L1) - A| in (km/s)^2: the escape energy unit.

    The depth of the L1 saddle below the maximum of Phi_eff on the orbital
    plane, and the denominator of Weatherford et al. (2024) Eq. (3) -- the
    energy per unit mass a star sitting at the A level would have to give up to
    be trapped inside the L1 neck.

    Both terms are solved for, neither is expanded: Phi_eff(L1) comes from the
    exact radial force balance of :func:`l1_distance` against the true M(<r')
    profile, and A from the brute-forced plane search of
    :func:`plane_potential_maximum`.  That is the whole definition, and it is
    the only energy scale this module offers -- there is deliberately no
    leading-order stand-in to normalise by instead, because every such stand-in
    (a leading-order saddle depth, an empirically calibrated boundary constant,
    and so on) is a different number that merely happens to be close, and a
    figure whose axis is one of them is not a figure of E_tilde.

    For orientation only, never for use: the scale is of order G m/r_J, a few
    tens of (km/s)^2 for a 2e5 Msun cluster in the inner Galaxy.  A computed
    |E_crit| orders away from that means the inputs are wrong, not the
    definition.

    :func:`critical_energy` returns the signed E_crit together with the A and
    Phi_eff(L1) it was built from, which is what a caller referencing E_J to the
    same A needs; this function is the convenience wrapper for the magnitude.
    """
    a_ref, _, _ = plane_potential_maximum(r_gc_mag,
                                          m_cluster,
                                          omega_mag,
                                          phi_host=phi_host,
                                          mass_host=mass_host)
    return abs(float(phi_l1) - a_ref)


def load_tidal_radius(radii_info_csv):
    """``(t_Myr, r_tidal_kpc)`` from a KRIOS ``radii_info_*`` time series.

    KRIOS's own tidal radius, from ``System::compute_tidal_radius``: the
    fixed-point solve of ``r_t = (G M_bound(<r_t) / lambda1_eff)^(1/3)`` with
    ``lambda1_eff = lambda1 - (lambda2 + lambda3)/2``, the Pfeffer et al. (2018)
    effective eigenvalue of the *host* tidal tensor.  That is a third route to
    the same scale: for a spherical host the leading eigenvalue of the effective
    tidal tensor is nu Omega^2, so lambda1_eff plays the role of the
    Omega^2 - Phi'' in :func:`jacobi_radius`.  It is not the
    restricted-three-body result, though.  The ``-(lambda2+lambda3)/2`` term is
    how the construction recovers the centrifugal contribution without being
    told Omega: for a circular orbit in a *spherical* host the two transverse
    eigenvalues are both -Phi'(R)/R = -Omega^2, so -(lambda2+lambda3)/2 = Omega^2
    and lambda1_eff = -Phi'' + Omega^2 = nu Omega^2 exactly, reproducing the
    centrifugal term rather than approximating it.  Away from sphericity or
    circularity it becomes the proxy it is usually described as.  Either way it
    is built from the host tensor alone, so it is blind to the cluster's own
    shape.  It agrees with :func:`l1_distance` to ~0.05%.

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
    """Phi_eff at primed positions r', given the cluster potential there.

        Phi_eff(r') = Phi_host(|r'|) + Phi_GC(|r|) - (1/2)|Omega x r'|^2

    -- primed argument for the host and centrifugal terms, unprimed for the
    cluster, exactly as in the module docstring.  ``pos_host`` is r'; the
    unprimed r never appears explicitly because ``phi_cluster`` arrives already
    evaluated at it.

    ``phi_cluster`` is the *specific* cluster potential at each position, in
    (km/s)^2 -- from KRIOS's SCF column for snapshots, or from the IC's own
    self-potential.  Using the real cluster potential rather than a point mass
    matters inside the cluster, which is where most of the particles are.
    """
    return (galaxy_potential(np.linalg.norm(pos_host, axis=-1))
            + np.asarray(phi_cluster, dtype=float)
            + centrifugal_potential(pos_host, omega_vec))


def rotating_frame_velocity(vel_inertial, pos_host, omega_vec):
    """Velocity in the co-rotating xyz frame: ``v_inertial - Omega x r'``.

    Primed again, and for the same reason the centrifugal term is: the frame
    turns about the host axis, so the transport velocity of the frame at a point
    is Omega x r' and not Omega x r.

    This is the ``v`` of the paper's Eq. (2) -- the speed in the frame Phi_eff
    is static in.  It is not the same as the cluster-centric velocity
    ``v - v_SCF``: for a circular orbit the two differ by ``Omega x r_cc``,
    which is ~0.2 km/s at the median cluster radius against a ~5 km/s
    dispersion, but grows to a few km/s out at the tidal boundary where the
    escape question is actually decided.
    """
    return (np.asarray(vel_inertial, dtype=float)
            - np.cross(omega_vec, np.asarray(pos_host, dtype=float)))


def jacobi_energy(pos_host, vel_inertial, phi_cluster, omega_vec, phi_ref):
    """E_J = (1/2)|v_rot|^2 + Phi_eff(r') - phi_ref, in (km/s)^2.

    ``phi_ref`` is the level E_J is measured from, and there are exactly two
    sensible choices.  They differ by E_crit and are trivially interconvertible,
    but they are not interchangeable and mixing them costs exactly the width of
    the conventional E_tilde >= 1 cut:

    ``A`` from :func:`plane_potential_maximum` or :func:`critical_energy`
        Weatherford et al. (2024) Eq. (2).  Pair with :func:`tilde_energy`.  The
        escape threshold sits at E_J = E_crit, not at zero.
    ``phi_l1`` from :func:`escape_reference`
        The L1-referenced form.  Pair with :func:`tilde_energy_from_l1`.  Puts
        the threshold at E_J = 0, so the sign of the return value *is* the
        escape criterion and the magnitude of a negative one is the energy per
        unit mass the star still needs to get out through L1.

    Either way the subtraction is part of the definition rather than a
    normalisation applied afterwards, and either way it cancels the ~1.4e5
    (km/s)^2 of host potential that would otherwise swamp the double precision.

    ``vel_inertial`` is the velocity in the fixed frame; it is converted to the
    rotating frame here, so callers pass the raw snapshot velocities.
    """
    v_rot = rotating_frame_velocity(vel_inertial, pos_host, omega_vec)
    kinetic = 0.5 * np.sum(v_rot**2, axis=-1)
    return (kinetic + effective_potential(pos_host, phi_cluster, omega_vec)
            - phi_ref)


def tilde_energy(e_jacobi_a, e_crit):
    """E_tilde = (E_J - E_crit)/|E_crit|, Weatherford et al. (2024) Eq. (3).

    ``e_jacobi_a`` must be the **A-referenced** Jacobi energy of Eq. (2),
    (1/2)|v_rot|^2 + Phi_eff(r') - A, i.e. :func:`jacobi_energy` called with the
    A from :func:`critical_energy` or :func:`plane_potential_maximum`.
    ``e_crit`` is Phi_eff(L1) - A, negative.

    Passing the L1-referenced E_J instead yields E_tilde - 1: the two zero
    points differ by exactly E_crit.  That is the worst possible size of error
    -- exactly the width of the conventional E_tilde >= 1 cut -- and it will not
    look obviously wrong.  Use :func:`tilde_energy_from_l1` for that input.

    Anchors:
        E_tilde = 0   C = Phi_eff(L1): the star can reach L1 and leave.
        E_tilde = 1   C = A: it can reach the maximum of Phi_eff on the orbital
                      plane, and is no longer confined to the neighbourhood of
                      the orbit at all.
    """
    return (np.asarray(e_jacobi_a, dtype=float) - e_crit) / abs(e_crit)


def tilde_energy_from_l1(e_jacobi_l1, e_crit):
    """The same E_tilde from the L1-referenced E_J of :func:`jacobi_energy`.

    Since E_J^A = E_J^L1 + E_crit, Eq. (3) collapses to a plain rescaling,

        E_tilde = E_J^L1 / |E_crit|,

    which is why the zero point is unaffected by the choice of A -- only the
    unit of the axis depends on it, through |E_crit|.  Brute-forcing A rather
    than assuming equilateral L4/L5 therefore rescales E_tilde by ~1% and moves
    no threshold.  Verified equal to :func:`tilde_energy` to
    machine zero.  ``e_crit`` may be passed as the signed value or as the
    positive scale from :func:`escape_energy_scale`; only its magnitude is used.
    """
    return np.asarray(e_jacobi_l1, dtype=float) / abs(e_crit)


def circular_frequency(r_gal, phi_host=None, mass_host=None):
    """Omega for a circular orbit at ``r_gal`` kpc, in (km/s)/kpc.

    Used by the IC script, where the cluster is placed on a circular orbit
    rather than read from a run's SCF frame.
    """
    _, mass_host = _resolve_host(phi_host, mass_host)
    return np.sqrt(NEWTON_G * mass_host(r_gal) / r_gal**3)


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

    # Where the brute-forced A actually lands, and how it compares with the two
    # analytic levels.  phi_max = 180 deg and A == Phi_eff(L3) are the two
    # claims the module docstring makes about plane_potential_maximum.
    header = ("  R    A - U(R)   -Gm/(2R)   A - Phi(L3)   phi_max   "
              "r_max/R   equilateral")
    print("\n" + header)
    print(f"{'kpc':>4}  {'(km/s)^2':>8} {'(km/s)^2':>10} {'(km/s)^2':>13} "
          f"{'deg':>9} {'':>9} {'A_eq - U(R)':>13}")
    print("-" * len(header))
    for r in radii:
        omega = float(circular_frequency(r))
        a_ref, r_max, phi_max = plane_potential_maximum(r, m_cluster, omega)
        u_r = corotation_potential(r, omega)
        phi_l3 = (float(galaxy_potential(l3_radius(r, m_cluster, omega)))
                  - NEWTON_G * m_cluster / (r + l3_radius(r, m_cluster, omega))
                  - 0.5 * omega**2 * l3_radius(r, m_cluster, omega)**2)
        print(f"{r:5.1f} {a_ref - u_r:9.4f} "
              f"{-NEWTON_G * m_cluster / (2.0 * r):10.4f} "
              f"{a_ref - phi_l3:13.2e} {np.degrees(phi_max):9.4f} "
              f"{r_max / r:9.6f} "
              f"{-NEWTON_G * m_cluster / r:13.4f}")
    print("\nColumns 2 and 3 are the claim A = U(R) - Gm/(2R); column 4 is the "
          "brute-forced\nmaximum against the independent l3_radius solve.  "
          "phi_max = 180 deg says every\nstationary point of Phi_eff on the "
          "plane is collinear, as it must be with the\nhost on the rotation "
          "axis.  The last column is what assuming equilateral L4/L5\nwould "
          "have given instead -- twice as far below U(R), because L3 is 2R from "
          "the\ncluster and the equilateral points would be R.")

    header = "  R    Phi_eff(L1)         A          E_crit      r_J/R"
    print("\n" + header)
    print(f"{'kpc':>4}  {'(km/s)^2':>12} {'(km/s)^2':>12} {'(km/s)^2':>11}")
    print("-" * len(header))
    for r in radii:
        omega = float(circular_frequency(r))
        e_crit, a_ref, phi_l1, _ = critical_energy(r, m_cluster, omega)
        print(f"{r:5.1f} {phi_l1:14.4f} {a_ref:13.4f} {e_crit:11.4f} "
              f"{jacobi_radius(r, m_cluster) / r:10.5f}")
    print("\nE_crit = Phi_eff(L1) - A is the only energy scale this module "
          "defines, and\nboth terms in it are solved for rather than "
          "approximated.  It is the unit of\nE_tilde and the denominator of "
          "Weatherford et al. (2024) Eq. (3).")


if __name__ == "__main__":
    _report()
