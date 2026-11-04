#!/usr/bin/env python3
"""Critical spin of a TIRO model, from the Jacobi integral.

A TIRO model is bound when every star it contains sits inside a closed
zero-velocity surface.  The most energetic bound orbit is a star at rest (in the
co-rotating frame) on the cluster's own boundary, and the surface first opens at
a collinear saddle on the host-cluster axis.  So the model is critically filling
when that star's Jacobi energy is exactly zero,

    E_J = Phi_eff(boundary) - Phi_eff(L1) = 0,

which is the escape criterion of ``jacobi_escape`` (E_J > 0 escapes through L1)
applied to the model's own edge.  ``critical_zeta`` solves that equation for
zeta, and hence for the critical spin omega_crit = Omega sqrt(1 + zeta_crit).

Why this is cheap in TIRO's own variables
-----------------------------------------
TIRO's dimensionless escape potential is psi = -a (Phi_King + Phi_host), i.e.
Phi_eff = -psi/a up to the additive constant that puts psi = 0 on the boundary.
For a star at rest at radius r, therefore,

    a E_J = psi(r_L1) - psi(r),

and on the boundary psi = 0, so the whole criterion collapses to

    E_J(boundary star) = 0   <=>   psi(r_L1) = 0,

with r_L1 the root of the external solution's radial gradient along +x.  One
Model integration and one 1-D root solve per trial zeta; psi(r_L1) is monotone
increasing in zeta, so the outer solve is a bracketed brentq that cannot wander.

A note on the axis, since the sign convention differs between this module and
``jacobi_escape``.  These are unprimed (cluster-centred) coordinates -- the
KRIOS convention is primed for the host, unprimed for the cluster, so with the
host at r' = 0 and the cluster at r'_GC = (R, 0, 0) the +x here points *away*
from the galactic centre.  It makes no difference: TIRO's perturbation is even
in x, so the inner and outer saddles are exactly degenerate at quadrupole order
and "L1" names either of them.  ``jacobi_escape`` solves the same balance
against the full M(<r') profile, keeps the terms this expansion drops, and finds
the two split by O(r_J/R) -- 0.4% of r_J at 5 kpc, with L1 the lower.  That is
the number to quote when the distinction matters; here it cannot.

This replaces a 2-D fsolve for the point where the external solution and its
gradient vanish together (the "critical curve" eps_crit(zeta), inverted for
zeta).  That formulation is the same physics, but as a simultaneous solve near a
double root it needed hand-tuned seeds, could converge on the analytic
continuation at negative radius, and had no bracket to fall back on.  The two
agree: at psi = 3, nu = 1.542, eps = 4.607e-5 both give zeta_crit = 26.632, and
at nu = 1.7425, eps = 3.718e-4 both give 1.611.
"""

import numpy as np
from astropy import units as u
from astropy.constants import G as G_apy
from scipy.optimize import brentq

import tiro_path  # noqa: F401  (puts the TIRO submodule on sys.path)
from TIRO import Model


class TidallyOverfilled(ValueError):
    """No spin closes the Jacobi surface: the model is unbound even at rest.

    A statement about rho_0 and r_gal, not about spin, and a physical result
    worth plotting rather than an error to suppress -- hence its own type, so a
    scan can tell it apart from a search that merely ran out of room.
    """


class SearchLimitExceeded(ValueError):
    """zeta_crit is above ``zeta_max``: the search gave up, the model did not."""


# Starting outer limit of the L1 search, in King radii.  The scan expands past
# this when it has to: L1 sits where cluster gravity balances the tide, so it
# recedes as roughly epsilon^(-1/3), and a dense cluster on a wide orbit puts it
# far outside the ODE integration range -- epsilon = 1.5e-8 (rho_0 = 1e5
# Msun/pc^3 at 2 kpc) pushes it past 400 r_0.  Evaluating out there is legitimate
# because external_gradient is analytic in r: it is built from the constants at
# r_trunc and the tidal and centrifugal harmonics, not from the solver output.
_R_MAX = 200.0
# Hard ceiling on that expansion, so a genuinely rootless case still terminates.
_R_MAX_LIMIT = 1.0e8
# Largest zeta the bracketing will climb to before giving up.  zeta_crit runs
# as 1/epsilon once the cluster is tidally underfilled -- at r_gal = 5 kpc a
# psi = 1 model has zeta_crit = 2.2e2, 2.2e3, 2.2e4 for rho_0 = 1e3, 1e4, 1e5
# Msun/pc^3 -- so a cap of order 1e3 silently rejects ordinary dense clusters as
# "no solution".  The bracket doubles, so a high cap costs a handful of extra
# evaluations, not a proportional slowdown.
_ZETA_MAX = 1.0e7
# omega^2 = (1+zeta) Omega^2, so this is the floor of the physical zeta range.
_ZETA_MIN = -1.0


def l1_radius(model, r_max=_R_MAX, n_scan=200):
    """Radius of the L1 saddle on the +x axis, in King radii r_0.

    L1 is where the radial gradient of the external solution vanishes: the
    cluster's pull inward balances the tidal plus centrifugal pull outward.  The
    bracket is found by scanning log-spaced radii for the first sign change and
    is deliberately started well inside ``r_trunc``.

    That is the difference from ``Model.tidal_radius()``, which brackets on
    [r_trunc, 200] and raises "f(a) and f(b) must have different signs" whenever
    the root is not in there.  Fast rotation moves L1 *inward* -- at psi = 3,
    eps = 3.7e-4, zeta = 26.6 it sits at 3.67 r_0 against r_trunc = 4.70 -- so
    that bracket fails exactly in the super-critical regime a criticality search
    has to be able to evaluate.  The saddle itself never stops existing.
    """

    def gradient(r):
        return float(model.external_gradient(r, np.pi / 2, 0))

    r_lo = 1e-2 * model.r_trunc
    r_hi = float(r_max)
    while True:
        r_scan = np.geomspace(r_lo, r_hi, n_scan)
        g = np.array([gradient(r) for r in r_scan])
        crossings = np.flatnonzero(np.diff(np.sign(g)))
        if crossings.size:
            i = crossings[0]
            return float(brentq(gradient, r_scan[i], r_scan[i + 1],
                                xtol=1e-12))
        if r_hi >= _R_MAX_LIMIT:
            raise RuntimeError(
                f"no L1 saddle on [{r_lo:g}, {r_hi:g}] r_0 for TIRO parameters "
                f"{list(model.param)}: external_gradient does not change sign.")
        r_hi = min(100.0 * r_hi, _R_MAX_LIMIT)


def boundary_jacobi_energy(model, r_l1=None):
    """Jacobi energy of a star at rest on the cluster boundary, in units of 1/a.

    Equal to psi(r_L1) (see the module docstring).  Negative means the Jacobi
    surface is closed and every star in the model is bound; zero is criticality;
    positive means the psi = 0 surface has opened past the saddle, so the model's
    own edge is outside the zero-velocity surface and it starts life shedding
    stars that need neither relaxation nor tidal heating to leave.
    """
    if r_l1 is None:
        r_l1 = l1_radius(model)
    return float(model.global_solution(r_l1, np.pi / 2, 0))


def _integrated_model(psi, epsilon, zeta, nu):
    model = Model([psi, epsilon, zeta, nu])
    model.integrate()
    return model


def critical_zeta(psi,
                  epsilon,
                  nu=3.0,
                  zeta_min=_ZETA_MIN,
                  zeta_max=_ZETA_MAX,
                  xtol=1e-4):
    """zeta at which the boundary star's Jacobi energy is exactly zero.

    E_J(zeta) is monotone increasing -- more spin, more centrifugal support,
    boundary closer to a saddle that is itself moving inward -- so the root is
    unique and is found by doubling out from zeta = 0 and calling brentq.

    Returns a value in ``[zeta_min, zeta_max]``.  A negative result is physical
    and means the cluster is already super-critical when co-rotating: it has to
    spin *slower* than the orbit to hold together.  ``ValueError`` is raised only
    when even zeta = zeta_min (a non-rotating cluster, omega = 0) is
    super-critical, which is a statement about rho_0 and r_gal rather than spin.
    """

    def e_jacobi(zeta):
        return boundary_jacobi_energy(_integrated_model(psi, epsilon, zeta, nu))

    if e_jacobi(0.0) < 0.0:
        lo, hi = 0.0, 1.0
        while e_jacobi(hi) < 0.0:
            lo, hi = hi, 2.0 * hi
            if hi > zeta_max:
                raise SearchLimitExceeded(
                    f"zeta_crit is above zeta_max = {zeta_max:g} for psi = "
                    f"{psi:g}, epsilon = {epsilon:.3e}, nu = {nu:g} (E_J at "
                    f"zeta = {lo:g} is still {e_jacobi(lo):+.4f}).  The model "
                    "is bound, the bracket just ran out; raise zeta_max.")
    else:
        lo, hi = float(zeta_min), 0.0
        if e_jacobi(lo) > 0.0:
            raise TidallyOverfilled(
                f"E_J = {e_jacobi(lo):+.4f} > 0 at zeta = {lo:g} for psi = "
                f"{psi:g}, epsilon = {epsilon:.3e}, nu = {nu:g}: the model is "
                "super-critical even without rotation, so no spin makes it "
                "bound.  Raise rho_0 or move the cluster to a larger r_gal.")

    return float(brentq(e_jacobi, lo, hi, xtol=xtol))


def critical_spin(psi, epsilon, nu=3.0, **kwargs):
    """``(zeta_crit, omega_crit)`` for one (psi, epsilon, nu).

    ``omega_crit = Omega sqrt(1 + zeta_crit)`` is the largest spin the cluster
    can carry with its Jacobi surface still closed, in TIRO's dimensionless
    units (sqrt(G rho_0), the units of ``TiroInitialConditionSampler.omega``):
    Omega^2 = 4 pi epsilon there, so omega_crit = sqrt(4 pi epsilon (1 + zeta)).

    epsilon and nu are fixed by the host and the cluster's orbital radius, so for
    a given host and rho_0 this is a function of (r_gal, psi) alone -- see
    :func:`critical_spin_at_radius`.
    """
    zeta_crit = critical_zeta(psi, epsilon, nu, **kwargs)
    omega_crit = np.sqrt(4.0 * np.pi * epsilon * (1.0 + zeta_crit))
    return zeta_crit, float(omega_crit)


def tidal_strength(host, r_gal_kpc, rho_0_msun_pc3):
    """epsilon = Omega^2 / (4 pi G rho_0), the TIRO tidal strength parameter."""
    rho_0 = rho_0_msun_pc3 * u.Msun / u.pc**3
    omega2 = host.omega2(r_gal_kpc)
    return (omega2 / (4.0 * np.pi * G_apy * rho_0)).decompose().value


def critical_spin_at_radius(psi, r_gal_kpc, host, rho_0_msun_pc3, **kwargs):
    """``(zeta_crit, omega_crit)`` for a cluster of concentration ``psi`` on a
    circular orbit of radius ``r_gal_kpc``.

    The (r_gal, psi) entry point: the host fixes nu(r_gal) and Omega(r_gal),
    rho_0 turns Omega into epsilon, and everything downstream is the dimensionless
    solve above.  ``host`` is a ``host_potential`` instance.
    """
    epsilon = float(tidal_strength(host, r_gal_kpc, rho_0_msun_pc3))
    return critical_spin(psi, epsilon, float(host.nu(r_gal_kpc)), **kwargs)
