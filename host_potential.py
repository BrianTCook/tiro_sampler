#!/usr/bin/env python3
"""Host galactic potentials for the TIRO tidal models.

The TIRO formalism needs two numbers from the host potential Phi(R) at the
cluster's galactocentric radius R:

  * Omega^2(R) = Phi'(R) / R                 -- circular orbital frequency^2,
  * nu(R)      = 1 - R Phi''(R) / Phi'(R)    -- radial tidal anisotropy.

nu is the "galactic potential coefficient" (TIRO Model param[3]); it sets the
radial tidal term in the perturbation
    a_pert = (9 eps / 2) [ -zeta (x^2 + y^2) + z^2 - nu x^2 ],
and the tidal strength is eps = Omega^2 / (4 pi G rho_0). Working through the
co-rotating effective potential to quadrupole order gives nu = 1 - Phi''/Omega^2,
which reduces to nu = 3 for a Kepler point mass.

Coordinates follow the KRIOS convention throughout this project: primed for the
host potential, unprimed for the cluster potential. The host sits at r' = 0, the
cluster centre at r'_GC = (R, 0, 0), and the (x, y, z) of a_pert above are the
unprimed r = r' - r'_GC, so +x runs along the host-cluster axis away from the
galactic centre and z along the orbital angular momentum. a_pert is even in each
of x, y and z, so the two collinear saddles are degenerate at this order and the
sign of x is immaterial; the degeneracy is broken only at O(r_J/R), by the third
derivative d^3Phi/dR^3 that the quadrupole truncation drops (see
``jacobi_escape.l2_distance``).

Two rotations enter, about two different axes, and nu concerns only the first:
the orbital Omega about the host axis, which together with the tide produces the
-nu x^2 + z^2 part of a_pert, and the cluster spin omega^2 = (1 + zeta) Omega^2
about the cluster axis, which produces the -zeta (x^2 + y^2) part.
``jacobi_escape`` sets the two out in full under "The centrifugal term".

Two hosts are provided:
  * KeplerHost          -- point mass; nu = 3 exactly, Omega^2 = G M / R^3.
  * SphericalMW2022Host -- every spherical component of gala's
                           MilkyWayPotential2022: the NFW halo plus the
                           Hernquist bulge and nucleus.

Both Phi' and Phi'' come from gala's own gradient and Hessian rather than
hand-derived expressions, so adding a component is a one-line change and cannot
introduce a differentiation error.

Sphericity is a requirement, not a convenience: the expression above for nu
assumes Phi depends on |r| alone, so that the tidal tensor at the cluster is
diagonal in the (radial, tangential, vertical) frame and one number describes
it. MilkyWayPotential2022's MN3 exponential disk is its only flattened
component and is therefore the only one dropped -- inside the disk that is a
real omission, not a rounding, and the models built here are the spherical
approximation to the Milky Way rather than the Milky Way.
"""

import warnings

import numpy as np
from astropy import units as u
from astropy.constants import G as G_apy

import gala.potential as gp
from gala.units import galactic

# Point-mass host (Arditi & Varri 2025, Sec. 2.1).
KEPLER_R_SUN = 8.23 * u.kpc
KEPLER_VC_SUN = 249.44 * u.km / u.s
KEPLER_M_GALAXY = (KEPLER_VC_SUN**2.0 * KEPLER_R_SUN) / G_apy

# The one flattened component of MilkyWayPotential2022; everything else in it
# (halo, bulge, nucleus) is spherical.
MW2022_FLATTENED_COMPONENTS = ("disk", )


class _GalaHost:
    """Radial derivatives and the tidal tensor from a host's gala potential.

    Subclasses implement ``_build_gala_potential()``; everything else follows
    from gala's gradient and Hessian, so no hand-derived Phi', Phi'' are needed.
    """

    def _build_gala_potential(self):
        raise NotImplementedError

    def gala_potential(self):
        """The host's gala potential, built once and cached.

        Cached because ``compute_critical_zeta`` and the parameter-space scans
        call ``nu``/``omega2`` thousands of times, and rebuilding a composite
        potential per call dominates the runtime.
        """
        if getattr(self, "_gala_potential", None) is None:
            self._gala_potential = self._build_gala_potential()
        return self._gala_potential

    def _radial_derivatives(self, r_gal_kpc):
        """``(Phi'(R), Phi''(R))`` as Quantities, for scalar or array R [kpc].

        Evaluated on the +x' axis (the host-centred frame, cluster at
        r' = (R, 0, 0)), where for a spherical host the radial derivatives are
        the x'x' components of gala's gradient and Hessian.
        """
        r = np.atleast_1d(u.Quantity(r_gal_kpc, u.kpc))
        q = np.zeros((3, r.size)) * u.kpc
        q[0] = r
        pot = self.gala_potential()
        return np.ravel(pot.gradient(q)[0]), np.ravel(pot.hessian(q)[0, 0])

    def omega2(self, r_gal_kpc):
        """Circular orbital frequency squared at R [kpc]; returns 1/s^2.

        Scalar in, scalar out; array in, array out (the parameter-space scans
        sweep R).
        """
        r = np.atleast_1d(u.Quantity(r_gal_kpc, u.kpc))
        phi_p, _ = self._radial_derivatives(r)
        out = (phi_p / r).to(u.s**-2)
        return out if np.ndim(r_gal_kpc) else out[0]

    def nu(self, r_gal_kpc):
        """nu = 1 - R Phi''(R)/Phi'(R), the TIRO galactic potential coefficient."""
        r = np.atleast_1d(u.Quantity(r_gal_kpc, u.kpc))
        phi_p, phi_pp = self._radial_derivatives(r)
        out = (1.0 - r * phi_pp / phi_p).to_value(u.dimensionless_unscaled)
        return out if np.ndim(r_gal_kpc) else float(out[0])

    def tidal_eigenvalue(self, r_gal_kpc):
        """Largest eigenvalue lambda_1 [1/s^2] of the effective (co-rotating)
        tidal tensor at galactocentric radius R, from gala.

        Placing the cluster at r' = (R, 0, 0), the effective tidal tensor is
            T_ij = -d^2Phi/dx'_i dx'_j + Omega^2 diag(1, 1, 0),
        with Omega^2 = (dPhi/dR)/R the circular frequency. The added term is the
        curvature of the *orbital* centrifugal potential -(1/2)|Omega x r'|^2,
        hence diag(1, 1, 0): flat along the rotation axis, and about the host
        axis rather than the cluster's, so it is a property of where the cluster
        sits rather than of how fast it spins. The cluster spin does not appear
        here at all; it enters downstream as zeta.

        The largest eigenvalue sets the generalised Jacobi radius
        r_J = (G M_GC / lambda_1)^(1/3). For a spherical host the Hessian is
        diag(Phi'', Omega^2, Omega^2), so T = Omega^2 diag(nu, 0, -1) and
        lambda_1 = nu Omega^2 exactly -- which is the identity
        ``generate_ics_for_spinning_clusters.host_tidal_diagnostics`` checks by
        reporting the ratio.
        """
        pot = self.gala_potential()
        q = [r_gal_kpc, 0.0, 0.0] * u.kpc
        omega2 = self.omega2(r_gal_kpc).to_value(u.s**-2)
        hess = np.reshape(pot.hessian(q).to_value(u.s**-2), (3, 3))
        T = -hess + omega2 * np.diag([1.0, 1.0, 0.0])
        # T is symmetric; eigvalsh returns ascending eigenvalues.
        return np.linalg.eigvalsh(T)[-1] / u.s**2

    def jacobi_escape_host(self):
        """``(phi_host, mass_host, m_host_total)`` for ``jacobi_escape``.

        That module is written against a bare spherical pair -- Phi(r) in
        (km/s)^2 and M(<r) in Msun, both over r in kpc -- rather than against
        gala, so that its Lagrange-point and A machinery works for any host and
        hence any nu.  This adapter supplies the pair for *this* host, so that
        ``--host kepler`` gets a Keplerian A and not a Milky Way one.

        M(<r) is taken as r^2 Phi'(r)/G rather than from gala's own
        ``mass_enclosed``, which guarantees it is the exact companion of the
        Phi being returned: the collinear solves balance forces from the mass
        while the energies come from the potential, and a mismatched pair would
        put a small inconsistency inside a difference of two numbers of order
        1e5.

        The third element is the mass that carries the host's *inertia*, which
        decides whether the host recoils and therefore whether A sits at L4/L5
        or at L3 -- an O(1) question, not a refinement.  See
        ``jacobi_escape.barycentre_fraction``.  Subclasses override
        ``m_host_total`` to answer it.
        """
        pot = self.gala_potential()

        def _column(r):
            """Flatten arbitrary-shaped radii into gala's (3, N) query array."""
            r = np.asarray(r, dtype=float)
            flat = np.maximum(np.ravel(np.atleast_1d(r)), 1e-12)
            q = np.zeros((3, flat.size)) * u.kpc
            q[0] = flat * u.kpc
            return r.shape, q

        def phi_host(r):
            shape, q = _column(r)
            return np.reshape(pot.energy(q).to_value((u.km / u.s)**2), shape)

        def mass_host(r):
            shape, q = _column(r)
            phi_prime = np.ravel(pot.gradient(q)[0])
            m = (phi_prime * q[0]**2 / G_apy).to_value(u.Msun)
            return np.reshape(m, shape)

        return phi_host, mass_host, self.m_host_total()

    def m_host_total(self):
        """Mass available to recoil, in Msun; ``None`` for "effectively infinite".

        The default is ``None``: an extended halo does not translate rigidly, so
        pinning it to the rotation axis is the right treatment regardless of how
        much mass it has.  ``KeplerHost`` overrides this, because a point mass
        has nowhere else to put the momentum.
        """
        return None


class KeplerHost(_GalaHost):
    """Point-mass (Keplerian) host: Phi(R) = -G M / R."""

    name = "kepler"
    nu_is_constant = True

    def __init__(self, m_galaxy=KEPLER_M_GALAXY):
        self.m_galaxy = u.Quantity(m_galaxy, u.Msun)

    def _build_gala_potential(self):
        return gp.KeplerPotential(m=self.m_galaxy, units=galactic)

    def omega2(self, r_gal_kpc):
        """Omega^2 = G M / R^3 [1/s^2], analytic (and a check on the base class)."""
        r = u.Quantity(r_gal_kpc, u.kpc)
        return (G_apy * self.m_galaxy / r**3).to(u.s**-2)

    def nu(self, r_gal_kpc):
        return 3.0

    def m_host_total(self):
        """M, because a point mass recoils.

        This is the case where the equilateral points are real: the host is the
        whole of the host's mass, it orbits the barycentre along with the
        cluster, and ``jacobi_escape.plane_potential_maximum`` returns L4 rather
        than L3.  Getting it wrong here is a ~1% error in E_crit, all of it in
        A.
        """
        return self.m_galaxy.to_value(u.Msun)


class SphericalMW2022Host(_GalaHost):
    """The spherical part of gala's MilkyWayPotential2022.

    That is the NFW halo (m = 5.5427e11 Msun, r_s = 15.626 kpc), the Hernquist
    bulge (5e9 Msun, c = 1 kpc) and the Hernquist nucleus (1.8142e9 Msun,
    c = 0.0689 kpc); the MN3 exponential disk is dropped as the only flattened
    component (see the module docstring).

    The components are read off the gala potential rather than re-declared here,
    so the parameters track whatever gala ships and cannot drift out of sync.
    Previously only the halo was used, which understates Phi' badly inside a few
    kpc: at R = 3 kpc the bulge and nucleus carry a large share of the enclosed
    mass, so both Omega^2 (hence epsilon) and nu were wrong there.
    """

    name = "mw2022_sph"
    nu_is_constant = False

    def _build_gala_potential(self):
        mw = gp.MilkyWayPotential2022(units=galactic)
        missing = [k for k in MW2022_FLATTENED_COMPONENTS if k not in mw]
        if missing:
            raise RuntimeError(
                f"MilkyWayPotential2022 has no component(s) {missing}: gala "
                f"{gp.__name__} has renamed or restructured them, and the "
                f"spherical/flattened split here needs rechecking.  Components "
                f"found: {sorted(mw.keys())}.")
        return gp.CCompositePotential(**{
            k: v
            for k, v in mw.items() if k not in MW2022_FLATTENED_COMPONENTS
        })


_HOSTS = {"kepler": KeplerHost, "mw2022_sph": SphericalMW2022Host}

# Runs generated before the bulge and nucleus were added recorded host = "nfw"
# in parameters.csv.  Resolving that to the current host keeps the analysis
# scripts running over those directories, but it is not the potential those ICs
# were built in, hence the warning.
_DEPRECATED_HOSTS = {"nfw": "mw2022_sph"}


def make_host(name):
    """Return a host-potential instance for name in {'kepler', 'mw2022_sph'}."""
    key = str(name).lower()
    if key in _DEPRECATED_HOSTS:
        replacement = _DEPRECATED_HOSTS[key]
        warnings.warn(
            f"host '{key}' is deprecated and resolves to '{replacement}': the "
            "NFW-only host has been replaced by all spherical components of "
            "MilkyWayPotential2022 (halo + bulge + nucleus).  Anything "
            f"recorded as '{key}' was built without the bulge and nucleus.",
            DeprecationWarning,
            stacklevel=2)
        key = replacement
    try:
        return _HOSTS[key]()
    except KeyError:
        raise ValueError(
            f"Unknown host '{name}'; choose from {sorted(_HOSTS)}.")
