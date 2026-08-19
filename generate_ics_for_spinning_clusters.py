#!/usr/bin/env python3
"""TIRO (Arditi & Varri 2025) initial-condition generator.

Samples phase-space ICs for rotating, tidally perturbed King models and writes
them in Henon units alongside measured rotation kinematics.

The degree of asynchronism zeta = omega^2/Omega^2 - 1 is set either directly
with --zeta (-1 is non-rotating, 0 is synchronous with the orbit, positive is
super-synchronous) or with --zeta_max, which spins the cluster as fast as it can
spin with its Jacobi surface still closed.  That limit is zeta_crit, the root of
E_J = 0 for a star at rest on the cluster boundary -- see ``critical_spin``. An explicit --zeta at or above zeta_crit is rejected: that surface is
open, so the model is unbound and its particles show super-escape Jacobi
energies at t = 0 through no fault of the escape diagnostic downstream. Pass
--allow_supercritical to sample one deliberately.

This supersedes generate_ics_at_fixed_zeta.py, a clone of this script that
carried only the --zeta half.
"""

import argparse
import warnings
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt

try:
    matplotlib.style.use("/home/btcook/Desktop/github_repositories/"
                         "krios_python/matplotlib_style_file")
except OSError:
    pass

import numpy as np
import pandas as pd
from numba import njit, prange
from scipy.integrate import cumulative_trapezoid, dblquad, tplquad
from scipy.spatial import cKDTree
from scipy.special import lambertw
from astropy import units as u
from astropy.constants import G as G_apy

import tiro_path  # noqa: F401  (puts the TIRO submodule on sys.path)
from TIRO import Model
from critical_spin import (boundary_jacobi_energy, critical_zeta, l1_radius,
                           tidal_strength)
from host_potential import make_host

_DEFAULT_M_GC = 2.0e5
_DEFAULT_RHO_0 = 1e3
_DEFAULT_R_GAL = 10.0
_DEFAULT_PSI = 3
# Host galactic potential: "kepler" (point mass, nu=3) or "mw2022_sph" (the
# spherical components of MilkyWayPotential2022 at r_gal). The host sets both
# nu and Omega^2 (hence epsilon).
_DEFAULT_HOST = "mw2022_sph"
# Degree of asynchronism, zeta = omega^2/Omega^2 - 1, when --zeta_max is not
# used. -1 is the non-rotating case (omega = 0), 0 is synchronous rotation.
_DEFAULT_ZETA = -1.0
# --zeta_max uses zeta = _ZETA_SAFETY * zeta_crit. Exactly at zeta_crit the
# psi = 0 surface touches the saddle, external_gradient has a double root, and
# assert_jacobi_surface_closed cannot confirm the surface is closed; backing off
# by a fraction of a percent keeps the model maximally spinning *and* verifiably
# bound. Set to 1.0 to sit exactly on the critical curve (the check will fail).
_ZETA_SAFETY = 0.99


@njit(parallel=True, fastmath=True)
def calc_pe_numba(pos, masses):
    """Sum all N*(N-1)/2 unique gravitational PE contributions (G = 1 units)."""
    n = pos.shape[0]
    total_pe = 0.0
    for i in prange(n):
        local_pe = 0.0
        for j in range(i + 1, n):
            dx = pos[i, 0] - pos[j, 0]
            dy = pos[i, 1] - pos[j, 1]
            dz = pos[i, 2] - pos[j, 2]
            inv_dist = 1.0 / np.sqrt(dx * dx + dy * dy + dz * dz)
            local_pe -= masses[i] * masses[j] * inv_dist
        total_pe += local_pe
    return total_pe


@njit(fastmath=True)
def calc_external_pe_numba(pos, masses, epsilon, zeta, nu):
    """Per-particle sum of TIRO external perturbation potential m_i * a_pert(r_i).

    a_pert = (9*eps/2) * [-zeta*(x^2+y^2) + z^2 - nu*x^2], matching NBody.py's
    potential_integrand convention (tidal + asynchronous centrifugal piece in
    the co-rotating frame).
    """
    n = pos.shape[0]
    total = 0.0
    coeff = 4.5 * epsilon
    for i in range(n):
        x = pos[i, 0]
        y = pos[i, 1]
        z = pos[i, 2]
        a_pert = coeff * (-zeta * (x * x + y * y) + z * z - nu * x * x)
        total += masses[i] * a_pert
    return total


def spherical_to_cartesian(r, theta, phi):
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)
    return np.array([x, y, z])


def assert_jacobi_surface_closed(model, zeta):
    """Return the model's tidal radius r_L1, raising if its Jacobi surface is open.

    The test is the Jacobi energy of a star at rest on the cluster's own
    boundary (:func:`critical_spin.boundary_jacobi_energy`): negative means the
    zero-velocity surface closes around the cluster and every star in the model
    is bound, positive means the psi = 0 surface has opened past the L1 saddle
    and the edge of the model is already outside it.

    This used to test whether ``Model.tidal_radius()`` raised.  That worked by
    accident -- its brentq bracket [r_trunc, 200] misses the saddle once fast
    rotation has pulled L1 inside r_trunc -- and it could only ever answer
    yes/no.  E_J answers with a number, and it is the same number
    :func:`critical_spin.critical_zeta` is the root of, so the check and the
    search cannot disagree about where zeta_crit is.
    """
    r_l1 = l1_radius(model)
    e_j = boundary_jacobi_energy(model, r_l1=r_l1)
    if e_j >= 0.0:
        raise ValueError(
            f"zeta = {zeta:g} is at or above zeta_crit for this "
            f"(psi, epsilon, nu): a star at rest on the cluster boundary has "
            f"E_J = {e_j:+.4f} >= 0, so the Jacobi surface is open, the model "
            f"is unbound and it is not a valid initial condition.  Pass "
            f"--allow_supercritical to sample it anyway.")
    return r_l1


class TiroInitialConditionSampler:
    """Sample TIRO ICs and rescale to Henon units."""

    def __init__(
        self,
        n_particles,
        psi_central=_DEFAULT_PSI,
        zeta=None,
        zeta_max=False,
        r_gal_kpc=_DEFAULT_R_GAL,
        m_gc_msun=_DEFAULT_M_GC,
        rho_0_msun_pc3=_DEFAULT_RHO_0,
        host=_DEFAULT_HOST,
        zeta_safety=_ZETA_SAFETY,
        allow_supercritical=False,
        verbose=True,
    ):
        self.n = n_particles
        self.psi_central = psi_central
        self.zeta_max = bool(zeta_max)
        self.zeta_safety = float(zeta_safety)
        self.allow_supercritical = bool(allow_supercritical)
        if self.zeta_max and zeta is not None:
            raise ValueError(
                f"pass either zeta (got {zeta}) or zeta_max, not both: "
                "zeta_max sets zeta from zeta_crit.")
        self.zeta_requested = None if self.zeta_max else (
            _DEFAULT_ZETA if zeta is None else float(zeta))
        # omega^2 = (1+zeta)*Omega^2, so zeta < -1 is unphysical.
        if self.zeta_requested is not None and self.zeta_requested < -1.0:
            raise ValueError(f"zeta must be >= -1 (got {self.zeta_requested}); "
                             "omega^2 = (1+zeta)*Omega^2 would be negative.")
        self.r_gal_kpc = r_gal_kpc
        self.m_gc_msun = m_gc_msun
        self.rho_0_msun_pc3 = rho_0_msun_pc3
        # Host potential sets both nu and Omega^2 (hence epsilon) at r_gal.
        self.host = make_host(host)
        self.host_name = self.host.name
        self.nu = self.host.nu(self.r_gal_kpc)
        self.verbose = verbose

        self.masses = np.ones(self.n) / self.n
        self.phys = {}

        if self.verbose:
            print("[1/4] Computing tidal strength parameter epsilon...")
        self._compute_epsilon()
        eps = self.phys["epsilon"]

        # zeta is either solved for (zeta_max: the fastest spin that keeps the
        # Jacobi surface closed) or taken as given and checked against that same
        # limit.  zeta = -1 is non-rotating, omega = sqrt(4*pi*eps*(1+zeta)) = 0.
        self.zeta_crit = None
        if self.zeta_max:
            if self.verbose:
                print("      Solving for zeta_crit at this epsilon...")
            self.zeta_crit = critical_zeta(self.psi_central, eps, self.nu)
            self.zeta = self.zeta_safety * self.zeta_crit
        else:
            self.zeta = self.zeta_requested
            self._reject_supercritical_zeta(eps)
        # chi = omega^2/(4 pi G rho_0) = epsilon*(1+zeta), from Arditi & Varri
        # (2025) equations (15), (16) and (23) (chi - eps = zeta*eps).
        self.chi = max(eps * (1.0 + self.zeta), 0.0)
        # Internal spin rate in units of sqrt(G*rho_0); omega^2 = 4*pi*chi
        # (NBody.py:21). This is the model quantity: it is what is written to
        # physical_units.csv and consumed downstream. It is NOT the rate to use
        # when adding omega x r to the sampled velocities -- those live in the
        # model's own velocity units, where the same rotation has rate
        # self.omega_vel_units below. See _sample_velocities for the derivation.
        self.omega = np.sqrt(4.0 * np.pi * self.chi)
        # The same rigid rotation expressed in the sampler's velocity units
        # (velocities in units of a^(-1/2), lengths in King radii r_0).
        self.omega_vel_units = 3.0 * np.sqrt(self.chi)

        tag = (f"tiro_ics"
               f"_N{self.n}_psi{self.psi_central}_zeta{self.zeta:.1f}"
               f"_host{self.host_name}_nu{self.nu:.3f}"
               f"_mgc{self.m_gc_msun:.0e}"
               f"_rho0{self.rho_0_msun_pc3:.0e}_rgal{self.r_gal_kpc}")
        self.out_dir = Path(tag)
        self.out_dir.mkdir(parents=True, exist_ok=True)

        if self.verbose:
            if self.zeta_max:
                mode = (f"maximally spinning "
                        f"(zeta = {self.zeta_safety:g} * zeta_crit)")
            elif self.zeta == -1.0:
                mode = "non-rotating (zeta = -1)"
            elif self.zeta == 0.0:
                mode = "synchronous (zeta = 0, omega = Omega)"
            else:
                mode = f"asynchronous (zeta = {self.zeta:g})"
            print(f"      Mode: {mode}")
            print(f"      host potential               = {self.host_name}")
            print(f"      nu (from host at r_gal)      = {self.nu:.4f}")
            print(f"      epsilon                      = {eps:.3e}")
            if self.zeta_crit is not None:
                print(f"      zeta_crit                    = "
                      f"{self.zeta_crit:.4f}")
            print(f"      zeta                         = {self.zeta:.4f}")
            print(f"      omega = sqrt(4*pi*eps*(1+zeta)) = {self.omega:.4f}")

        if self.verbose:
            print(f"[2/4] Integrating TIRO model "
                  f"(psi={self.psi_central}, eps={eps:.2e}, "
                  f"zeta={self.zeta:.4f}, nu={self.nu})...")
        self.model = Model([self.psi_central, eps, self.zeta, self.nu])
        self.model.integrate()

        # Refuse to sample a model whose Jacobi surface has opened: above
        # zeta_crit the particles are genuinely unbound at t = 0.
        self.r_tidal_dimless = None
        if self.allow_supercritical:
            try:
                self.r_tidal_dimless = assert_jacobi_surface_closed(
                    self.model, self.zeta)
            except ValueError as exc:
                warnings.warn(f"--allow_supercritical: {exc}")
        else:
            self.r_tidal_dimless = assert_jacobi_surface_closed(
                self.model, self.zeta)
        if self.verbose and self.r_tidal_dimless is not None:
            print(f"      Jacobi surface closed; r_tidal = "
                  f"{self.r_tidal_dimless:.4f} r_0 "
                  f"(r_trunc = {self.model.r_trunc:.4f} r_0)")

        if self.verbose:
            print("[3/4] Deriving physical units and enclosed mass scales...")
        self._compute_physical_units()

        if self.verbose:
            print("Initialisation complete.\n")

    def _reject_supercritical_zeta(self, eps):
        """Refuse a requested zeta at or above zeta_crit(psi, eps, nu).

        :func:`assert_jacobi_surface_closed` catches the same models further
        down, but only once the model has been integrated and without a number
        to compare against -- brentq failing to bracket a root is a thin way to
        tell a user their zeta was 3x too big.  Solving for zeta_crit up front
        lets the error say by how much.

        The solve is skipped entirely under ``allow_supercritical``: it is the
        expensive part of setup and nothing would be done with the answer.
        """
        if self.allow_supercritical:
            return
        if self.verbose:
            print(f"      Solving for zeta_crit to check zeta = "
                  f"{self.zeta:g}...")
        try:
            self.zeta_crit = critical_zeta(self.psi_central, eps, self.nu)
        except (ValueError, RuntimeError) as exc:
            # zeta_crit out of reach (every zeta >= -1 super-critical, or the
            # critical-curve solve did not converge).  The closed-surface check
            # on the integrated model is the ground truth anyway, so fall back
            # to it rather than refusing a zeta that may well be fine.
            warnings.warn(f"could not determine zeta_crit ({exc}); falling "
                          "back to the Jacobi-surface check on the model.")
            return
        if self.zeta >= self.zeta_crit:
            raise ValueError(
                f"zeta = {self.zeta:g} is at or above zeta_crit = "
                f"{self.zeta_crit:.4f} for psi = {self.psi_central:g}, "
                f"epsilon = {eps:.3e}, nu = {self.nu:.4f}: the Jacobi surface "
                f"is open, so the model is unbound and is not a valid initial "
                f"condition.  Use --zeta_max for the largest usable zeta, or "
                f"--allow_supercritical to sample this one anyway.")

    def _compute_epsilon(self):
        """epsilon = Omega^2/(4 pi G rho_0), Omega^2 = Phi'(R_gal)/R_gal."""
        self.phys["epsilon"] = float(
            tidal_strength(self.host, self.r_gal_kpc, self.rho_0_msun_pc3))

    def _compute_dimless_mass_rhm_sigma(self, n_r=80):
        """Total mass, half-mass radius, mass-weighted sigma via scipy.integrate.tplquad.

        Matches NBody.py: 8-fold octant symmetry, ψ from global_solution (full
        first-order correction), sigma clipped to 0 where ψ<=0.
        """
        r_trunc = self.model.r_trunc

        def _psi_pos(r, theta, phi):
            psi = float(self.model.global_solution(r, theta, phi))
            return psi if psi > 0.0 else 0.0

        def _mass_integrand(phi, theta, r):
            psi = _psi_pos(r, theta, phi)
            if psi <= 0.0:
                return 0.0
            return float(self.model.density(psi)) * r * r * np.sin(theta)

        def _sigsq_integrand(phi, theta, r):
            psi = _psi_pos(r, theta, phi)
            if psi <= 0.0:
                return 0.0
            sig = float(self.model.velocity_dispersion(psi))
            rho = float(self.model.density(psi))
            return sig * sig * rho * r * r * np.sin(theta)

        self.total_mass_dimless = 8.0 * tplquad(_mass_integrand, 0.0, r_trunc,
                                                0.0, 0.5 * np.pi, 0.0,
                                                0.5 * np.pi)[0]
        sigsq_times_M = 8.0 * tplquad(_sigsq_integrand, 0.0, r_trunc, 0.0,
                                      0.5 * np.pi, 0.0, 0.5 * np.pi)[0]
        self.sigma_global_dimless = float(
            np.sqrt(sigsq_times_M / self.total_mass_dimless))

        r_grid = np.linspace(0.0, r_trunc, n_r)
        dmdr = np.zeros(n_r)
        for i in range(1, n_r):
            r = r_grid[i]

            def _shell(phi, theta, _r=r):
                psi = _psi_pos(_r, theta, phi)
                if psi <= 0.0:
                    return 0.0
                return float(self.model.density(psi)) * np.sin(theta)

            shell = 8.0 * dblquad(_shell, 0.0, 0.5 * np.pi, 0.0,
                                  0.5 * np.pi)[0]
            dmdr[i] = r * r * shell
        M_cum = cumulative_trapezoid(dmdr, r_grid, initial=0.0)
        self.r_hm_dimless = float(
            np.interp(0.5 * self.total_mass_dimless, M_cum, r_grid))

    def _compute_physical_units(self):
        rho_0_phys = self.rho_0_msun_pc3 * u.Msun / u.pc**3
        m_gc = self.m_gc_msun * u.Msun
        dens_norm = rho_0_phys / self.model.density(self.model.param[0])

        if self.verbose:
            print("      -> Integrating total dimensionless mass...")
        self._compute_dimless_mass_rhm_sigma()

        r0_val = float(((m_gc / (dens_norm * self.total_mass_dimless)).to(
            u.pc**3)).value**(1.0 / 3.0))
        self.r_king = r0_val * u.pc
        self.r_hm = self.r_hm_dimless * self.r_king

        rho_si = rho_0_phys.to(u.kg / u.m**3)
        self.r_king_m = self.r_king.to(u.m)
        vel_scale_si = (9.0 /
                        (4.0 * np.pi * G_apy * rho_si * self.r_king_m**2)).to(
                            u.s**2 / u.m**2)
        vel_scale_inv_kms = vel_scale_si.to(u.s**2 / u.km**2)

        sigma_hat_sq = self.model.velocity_dispersion(self.model.param[0])**2
        sigma_0 = np.sqrt(sigma_hat_sq / vel_scale_si).to(u.km / u.s)
        sigma_hat_0 = np.sqrt(sigma_hat_sq)
        self.sigma_global = (self.sigma_global_dimless / sigma_hat_0) * sigma_0

        # NOTE: this is the TRUNCATION radius (where psi = 0 along the
        # spherical zeroth-order solution), not the tidal/Jacobi radius, which
        # would be model.tidal_radius(). It is exported below under both the
        # correctly named key "r_trunc_pc" and the historical (misnamed) key
        # "r_tidal_pc"; the latter is kept because it is read by
        # ../krios_python/python_scripts/get_v_phi_plots.py and by
        # collate_ics_table.py, so renaming it would break both.
        r_trunc_phys = self.model.r_trunc * self.r_king
        m_gc_phys = (dens_norm * self.total_mass_dimless * self.r_king**3).to(
            u.Msun)

        self.phys.update({
            "r_king_pc":
            self.r_king.value,
            "r_trunc_pc":
            r_trunc_phys.value,
            # Same value as r_trunc_pc; misnamed but kept for consumers.
            "r_tidal_pc":
            r_trunc_phys.value,
            "log10_rt_div_r_king":
            np.log10(r_trunc_phys.value / self.r_king.value),
            "sigma_0_kms":
            sigma_0.value,
            "rho_0_msun_pc3":
            self.rho_0_msun_pc3,
            "dens_norm_msun_pc3":
            dens_norm.value,
            "vel_scale_inv_kms":
            vel_scale_inv_kms.value,
            "r_hm_pc":
            self.r_hm.value,
            "m_gc_phys_msun":
            m_gc_phys.value,
            "omega":
            self.omega,
        })

    def _get_psi(self, r, theta, phi):
        return float(
            np.clip(self.model.global_solution(r, theta, phi), 0.0,
                    self.psi_central))

    def _get_rho(self, psi_arr):
        return np.clip(self.model.density(psi_arr), 0.0,
                       self.model.density(self.psi_central))

    def _get_sigma(self, psi_arr):
        return np.clip(
            self.model.velocity_dispersion(psi_arr),
            0.0,
            self.model.velocity_dispersion(self.psi_central),
        )

    def _prepare_shells(self):
        n_shells = max(int(np.ceil(1e-3 * self.n)), 10)
        r_max = self.model.boundary(np.pi / 2.0, 0.0)
        r_edges = np.geomspace(1e-5 * r_max, r_max, n_shells + 1)
        shell_vols = (4.0 / 3.0) * np.pi * (r_edges[1:]**3 - r_edges[:-1]**3)
        r_mid = 0.5 * (r_edges[:-1] + r_edges[1:])

        rho_max_in_shell = np.zeros(n_shells)
        rho_avg_in_shell = np.zeros(n_shells)
        for i, r in enumerate(r_mid):
            rho_samples = []
            rho_max = 0.0
            n_attempts = 0
            prev_avg = self._get_rho(
                np.array([self._get_psi(r, np.pi / 2.0, 0.0)]))[0]
            prev_max = prev_avg
            while n_attempts < 500:
                n_attempts += 1
                theta = np.arccos(self.rng.uniform(-1.0, 1.0))
                phi = self.rng.uniform(0.0, 2.0 * np.pi)
                psi = self._get_psi(r, theta, phi)
                rho = self._get_rho(np.array([psi]))[0]
                rho_samples.append(rho)
                rho_max = max(rho, rho_max)
                if n_attempts >= 30 and n_attempts % 10 == 0:
                    cur_avg = float(np.mean(rho_samples))
                    if (abs(cur_avg - prev_avg) / (prev_avg + 1e-30) < 1e-3
                            and abs(rho_max - prev_max) /
                        (prev_max + 1e-30) < 1e-3):
                        break
                    prev_avg = cur_avg
                    prev_max = rho_max
            rho_max_in_shell[i] = rho_max * 1.1
            rho_avg_in_shell[i] = float(np.mean(rho_samples))

        self.r_edges = r_edges
        self.rho_max_in_shell = rho_max_in_shell
        weights = rho_avg_in_shell * shell_vols
        self.shell_probs = weights / weights.sum()

    def generate_realization(self, seed):
        if self.verbose:
            print(f"--- Generating Realisation (seed={seed}) ---")
            print("  -> Building density rejection-sampling shells...")

        self.rng = np.random.default_rng(seed)

        parent_tag = (f"tiro_ics"
                      f"_N{self.n}_psi{self.psi_central}_zeta{self.zeta:.1f}"
                      f"_host{self.host_name}_nu{self.nu:.3f}"
                      f"_mgc{self.m_gc_msun:.0e}"
                      f"_rho0{self.rho_0_msun_pc3:.0e}_rgal{self.r_gal_kpc}")
        self.out_dir = Path(parent_tag) / f"seed{seed}"
        self.out_dir.mkdir(parents=True, exist_ok=True)

        self._prepare_shells()

        if self.verbose:
            print(f"  -> Rejection-sampling positions ({self.n} particles)...")
        self._sample_positions()

        if self.verbose:
            print("  -> Sampling velocities from distribution function...")
        self._sample_velocities()

        if self.verbose:
            print("  -> Rescaling to Henon units...")
        self._rescale_to_henon_units()

        if self.verbose:
            print("  -> Measuring rotation-over-dispersion kinematics...")

        kin = self._measure_rotation()

        if self.verbose:
            print(f"      <v_rot/sigma>  = {kin['rot_over_disp_mean']:+.4f} "
                  f"+/- {kin['rot_over_disp_mean_uncertainty']:.4f}")
            print(f"      stdev          =  {kin['rot_over_disp_stdev']:.4f} "
                  f"+/- {kin['rot_over_disp_stdev_uncertainty']:.4f}")

        self._plot_rotation_diagnostic(
            kin["rot_over_disp"],
            kin["rot_over_disp_mean"],
            kin["rot_over_disp_mean_uncertainty"],
            kin["rot_over_disp_stdev"],
            kin["rot_over_disp_stdev_uncertainty"],
        )

        kin["omega"] = self.omega
        self.kinematics = kin

    def _sample_positions(self):
        p_r = np.zeros(self.n)
        p_theta = np.zeros(self.n)
        p_phi = np.zeros(self.n)
        p_psi = np.zeros(self.n)
        resample = np.ones(self.n, dtype=bool)
        while np.any(resample):
            n_rem = int(resample.sum())
            idx = self.rng.choice(len(self.shell_probs),
                                  size=n_rem,
                                  p=self.shell_probs)
            r_in = self.r_edges[idx]
            r_out = self.r_edges[idx + 1]
            u_ = self.rng.random(n_rem)
            r_c = (r_in**3 + u_ * (r_out**3 - r_in**3))**(1.0 / 3.0)
            cos_t = self.rng.uniform(-1.0, 1.0, n_rem)
            theta_c = np.arccos(cos_t)
            phi_c = self.rng.uniform(0.0, 2.0 * np.pi, n_rem)
            psi_c = np.array([
                self._get_psi(r, t, p) for r, t, p in zip(r_c, theta_c, phi_c)
            ])
            rho_c = self._get_rho(psi_c)
            accept = self.rng.random(n_rem) < (rho_c /
                                               self.rho_max_in_shell[idx])
            indices = np.where(resample)[0][accept]
            p_r[indices] = r_c[accept]
            p_theta[indices] = theta_c[accept]
            p_phi[indices] = phi_c[accept]
            self.p_psi = p_psi
            self.p_psi[indices] = psi_c[accept]
            resample[indices] = False
        self.pos = spherical_to_cartesian(p_r, p_theta, p_phi).T

    def _sample_velocities(self):
        # King DF weight: v^2 * [exp(psi - v^2/2) - 1], v_esc = sqrt(2*psi).
        v_esc = np.sqrt(np.maximum(2.0 * self.p_psi, 0.0))
        v_sq_star = 2.0 * (1.0 - lambertw(np.exp(1.0 - self.p_psi)).real)
        f_max = v_sq_star * (
            np.exp(np.maximum(self.p_psi - 0.5 * v_sq_star, 0.0)) - 1.0)

        v_final = np.zeros(self.n)
        resample = np.ones(self.n, dtype=bool)
        while np.any(resample):
            n_rem = int(resample.sum())
            v_c = self.rng.random(n_rem) * v_esc[resample]
            f_val = v_c**2 * (np.exp(self.p_psi[resample] - 0.5 * v_c**2) -
                              1.0)
            accept = self.rng.random(n_rem) < (f_val / f_max[resample])
            indices = np.where(resample)[0][accept]
            v_final[indices] = v_c[accept]
            resample[indices] = False

        cos_t = self.rng.uniform(-1.0, 1.0, self.n)
        theta = np.arccos(cos_t)
        phi = self.rng.uniform(0.0, 2.0 * np.pi, self.n)
        vx_cr = v_final * np.sin(theta) * np.cos(phi)
        vy_cr = v_final * np.sin(theta) * np.sin(phi)
        vz_cr = v_final * np.cos(theta)

        # Co-rotating (alpha, beta, z) -> inertial frame at t=0: add omega x r,
        # with omega along +z, so v_x -= omega*y and v_y += omega*x.
        #
        # The rate here must be in the SAME units as the sampled velocities,
        # which are in Arditi & Varri's (2025) a^(-1/2) units (the DF above is
        # sampled with v_esc = sqrt(2*psi)), while self.pos is in King radii
        # r_0. Their Section 2.3 sets r_0 = (9 / (4 pi a G rho_0))^(1/2), and
        # their equations (15)/(23) give chi = omega^2/(4 pi G rho_0) =
        # epsilon*(1+zeta), so a rigid rotation at rate omega has speed
        #
        #   v_rot = omega * r_0 * sqrt(a) * R
        #         = sqrt(4 pi G rho_0 chi) * sqrt(9/(4 pi a G rho_0)) * sqrt(a) * R
        #         = 3 * sqrt(chi) * R.
        #
        # NOT sqrt(4*pi*chi)*R = self.omega*R, which is too large by
        # sqrt(4*pi/9) = 1.1816 (40% too much rotational kinetic energy) and
        # realises the model above omega_crit, opening the zero-velocity
        # surface. Cross-check: NBody.py's kinetic_integrand pairs
        # (9/(4*pi))*r^2*sin^2(theta)*omega^2 with 3*sigma^2, i.e. the same
        # 9/(4*pi) conversion from omega^2 to velocity units; its
        # potential_integrand carries the matching (9/(8*pi)).
        vel = np.zeros((self.n, 3))
        vel[:, 0] = vx_cr - self.omega_vel_units * self.pos[:, 1]
        vel[:, 1] = vy_cr + self.omega_vel_units * self.pos[:, 0]
        vel[:, 2] = vz_cr
        self.vel = vel

    def _rescale_to_henon_units(self):
        # Self-gravity PE (per-pair ~ 1/r, so rescales as 1/A under pos *= A).
        u_s = -float(calc_pe_numba(self.pos, self.masses))
        # External tidal+centrifugal PE (a_pert ~ r^2, so rescales as A^2).
        u_e = float(
            calc_external_pe_numba(self.pos, self.masses, self.phys["epsilon"],
                                   self.zeta, self.nu))
        # Require U_total(new) = -u_s/A + u_e*A^2 = -1/2, i.e.
        #     u_e*A^3 + (1/2)*A - u_s = 0.
        # Select the root nearest A = 2*u_s (the u_e -> 0 self-gravity limit).
        if abs(u_e) < 1e-14 * max(u_s, 1.0):
            A = 2.0 * u_s
        else:
            roots = np.roots([u_e, 0.0, 0.5, -u_s])
            real_pos = [
                r.real for r in roots
                if abs(r.imag) < 1e-10 * max(abs(r.real), 1.0) and r.real > 0.0
            ]
            if not real_pos:
                warnings.warn(
                    "External PE too strong for Henon rescaling: no positive "
                    "real root. Falling back to self-gravity-only convention.")
                A = 2.0 * u_s
            else:
                A = min(real_pos, key=lambda x: abs(x - 2.0 * u_s))

        self.pos *= A
        ke = 0.5 * np.sum(self.masses * np.sum(self.vel**2, axis=1))
        self.vel /= np.sqrt(4.0 * ke)

    def _measure_rotation(self):
        X, Y, _ = self.pos.T
        v_X, v_Y, _ = self.vel.T

        R_proj = np.maximum(np.sqrt(X**2 + Y**2), 1e-30)
        omega = (X * v_Y - Y * v_X) / R_proj**2

        k_nn = 30
        tree = cKDTree(self.pos)
        _, idx = tree.query(self.pos, k=k_nn + 1)
        idx = idx[:, 1:]

        sigma_local = np.empty(len(self.pos))
        for i in range(len(self.pos)):
            nbr = idx[i]
            dv = self.vel[nbr] - np.mean(self.vel[nbr], axis=0)
            sigma_local[i] = np.sqrt(np.mean(dv**2))
        sigma_local = np.maximum(sigma_local, 1e-30)

        rot_over_disp = R_proj * omega / sigma_local

        # Normal-Inverse-Chi-Squared conjugate update. "Uninformed" prior N(0, 1)
        # enters as one pseudo-observation: mu_0=0, sigma_0^2=1, kappa_0=nu_0=1.
        n = rot_over_disp.size
        x_bar = float(np.mean(rot_over_disp))
        s_sq = float(np.var(rot_over_disp, ddof=1))
        mu_0, sigma_0_sq, kappa_0, nu_0 = 0.0, 1.0, 1.0, 1.0

        kappa_n = kappa_0 + n
        nu_n = nu_0 + n
        mu_n = (kappa_0 * mu_0 + n * x_bar) / kappa_n
        sigma_n_sq = (nu_0 * sigma_0_sq + (n - 1) * s_sq +
                      (kappa_0 * n / kappa_n) * (x_bar - mu_0)**2) / nu_n

        # Marginal posterior on mu: t_{nu_n}(mu_n, sigma_n^2 / kappa_n).
        mean_unc = np.sqrt(sigma_n_sq / kappa_n * nu_n / (nu_n - 2.0))
        # Marginal posterior on sigma^2: Inv-Chi2(nu_n, sigma_n^2); delta method
        # on sqrt gives Var[sigma] ~ sigma_n^2 / (2 nu_n).
        stdev = float(np.sqrt(sigma_n_sq))
        stdev_unc = float(np.sqrt(sigma_n_sq / (2.0 * nu_n)))

        return {
            "rot_over_disp": rot_over_disp,
            "rot_over_disp_mean": float(mu_n),
            "rot_over_disp_mean_uncertainty": float(mean_unc),
            "rot_over_disp_stdev": stdev,
            "rot_over_disp_stdev_uncertainty": stdev_unc,
        }

    def _plot_rotation_diagnostic(self, rot_over_disp, mean, mean_unc, stdev,
                                  stdev_unc):
        fig, ax = plt.subplots(1, 1, figsize=(5, 4), constrained_layout=True)

        ax.hist(rot_over_disp, bins=200, histtype="step", color="k", lw=1.0)
        ax.axvline(mean, color="red", lw=2.0)

        ax.set_xlabel(r"$r_{\perp}\,\omega\,/\,\sigma_{\rm local}$",
                      fontsize=16)
        ax.set_ylabel(r"$N_{\mathrm{particles}}$", fontsize=16)
        ax.annotate(rf"$\mu = {mean:.4f} \pm {mean_unc:.4f}$",
                    xy=(0.7, 0.9),
                    xycoords="axes fraction",
                    fontsize=10)
        ax.annotate(rf"$\sigma = {stdev:.4f} \pm {stdev_unc:.4f}$",
                    xy=(0.65, 0.85),
                    xycoords="axes fraction",
                    fontsize=11)
        ax.legend(fontsize=10)

        fig.savefig(self.out_dir / "rotation_diagnostic.png",
                    dpi=500,
                    bbox_inches="tight")
        plt.close(fig)

    def save_realization(self, seed):
        if self.verbose:
            print(f"  -> Writing output to {self.out_dir}/")

        pd.DataFrame([{
            "n_particles": self.n,
            "psi_central": self.psi_central,
            "zeta": self.zeta,
            "zeta_crit": self.zeta_crit,
            "host": self.host_name,
            "nu": self.nu,
            "r_gal_kpc": self.r_gal_kpc,
            "m_gc_msun": self.m_gc_msun,
            "rho_0_msun_pc3": self.rho_0_msun_pc3,
            "seed": seed,
        }]).to_csv(self.out_dir / "parameters.csv", index=False)

        pd.DataFrame([self.phys]).to_csv(self.out_dir / "physical_units.csv",
                                         index=False)

        df_particles = pd.DataFrame({
            "id": np.arange(1, self.n + 1),
            "k": 0.0,
            "m": 1.0 / self.n,
            "Reff": 0.0,
            "x": self.pos[:, 0],
            "y": self.pos[:, 1],
            "z": self.pos[:, 2],
            "vx": self.vel[:, 0],
            "vy": self.vel[:, 1],
            "vz": self.vel[:, 2],
            "id2": 0.0,
            "k2": 0.0,
            "m2": 0.0,
            "Reff2": 0.0,
            "a": 0.0,
            "e": 0.0,
        })

        np.savetxt(
            self.out_dir / "particles.csv",
            df_particles.values,
            delimiter=", ",
            fmt="%.12f",
        )


def main():
    parser = argparse.ArgumentParser(
        description="TIRO IC generator -- Arditi & Varri (2025)")
    parser.add_argument("N", type=int, help="Number of particles")
    parser.add_argument("--psi", type=float, default=_DEFAULT_PSI)
    zeta_group = parser.add_mutually_exclusive_group()
    zeta_group.add_argument("--zeta",
                            type=float,
                            default=None,
                            help="Degree of asynchronism, zeta = "
                            "omega^2/Omega^2 - 1. 0 is synchronous rotation, "
                            f"-1 is non-rotating (default {_DEFAULT_ZETA:g}). "
                            "Must be >= -1 and below zeta_crit, or the model "
                            "is unbound; --allow_supercritical overrides.")
    zeta_group.add_argument("--zeta_max",
                            action="store_true",
                            help="Solve for zeta_crit at this (psi, epsilon, "
                            "nu) and use zeta = zeta_safety * zeta_crit -- the "
                            "fastest spin whose Jacobi surface is still "
                            "closed.")
    parser.add_argument("--zeta_safety",
                        type=float,
                        default=_ZETA_SAFETY,
                        help="Fraction of zeta_crit used by --zeta_max "
                        f"(default {_ZETA_SAFETY}). 1.0 sits exactly on the "
                        "critical curve, where the closed-surface check cannot "
                        "succeed.")
    parser.add_argument("--allow_supercritical",
                        action="store_true",
                        help="Skip the zeta_crit check and downgrade the "
                        "open-Jacobi-surface error to a warning, sampling the "
                        "model anyway. Its particles will include genuinely "
                        "unbound stars at t = 0.")
    parser.add_argument("--host",
                        choices=["kepler", "mw2022_sph"],
                        default=_DEFAULT_HOST,
                        help="Host galactic potential: 'kepler' (point mass, "
                        "nu=3) or 'mw2022_sph' (the spherical components of "
                        "MilkyWayPotential2022 -- NFW halo, Hernquist bulge "
                        "and nucleus; nu and Omega^2 evaluated at r_gal).")
    parser.add_argument("--M_GC", type=float, default=_DEFAULT_M_GC)
    parser.add_argument("--rho_0", type=float, default=_DEFAULT_RHO_0)
    parser.add_argument("--r_gal", type=float, default=_DEFAULT_R_GAL)
    parser.add_argument("--n_real", type=int, default=1)
    parser.add_argument("-q", "--quiet", action="store_false", dest="verbose")

    args = parser.parse_args()

    if args.verbose:
        print("\n=== TIRO IC Generator (Arditi & Varri 2025) ===\n")

    sampler = TiroInitialConditionSampler(
        n_particles=args.N,
        psi_central=args.psi,
        zeta=args.zeta,
        zeta_max=args.zeta_max,
        zeta_safety=args.zeta_safety,
        allow_supercritical=args.allow_supercritical,
        host=args.host,
        m_gc_msun=args.M_GC,
        rho_0_msun_pc3=args.rho_0,
        r_gal_kpc=args.r_gal,
        verbose=args.verbose,
    )

    for i in range(args.n_real):
        seed = 42 + i * 100
        sampler.generate_realization(seed)

        lz = (sampler.pos[:, 0] * sampler.vel[:, 1] -
              sampler.pos[:, 1] * sampler.vel[:, 0])

        if args.verbose:
            k = sampler.kinematics
            print(f"  Diagnostics (seed={seed}):")
            print(f"    mean_Lz/particle              = {np.mean(lz):.4e}  "
                  "(Henon units)")
            print(f"    omega                         = {sampler.omega:.4f}")
            print(f"    <v_rot/sigma>                 = "
                  f"{k['rot_over_disp_mean']:+.4f} +/- "
                  f"{k['rot_over_disp_mean_uncertainty']:.4f}")
            print(f"    stdev(v_rot/sigma)            =  "
                  f"{k['rot_over_disp_stdev']:.4f} +/- "
                  f"{k['rot_over_disp_stdev_uncertainty']:.4f}")

        sampler.save_realization(seed)

    if args.verbose:
        print("\nDone.")


if __name__ == "__main__":
    main()
