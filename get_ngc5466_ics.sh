# TIRO initial conditions for the NGC 5466 KRIOS runs (KRIOS/src/ngc5466_config_files and
# ngc5466_tuned_config_files): one non-spinning control (zeta = -1) and one maximally spinning
# model (--zeta_max = 0.99 zeta_crit) per orbit, one realisation each (seed 42).
#
# --host mw2022_sph here sets only the two numbers TIRO takes from a host, the circular frequency
# Omega and the tidal anisotropy nu at --r_gal, which is why the sampler needs a spherical host. The
# KRIOS runs themselves orbit in the FULL MilkyWayPotential2022 (HOST_POTENTIAL_TO_USE=1, disk
# included) on NGC 5466's own eccentric, inclined orbit. r_gal is therefore not where the cluster
# sits: it is the radius whose circular frequency in the spherical part equals the orbit-averaged
# <Omega_c^2>^1/2 of the real orbit over the radial period containing t = -5 Gyr. nu at that radius
# (1.83 and 1.77) is 11-12% below the orbit's effective value, a mismatch the generator cannot take.
#
# M_GC and rho_0 are the planner's: M_GC is the t = -5 Gyr mass that the KRIOS mass-loss law brings
# to the catalogue's present-day 5.61e4 Msun along each orbit, and rho_0 makes the psi = 3 model's
# r_hm equal the planned r_hm0 at that mass (14.15 and 14.83 pc). Both numbers are per orbit; do not
# reuse one set's values on the other. N = 150000 so the present-day t_rh matches the catalogue's.
# Every number comes from spinning_gcs_paper/ngc5466_krios/plan_ngc5466_ics.py and
# plan_ngc5466_tuned_ics.py (outputs/ngc5466_plan.txt, outputs/tuned/ngc5466_stream_tuned_plan.txt).
#
# Expected output directories (the generator names them from its arguments; the spinning one carries
# 0.99 zeta_crit to one decimal) must match ICS_FILENAME in the inis, under
# /proj/rodriguezlab/users/brian/KRIOS/data/ngc5466_tiro_ics_subdir/ on Longleaf:
#   catalogue orbit: tiro_ics_N150000_psi3.0_zeta-1.0_hostmw2022_sph_nu1.833_mgc5.95e+04_rho08e+00_rgal19.0897
#                    tiro_ics_N150000_psi3.0_zeta4.4_hostmw2022_sph_nu1.833_mgc5.95e+04_rho08e+00_rgal19.0897
#   tuned orbit:     tiro_ics_N150000_psi3.0_zeta-1.0_hostmw2022_sph_nu1.766_mgc6.55e+04_rho08e+00_rgal15.3205
#                    tiro_ics_N150000_psi3.0_zeta2.2_hostmw2022_sph_nu1.766_mgc6.55e+04_rho08e+00_rgal15.3205
# Check physical_units.csv afterwards: r_hm_pc 14.15 / 14.25 (catalogue control / spinning) and
# 14.83 / 14.92 (tuned), m_gc_phys_msun equal to M_GC.
#
# Run with the krios_env python, one command at a time (the Mac tolerates ~2 workers, 1 BLAS thread):
#   OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 bash get_ngc5466_ics.sh

N=150000
PSI=3

# --- catalogue orbit (Vasiliev & Baumgardt 2021 coordinates; KRIOS/src/ngc5466_config_files) ---
# already generated 2026-10-22
# python generate_ics_for_spinning_clusters.py $N --psi $PSI --host mw2022_sph --r_gal 19.0897 --M_GC 5.947e+04 --rho_0 8.066 --n_real 1 --zeta -1
# python generate_ics_for_spinning_clusters.py $N --psi $PSI --host mw2022_sph --r_gal 19.0897 --M_GC 5.947e+04 --rho_0 8.066 --n_real 1 --zeta_max

# --- stream-tuned orbit (fit_orbit_expanding.py, |phi1| < 25 deg; KRIOS/src/ngc5466_tuned_config_files) ---
python generate_ics_for_spinning_clusters.py $N --psi $PSI --host mw2022_sph --r_gal 15.3205 --M_GC 6.547e+04 --rho_0 7.704 --n_real 1 --zeta -1
python generate_ics_for_spinning_clusters.py $N --psi $PSI --host mw2022_sph --r_gal 15.3205 --M_GC 6.547e+04 --rho_0 7.704 --n_real 1 --zeta_max
